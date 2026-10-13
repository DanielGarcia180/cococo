# LÍNEAS — Documento de diseño

Juego web de gestión de redes de transporte para un solo jugador. Sin servidor, sin
microtransacciones. TypeScript + Vite + Three.js. Toda la simulación es pura y
determinista (sin dependencias de render) y se puede ejecutar en Node para tests y
para el bot de equilibrio.

## 1. Arquitectura

```
src/
  sim/        simulación pura (sin DOM, sin Three). Determinista por semilla.
    rng.ts         mulberry32 + utilidades (gauss, pick, hash de cadenas)
    types.ts       modelos de datos del estado completo (serializable a JSON)
    world.ts       generación procedural: terreno, agua, ciudades, barrios, industrias
    scenarios.ts   escenarios prediseñados (Isla tutorial, Dublín Norte, España)
    pathfind.ts    A* sobre la rejilla para trazar vías con coste por terreno
    network.ts     estaciones, vías, grafo, rutas de líneas, tabla de transbordos
    demand.ts      modelo gravitatorio O-D, cobertura, generación de pasajeros y carga
    vehicles.ts    movimiento individual, paradas, carga/descarga, desgaste, averías
    economy.ts     libro mayor, préstamos, fondo indexado (DCA), ciclos, reputación
    tech.ts        eras e investigación
    progress.ts    obras con temporizador, progreso offline, objetivos diarios, racha
    prestige.ts    refundación y árbol permanente
    tuning.ts      mejoras por vehículo y libreas
    stats.ts       historial, KPIs, Monte Carlo, CSV
    challenges.ts  licencias (medallas) y retos semanales con semilla
    save.ts        serialización, ranuras, exportar/importar
    game.ts        clase Game: estado + tick(dt) + API de acciones (comandos)
    data/*.json    datos del juego (vehículos, eras, investigación, licencias)
  render/     capa Three.js: terreno, ciudades, vías, estaciones, vehículos instanciados,
              día/noche, clima, heatmap, modo foto
  ui/         DOM vanilla: HUD, paneles, gráficos canvas, i18n, atajos
  main.ts     bucle: paso fijo de simulación (20 Hz × velocidad), render a 60 fps
```

Reglas:
- `sim/` no importa nada de `render/` ni `ui/`. Solo tipos y datos.
- Todo el estado vive en `GameState` (JSON plano). Guardar = `JSON.stringify(state)`.
- La aleatoriedad usa exclusivamente `state.rng` (mulberry32). Misma semilla + mismas
  acciones ⇒ mismo estado byte a byte (test `determinism.test.ts`).
- El paso de simulación es fijo (`DT = 0.05 s`). La velocidad de juego multiplica el
  número de pasos por frame, nunca el tamaño del paso.

## 2. Unidades y escalas

| Magnitud | Unidad interna | Nota |
|---|---|---|
| Posición | metros (mundo) | rejilla de `size × size` celdas de `cell` m (por defecto 128 × 300 m = 38,4 km) |
| Altura | metros | 0 = nivel del mar |
| Tiempo calendario | días (float) | **1 día = 2 s reales a 1×** → 1 mes (30 d) = 1 min, 1 año = 12 min |
| Velocidad vehículos | km/h de ficha | se mueven a `kmh / 3,6 × VSCALE` m/s con `VSCALE = 20` (escala estilo Transport Fever: el calendario y el movimiento viven en escalas distintas) |
| Dinero | € | sin inflación general; las tarifas base suben por era |
| Pasajeros | unidades de 100 viajeros (`PAX_UNIT`) | la interfaz muestra viajeros; la simulación mueve unidades |

## 3. Mundo procedural

Ruido de valor fractal (4 octavas) con semilla → mapa de alturas. Nivel del mar
configurable por escenario (isla, costa, interior). Ríos: descenso por gradiente desde
picos aleatorios hasta el mar. Ciudades: `n` puntos con separación mínima, evitando agua,
preferencia por costa/río y zonas llanas. Cada ciudad tiene barrios (`zones`) de tipo
residencial / comercial / industrial dispuestos alrededor del centro con pesos que suman
la población. Industrias: cadenas `mina → acería → fábrica → ciudad (bienes)`,
`bosque → aserradero → ciudad`, `granja → ciudad (alimentos)`, `pozo → refinería → ciudad`.

## 4. Demanda

Modelo gravitatorio entre estaciones (no entre zonas, para mantener O(E²)):

```
P_s = Σ_zonas_res_en_radio  peso × población_ciudad × f(dist)     (producción)
A_d = Σ_zonas_(com|ind)_en_radio peso × población_ciudad × f(dist)  (atracción)
T_sd (pax/día) = K × P_s × A_d / (1 + (dist_sd / D0)²) × M_ciclo × M_reputación × M_precio × M_clima
```
- `f(dist)` = cobertura lineal decreciente hasta el radio de captación (800 m bus/tranvía,
  1500 m tren/metro, 4000 m alta velocidad).
- `M_precio = (1 / tarifa_rel)^1.2` (elasticidad −1,2).
- Solo se generan viajes si existe ruta en la red (tabla `nextHop[s][d]` por BFS sobre el
  grafo de paradas consecutivas). Los pasajeros esperan en `station.waiting[d]`; un 2 %/h
  se va si nadie los recoge (penaliza reputación).
- Carga: cada industria produce `rate` toneladas/mes de su `output` si tiene insumo (o sin
  insumo las primarias); se acumula en la estación de mercancías en su radio y busca
  destinos que acepten ese `cargo` (industrias o ciudades para bienes/alimentos).
- Crecimiento urbano mensual: `pop *= 1 + 0.004 × (servicio − 0.3)` donde `servicio` es la
  fracción de demanda potencial servida (0..1). Sin servicio, la ciudad decrece lentamente.

## 5. Construcción

Nodos (estaciones o puntos de paso) y vías entre nodos. La vía se traza por A* sobre
celdas con coste `base_kind × (1 + 6·pendiente) + agua ⇒ puente ×4 + pendiente > 0,25 ⇒
túnel ×6`. El metro ignora el terreno (siempre túnel, coste alto, recto). La traza se
suaviza con Chaikin (2 iteraciones). Coste y plazo se muestran antes de confirmar; el
plazo en días = `longitud_km / km_por_día_kind` + fijo. Deshacer/rehacer con pila de
operaciones inversas (solo construcción).

## 6. Líneas y vehículos

Línea = secuencia de paradas de un tipo de vía; el recorrido es ida y vuelta
(A→B→C→B→A) o circular. El trayecto se resuelve con Dijkstra sobre el grafo de vías y se
aplana en una polilínea con distancias acumuladas. Cada vehículo tiene `dist` a lo largo
del bucle, velocidad, estado (`moving|dwell|broken|delivery`), carga por destino, km,
desgaste (0..1), fiabilidad. Aceleración/frenado reales (`v = min(vmax, √(2·a·restante))`).
Avería: probabilidad por km = `(1 − fiabilidad) × (0,2 + desgaste) / 50 km`; parada de
0,5–2 días. Mantenimiento mensual reduce desgaste a cambio de coste.

## 7. Economía

- Ingreso por pasajero = `tarifa_base_era × (0,6 + 0,25 × km^0,75) × tarifa_rel × (1 + confort×0,05)`.
  (Sublineal con la distancia: una línea larga gana más por viaje, pero no tanto como para que
  el tren interurbano se amortice en tres meses.)
- Ingreso por tonelada = `tarifa_carga_era × (1 + 0,3 × km^0,8)`.
- Costes: por km recorrido (consumo × precio combustible), mantenimiento mensual por
  vehículo, mantenimiento de estaciones y vías por km, salarios por vehículo.
- Depreciación lineal 20 años (valor contable para el balance).
- Préstamos: hasta 3 tramos, interés anual 4–8 % según reputación, amortización mensual.
- Fondo indexado: NAV sigue un movimiento browniano geométrico (μ 6 %/año, σ 15 %) con RNG
  del estado. DCA automático: importe mensual configurable que se compra si hay caja.
- Ciclo económico: `M_ciclo = 1 + 0,15·sin(2π·año/9 + fase) + ruido AR(1)`; precio
  combustible análogo con su propia fase. Recesión (`M_ciclo < 0,9`) muestra aviso.
- Reputación 0..100: sube con puntualidad y demanda servida; baja con averías y
  pasajeros perdidos. Multiplica demanda `0,7 + 0,6 × rep/100`.
- Subvenciones: contratos de ciudad (servir X pax/mes a una ciudad durante N meses).
- Valor de compañía = caja + valor contable de vehículos + 50 % infraestructura + fondo −
  deuda. Es la métrica para prestigio y retos.

## 8. Eras y tecnología

Eras 1900, 1950, 1990, 2030, 2060. Cada nodo de investigación tiene coste (€) y plazo
(días), prerrequisitos y desbloqueos (vehículos, tipos de vía, funciones). Las eras se
investigan como nodos especiales disponibles a partir de `año_era − 10`. Investigación
acelerable con dinero (como las obras).

## 9. Progreso estilo Clash

Obras y entregas de vehículos con fecha de fin; se pueden acelerar pagando
`coste × 0,002 × días_restantes`. Al abrir el juego se calcula el tiempo real ausente
(máx. 8 h), se avanzan los días a 1× con eficiencia 70 % (mejorable por prestigio) usando
la media de beneficio diario de los últimos 30 días, se completan obras e investigación y
se muestra un resumen. Objetivos diarios (3, sembrados por fecha) con racha.

## 10. Prestigio

Al alcanzar un valor de compañía ≥ 5 M€ se puede "Refundar": nueva región (semilla
nueva o escenario), se pierden red y caja, se ganan puntos `floor(log2(valor/2,5M))`.
Árbol permanente: ingresos +4 %/pt, costes −3 %/pt, obras −8 %/pt, caja inicial +25 %/pt,
eficiencia offline +5 %/pt, investigación −8 %/pt. La colección y las licencias persisten.

## 11. Datos

Por línea y global, instantánea mensual: ingresos, costes, beneficio, pasajeros, factor de
carga medio, puntualidad (paradas dentro del intervalo objetivo), ROI acumulado, periodo
de recuperación. Monte Carlo: 500 trayectorias de 24 meses con `profit_m = μ + σ·ε`
estimados del historial (mínimo 3 meses; antes, σ por defecto 35 %), bandas P10/P50/P90.
CSV exportable con todo el historial.

## 12. Plan por hitos

1. Rebanada vertical: mundo, estación, vía, línea, vehículo, dinero, guardado.
2. Economía completa, préstamos, ciclos, reputación, carga.
3. Eras e investigación; catálogo de 60+ vehículos.
4. Datos: paneles, gráficos, Monte Carlo, CSV.
5. Progreso offline, obras con plazo, diarios, prestigio.
6. Tuning, libreas, colección, licencias, reto semanal.
7. Modo foto, día/noche, clima.
8. Modding JSON + mods de ejemplo, README, bot de equilibrio, pulido.
