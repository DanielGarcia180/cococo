# LÍNEAS

Tycoon de redes de transporte en el navegador. Empiezas con un tranvía de pueblo en 1900 y
acabas con metro, alta velocidad, autobuses eléctricos y logística de mercancías que conecta
una región entera. Sin servidor, sin cuentas, sin micropagos: todo vive en tu navegador.

**Jugar:** abre `dist/index.html` con doble clic (Chrome, Edge o Firefox). No hace falta instalar nada.

## Cómo se juega

1. **Mira el mapa.** Pulsa `H` para ver el mapa de demanda: el ámbar marca barrios sin cobertura.
2. **Construye estaciones** (`2`) dentro de la ciudad grande, a 1–3 km unas de otras. El tipo de vía
   (carretera, ferrocarril, metro, alta velocidad) se elige en la barra de arriba a la izquierda o con `Tab`.
3. **Traza la vía** (`3`): clic en un nodo y clic en otro. Antes de confirmar ves longitud, coste,
   túneles, puentes y plazo de obra. El trazado busca solo el camino más barato por el terreno.
4. **Crea una línea** (`4` → "Nueva línea") y haz clic en las estaciones en orden. Las líneas van
   y vuelven (A→B→C→B→A) o son circulares.
5. **Compra vehículos** desde el panel de la línea. Llegan en unos días; puedes acelerar la entrega.
6. **Mira los números.** Cada línea tiene factor de carga, beneficio, ROI, periodo de recuperación,
   puntualidad y una previsión Monte Carlo (banda P10–P90). El panel Datos (`D`) compara líneas,
   ciudades e industrias y exporta todo a CSV.
7. **Reinvierte o invierte.** El fondo indexado (`I`) permite DCA automático con interés compuesto.
   La pregunta de siempre: ¿rinde más una línea nueva o el fondo?

El primer ingreso llega en menos de un minuto. Las primeras 5–10 horas pasan por cinco eras
(1900, 1950, 1990, 2030, 2060), 16 investigaciones, 67 vehículos coleccionables, 12 licencias
con medallas, contratos de ciudad, averías, recesiones y ciclos de combustible. Al superar
5 M€ de valor de compañía puedes **refundar** en otra región con bonos permanentes.

### Progreso mientras no juegas
Al cerrar la pestaña se guarda la partida. Al volver, la compañía habrá ganado la media diaria
de los últimos 30 días durante el tiempo ausente (al 70 %, hasta 8 horas) y las obras e
investigaciones habrán avanzado. Hay tres objetivos diarios y una racha.

### Escala
La simulación mueve vehículos a escala visual (un tranvía cruza 2 km en unos 15 s) mientras el
calendario avanza a 1 día cada 2 s a velocidad 1×. Cada "unidad" de pasajeros equivale a 100
viajeros mostrados; las tarifas de la ficha son por unidad (0,16 € por viajero y 2 km en 1900).

## Atajos de teclado

| Tecla | Acción |
|---|---|
| `Espacio` | Pausa |
| `1`, `+`, `−` | Velocidad 1× / subir / bajar (hasta 8×) |
| `Q` `2` `3` `4` `5` `X` | Seleccionar · Estación · Vía · Línea · Punto de paso · Demoler |
| `Tab` | Cambiar tipo de vía |
| `N` | Nueva línea |
| `H` | Mapa de demanda y cobertura |
| `Z` / `Y` | Deshacer / rehacer construcción |
| `F` `D` `R` `V` `I` `C` | Finanzas · Datos · Investigación · Vehículos · Fondo · Retos |
| `P` | Modo foto (hora del día, apertura, enfoque, filtros, captura PNG a 2× o 4×) |
| `Ctrl+S` | Guardar |
| `Esc` / clic derecho | Cancelar / cerrar |

Cámara: arrastrar para mover, botón derecho para orbitar, rueda para zoom.

## Semillas y reto semanal

Cada mapa sale de una semilla (número o texto). Dos personas con la misma semilla y el mismo
escenario ven el mismo mapa: las ciudades, las industrias y hasta la evolución del fondo.
La semilla se muestra en Ajustes → Acerca de.

El **reto semanal** (menú principal) usa la semilla de la semana ISO actual (`2026W42`). Objetivo:
máximo valor de compañía a los 15 años. Al terminar, el panel Retos da un código como
`LN-2026W42-18374221-K7Q` con una suma de verificación. Pásalo a tus amigos: el panel Retos
verifica cualquier código sin necesidad de servidor.

## Guardado

Autoguardado cada minuto (y al cerrar la pestaña) en `localStorage`. Cinco ranuras manuales en
Ajustes. Exportar/importar a archivo `.json` para copias o para mover la partida a otro ordenador.

## Mods (datos en JSON)

Ajustes → Mods → Importar mod. Un mod es un archivo `.json` con esta forma (todos los bloques
son opcionales); ejemplos en `mods/`:

```json
{
  "name": "Mi mod", "version": "1.0", "author": "yo",
  "vehicles": [
    { "id": "tram_luas", "name": "Luas Citadis 402", "mode": "tram", "track": "rail",
      "era": 2, "year": 2004, "price": 420000, "speed": 70, "capacity": 358,
      "accel": 1.1, "consumption": 22, "reliability": 0.95, "maintenance": 2600,
      "lifespan": 30, "unlock": "era_1990", "desc": "Tranvía de piso bajo." }
  ],
  "tech": [ { "id": "x", "name": "...", "era": 0, "cost": 30000, "days": 30, "req": [], "desc": "..." } ],
  "eras": [ { "index": 2, "year": 1990, "name": "...", "fare": 40, "cargoFare": 15, "fuel": 50 } ],
  "scenarios": [ { "id": "galicia", "name": "...", "desc": "...", "startCash": 300000, "startYear": 1900,
      "world": { "name": "Rías", "seed": 1981, "size": 128, "cell": 300, "coast": "coast", "relief": 1.1,
                 "industries": 8, "fixedCities": [ { "name": "Vigo", "x": 0.26, "y": 0.78, "pop": 120000 } ] } } ]
}
```

Campos de vehículo: `mode` ∈ `bus tram train metro hsr truck freight`; `track` ∈ `road rail metro hsr`;
`cargo: true` para mercancías; `capacity` en unidades de pasajeros (×100 viajeros) o toneladas;
`speed` km/h; `accel` m/s²; `consumption` L/100 km equivalentes; `reliability` 0–1; `maintenance` €/mes;
`unlock` id de investigación (o `null`). Si el `id` coincide con uno existente, lo sustituye.
Un mod inválido se rechaza con un mensaje que indica el campo que falta.

## Desarrollo

```
npm install
npm run dev        # servidor de desarrollo
npm test           # tests de simulación (vitest)
npm run build      # typecheck + build en dist/index.html (un solo archivo)
npm run balance    # bot de equilibrio: curva de 10 h en CSV (docs/BALANCE.md)
```

Arquitectura en `docs/DISEÑO.md`. La simulación (`src/sim`) es pura y determinista, sin DOM ni
Three.js: los tests y el bot la ejecutan en Node. Render en `src/render`, interfaz en `src/ui`.
