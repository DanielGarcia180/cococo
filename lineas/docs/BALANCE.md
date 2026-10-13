# Equilibrio

Curvas producidas por `npm run balance` (bot sencillo, velocidad media 3×, decisiones cada 20 s de
simulación). El bot: añade vehículos a líneas con factor de carga > 80 %, abre una estación y línea
nueva en la zona con más demanda sin cubrir cuando tiene caja, investiga cuando puede y, con caja
sobrante, activa DCA en el fondo. No usa carga ni tuning, así que un jugador humano debería ir
más rápido.

## Región aleatoria (semilla 2026), 10 h

| Hora | Año | Caja | Valor | Beneficio/mes | Líneas | Veh. | Estaciones | Pax/mes | Era | Rep. | Población |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 1900 | 122,321 | 223,661 | 0 | 1 | 1 | 2 | 0 | 0 | 50 | 235,985 |
| 1.00 | 1903 | 144,114 | 712,491 | 26,888 | 5 | 6 | 10 | 261,606 | 0 | 78 | 239,938 |
| 2.00 | 1907 | 324,577 | 1,520,361 | 33,963 | 9 | 11 | 18 | 388,357 | 0 | 76 | 252,453 |
| 3.00 | 1911 | 1,137,671 | 2,814,927 | 36,808 | 12 | 13 | 24 | 450,150 | 0 | 75 | 277,780 |
| 4.00 | 1914 | 2,173,945 | 4,823,849 | 67,237 | 12 | 15 | 24 | 466,864 | 0 | 70 | 310,165 |
| 5.00 | 1918 | 3,362,684 | 6,747,186 | 604 | 12 | 17 | 24 | 626,192 | 0 | 69 | 347,849 |
| 6.00 | 1922 | 4,764,921 | 7,911,918 | -37,634 | 12 | 18 | 24 | 651,866 | 0 | 69 | 390,346 |
| 7.00 | 1926 | 6,665,942 | 11,185,462 | 135,450 | 12 | 23 | 24 | 941,996 | 0 | 74 | 439,514 |
| 8.00 | 1929 | 11,047,228 | 16,998,428 | 139,616 | 12 | 23 | 24 | 1,059,429 | 0 | 83 | 488,170 |
| 9.00 | 1933 | 18,887,491 | 28,279,232 | 826,788 | 12 | 23 | 24 | 1,091,023 | 1 | 70 | 539,592 |
| 10.00 | 1937 | 29,188,932 | 44,750,179 | 235,941 | 12 | 25 | 24 | 1,322,372 | 1 | 59 | 586,048 |

## Isla pequeña (tutorial), 6 h

| Hora | Año | Caja | Valor | Beneficio/mes | Líneas | Veh. | Estaciones | Pax/mes | Era | Rep. | Población |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 1900 | 58,998 | 156,999 | 0 | 1 | 1 | 2 | 0 | 0 | 50 | 92,084 |
| 0.50 | 1901 | 98,766 | 245,452 | 7,203 | 3 | 1 | 6 | 75,357 | 0 | 81 | 92,837 |
| 1.00 | 1903 | 172,533 | 443,601 | 8,732 | 5 | 2 | 10 | 102,312 | 0 | 92 | 94,384 |
| 1.50 | 1905 | 24,222 | 285,004 | 16,818 | 5 | 2 | 10 | 147,136 | 0 | 100 | 96,211 |
| 2.00 | 1907 | 381,347 | 698,458 | 14,251 | 5 | 3 | 10 | 139,371 | 0 | 100 | 98,552 |
| 2.50 | 1909 | 751,805 | 984,462 | 15,010 | 5 | 2 | 10 | 136,634 | 0 | 100 | 101,198 |
| 3.00 | 1911 | 592,955 | 1,198,682 | -756 | 5 | 3 | 10 | 151,927 | 0 | 100 | 104,413 |
| 3.50 | 1913 | 441,904 | 1,549,869 | 35,248 | 5 | 4 | 10 | 165,465 | 0 | 98 | 107,826 |
| 4.00 | 1914 | 694,658 | 2,244,948 | -124,780 | 5 | 4 | 10 | 238,131 | 0 | 100 | 111,844 |
| 4.50 | 1916 | 1,083,110 | 3,367,811 | -20,319 | 5 | 5 | 10 | 246,420 | 0 | 100 | 116,609 |
| 5.00 | 1918 | 1,391,980 | 3,382,478 | -51,401 | 5 | 4 | 10 | 233,473 | 0 | 100 | 121,742 |
| 5.50 | 1920 | 1,621,670 | 3,724,948 | 116 | 5 | 5 | 10 | 279,181 | 0 | 100 | 127,758 |
| 6.00 | 1922 | 1,957,850 | 4,670,663 | 20,995 | 5 | 7 | 10 | 349,003 | 0 | 99 | 134,197 |

## Lectura de la curva

- **Primeros 60 s**: una línea de tranvía de 2 paradas cuesta ≈ 90 k€ (dos estaciones a 12 k€,
  2 km de vía a ≈ 14–40 €/m según pendiente, un tranvía de 16,5 k€). El primer billete entra a los
  ~20 s. Con un tranvía rinde ≈ +1 k€/mes (1 minuto real); cada tranvía adicional en una línea
  saturada añade ≈ +1,5 k€/mes hasta agotar la demanda (de 5 a 10 tranvías en una ciudad grande).
  Esa es la decisión de "una más y cierro".
- **Horas 1–3**: segunda y tercera ciudad, señalización (nivel 2) y taller. El valor de compañía
  pasa de 0,2 a 1 M€. La reputación sube con puntualidad y cae con pasajeros perdidos: obliga a
  comprar vehículos antes que estaciones.
- **Horas 3–6**: era 1950 (400 k€ + 150 días), préstamos para saltar a tren entre ciudades, primera
  cadena de mercancías (mina → acería: 120 t/mes × 6 €/t × km^0,8). Umbral de prestigio (5 M€)
  hacia la hora 4–6 según escenario. Las ciudades servidas crecen (+150 % en 35 años en la región
  aleatoria), así que una red saturada sigue necesitando vehículos nuevos.
- **Horas 6–10**: era 1990 (1,8 M€), metro en la capital, alta velocidad (3,5 M€ + vía a 55 €/m) y
  fondo indexado como alternativa a expandirse (6,5 % anual frente a ROI de línea del 10–40 %).
- **Después**: eras 2030 y 2060, refundaciones (cada una +4 % ingresos/−3 % costes…), licencias
  oro (valor 50 M€, 1 M km de flota) y retos semanales.

## Decisiones que no tienen una respuesta única

| Decisión | Palancas que la hacen interesante |
|---|---|
| ¿Más vehículos o más estaciones? | Estaciones cuestan mantenimiento fijo (350 €/mes rail); vehículos cuestan personal (320 €/mes) y combustible. Con factor de carga < 60 % no compensa el vehículo. |
| ¿Subir la tarifa? | Elasticidad −1,2: ×1,5 de tarifa da ×0,61 de demanda. Compensa solo en líneas saturadas. |
| ¿Tren o tranvía entre ciudades? | Vía de ferrocarril 14 €/m con pendiente máxima 4 %; túneles a 160 €/m. En terreno accidentado la carretera (9 €/m, 12 %) con autobuses puede salir mejor. |
| ¿Metro? | 200 €/m pero ignora el terreno y tiene capacidad 30 000 pax/tren. Solo compensa con > 60 000 habitantes cubiertos. |
| ¿Fondo o línea? | El fondo rinde 6,5 % anual con volatilidad 16 %; una línea bien dimensionada el 10–40 % pero con averías, ciclo económico y combustible. |
| ¿Préstamo? | Interés 3–8 % según reputación; una línea saturada devuelve el 20 % anual. |
| ¿Acelerar obras? | 0,25 % del coste por día acelerado: barato para estaciones, caro para investigación de era. |
| ¿Tuning? | Motor +8 % velocidad pero +10 % desgaste (más averías); confort +5 % tarifa por +2 % consumo. |

## Constantes (src/sim/constants.ts)

`demandK` 4e-5 con exponente 0,7 sobre P×A; `demandD0` 2600 m; `waitDecay` 3 %/día;
`wearPerKm` 3,5e-4; `breakdownBase` 1/60 por km × (1−fiabilidad) × (0,2+desgaste);
`fundMu` 6,5 %, `fundSigma` 16 %; ciclo ±15 % con periodo 9 años; combustible ±30 % con periodo 6;
`cityGrowth` 0,6 %/mes × (servicio − 0,25), `cityDecline` 0,08 %/mes; `offlineEff` 70 %, máximo 8 h;
`prestigeThreshold` 5 M€. Tarifas por era: 16 / 24 / 40 / 60 / 80 € por unidad (100 viajeros).

## Cambios hechos a partir del bot

1. La primera versión daba 1 272 pax/día entre dos paradas y 0,4 € por pasajero: el tranvía nunca
   se amortizaba. Se pasó a unidades de 100 viajeros, exponente 0,7 y tarifa 16 €/unidad.
2. Vía de ferrocarril de 30 a 14 €/m y estaciones de 22 a 12 k€ para que la primera línea quepa
   en la caja inicial del tutorial (180 k€) y deje margen para 3–4 tranvías.
3. La puntualidad se medía contra un intervalo imposible (sin tiempo de parada ni aceleración) y
   hundía la reputación a 10 en un año. Ahora el intervalo objetivo incluye paradas y aceleración
   con tolerancia del 30 %.
4. Sin servicio, la reputación subía sola (+1/mes) y las ciudades perdían un 2,4 % anual. Ahora
   la reputación solo cambia con actividad y el declive es del 1 % anual.
5. Las aceleraciones de obra y las mejoras de tuning se contabilizaban como gasto y como
   inversión a la vez; ahora son inversión (no pasan por la cuenta de resultados).
6. La tarifa crecía linealmente con la distancia: un tren lleno de 15 km ganaba 20 k€/mes y se
   amortizaba en tres meses, y el bot llegaba a 53 M€ en 10 h. Con `km^0,75` el mismo tren gana
   ≈ 11 k€/mes sobre ≈ 300 k€ de inversión (vía + estaciones), unos 25 meses de recuperación.
7. La generación de demanda pasó de 1 ms a 0,1 ms por paso (se ejecuta cada 4 pasos y sin
   recalcular sumas por par): un paso con 300 vehículos cuesta 0,35 ms.
