# Equilibrio

Curvas producidas por `npm run balance` (bot sencillo, velocidad media 3×, decisiones cada 20 s de
simulación). El bot: añade vehículos a líneas con factor de carga > 80 %, abre una estación y línea
nueva en la zona con más demanda sin cubrir cuando tiene caja, investiga cuando puede y, con caja
sobrante, activa DCA en el fondo. No usa carga ni tuning, así que un jugador humano debería ir
más rápido.

## Región aleatoria (semilla 2026), 10 h

| Hora | Año | Caja | Valor | Beneficio/mes | Líneas | Veh. | Estaciones | Pax/mes | Era | Rep. | Población |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 1900 | 131,484 | 228,242 | 0 | 1 | 1 | 2 | 0 | 0 | 50 | 184,231 |
| 1.00 | 1903 | 93,753 | 330,000 | 12,985 | 4 | 2 | 8 | 119,579 | 0 | 79 | 188,632 |
| 2.00 | 1907 | 222,691 | 907,948 | 173,851 | 6 | 4 | 12 | 121,354 | 0 | 100 | 194,080 |
| 3.00 | 1911 | 113,737 | 1,041,404 | 5,749 | 8 | 5 | 16 | 123,703 | 0 | 99 | 195,248 |
| 4.00 | 1914 | 289,667 | 1,360,517 | 2,992 | 9 | 6 | 18 | 122,515 | 0 | 100 | 197,668 |
| 5.00 | 1918 | 222,535 | 1,563,770 | 840 | 11 | 8 | 22 | 140,940 | 0 | 100 | 199,067 |
| 6.00 | 1922 | 409,221 | 1,663,972 | 1,396 | 11 | 8 | 22 | 138,618 | 0 | 100 | 201,792 |
| 7.00 | 1926 | 611,464 | 1,794,340 | 2,461 | 11 | 8 | 22 | 151,013 | 0 | 99 | 206,574 |
| 8.00 | 1929 | 660,497 | 2,116,443 | 4,729 | 11 | 8 | 22 | 160,024 | 0 | 100 | 209,378 |
| 9.00 | 1933 | 228,166 | 2,550,821 | 477,502 | 11 | 8 | 22 | 175,540 | 1 | 100 | 214,828 |
| 10.00 | 1937 | 831,668 | 4,678,210 | 26,829 | 11 | 10 | 22 | 246,079 | 1 | 94 | 216,122 |

## Isla pequeña (tutorial), 6 h

| Hora | Año | Caja | Valor | Beneficio/mes | Líneas | Veh. | Estaciones | Pax/mes | Era | Rep. | Población |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 1900 | 58,998 | 156,999 | 0 | 1 | 1 | 2 | 0 | 0 | 50 | 92,084 |
| 0.50 | 1901 | 105,909 | 252,595 | 7,596 | 3 | 1 | 6 | 75,132 | 0 | 80 | 92,837 |
| 1.00 | 1903 | 185,408 | 382,727 | 7,586 | 5 | 1 | 10 | 72,170 | 0 | 87 | 94,434 |
| 1.50 | 1905 | 81,731 | 272,019 | 10,171 | 5 | 1 | 10 | 94,368 | 0 | 59 | 96,254 |
| 2.00 | 1907 | 245,717 | 499,807 | 13,149 | 5 | 2 | 10 | 120,204 | 0 | 69 | 98,597 |
| 2.50 | 1909 | 561,821 | 801,849 | 14,898 | 5 | 2 | 10 | 129,331 | 0 | 87 | 101,250 |
| 3.00 | 1911 | 697,023 | 1,108,816 | 9,004 | 5 | 3 | 10 | 159,146 | 0 | 99 | 104,472 |
| 3.50 | 1913 | 565,396 | 1,509,448 | 21,087 | 5 | 5 | 10 | 185,136 | 0 | 98 | 107,877 |
| 4.00 | 1914 | 774,677 | 1,824,889 | 24,337 | 5 | 5 | 10 | 231,585 | 0 | 100 | 111,811 |
| 4.50 | 1916 | 1,007,676 | 2,633,315 | 4,394 | 5 | 4 | 10 | 238,415 | 0 | 100 | 116,574 |
| 5.00 | 1918 | 1,122,484 | 3,312,210 | 151,359 | 5 | 5 | 10 | 192,636 | 0 | 100 | 121,708 |
| 5.50 | 1920 | 1,509,049 | 4,789,868 | 146,217 | 5 | 5 | 10 | 218,006 | 0 | 97 | 127,724 |
| 6.00 | 1922 | 1,702,374 | 5,941,304 | -2,003 | 5 | 8 | 10 | 321,754 | 0 | 95 | 134,160 |

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
  cadena de mercancías (mina → acería: 120 t/mes × 6 €/t × km). Umbral de prestigio (5 M€) hacia
  la hora 6–8 según escenario.
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
