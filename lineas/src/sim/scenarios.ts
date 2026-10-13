import type { WorldParams } from './world';

export interface Scenario {
  id: string; name: string; desc: string; startCash: number; startYear: number;
  world: Omit<WorldParams, 'seed'> & { seed?: number };
  tutorial?: boolean;
}

export const SCENARIOS: Scenario[] = [
  {
    id: 'isla', name: 'Isla pequeña (tutorial)',
    desc: 'Una isla con tres pueblos y una mina. Aprende a construir una línea rentable en cinco minutos.',
    startCash: 180000, startYear: 1900, tutorial: true,
    world: { name: 'Isla de Arán', seed: 7, size: 96, cell: 250, coast: 'island', relief: 0.8, cities: 3, industries: 3, popRange: [9000, 22000], bigPop: [28000, 40000] },
  },
  {
    id: 'dublin', name: 'Dublín Norte ↔ Aeropuerto ↔ Swords',
    desc: 'Un corredor de costa inspirado en MetroLink: conecta el centro, el aeropuerto y Swords antes de que el tráfico colapse.',
    startCash: 320000, startYear: 1900,
    world: {
      name: 'Dublín Norte', seed: 1916, size: 128, cell: 300, coast: 'coast', relief: 0.6, industries: 6,
      fixedCities: [
        { name: 'Dublín Centro', x: 0.42, y: 0.78, pop: 150000 },
        { name: 'Drumcondra', x: 0.40, y: 0.66, pop: 30000 },
        { name: 'Ballymun', x: 0.36, y: 0.55, pop: 24000 },
        { name: 'Aeropuerto', x: 0.46, y: 0.44, pop: 12000 },
        { name: 'Swords', x: 0.50, y: 0.28, pop: 45000 },
        { name: 'Malahide', x: 0.62, y: 0.34, pop: 18000 },
        { name: 'Howth', x: 0.70, y: 0.60, pop: 9000 },
        { name: 'Blanchardstown', x: 0.20, y: 0.60, pop: 40000 },
      ],
    },
  },
  {
    id: 'espana', name: 'Valencia ↔ Madrid ↔ Sevilla',
    desc: 'Tres capitales y una meseta por medio. El escenario para la alta velocidad y las mercancías a larga distancia.',
    startCash: 450000, startYear: 1900,
    world: {
      name: 'Península', seed: 1992, size: 160, cell: 350, coast: 'inland', relief: 1.4, industries: 10,
      fixedCities: [
        { name: 'Madrid', x: 0.45, y: 0.42, pop: 180000 },
        { name: 'Valencia', x: 0.86, y: 0.50, pop: 90000 },
        { name: 'Sevilla', x: 0.22, y: 0.84, pop: 80000 },
        { name: 'Toledo', x: 0.40, y: 0.54, pop: 22000 },
        { name: 'Cuenca', x: 0.62, y: 0.42, pop: 16000 },
        { name: 'Albacete', x: 0.66, y: 0.62, pop: 28000 },
        { name: 'Córdoba', x: 0.33, y: 0.72, pop: 40000 },
        { name: 'Ciudad Real', x: 0.42, y: 0.66, pop: 20000 },
        { name: 'Guadalajara', x: 0.54, y: 0.34, pop: 18000 },
        { name: 'Castellón', x: 0.90, y: 0.36, pop: 24000 },
      ],
    },
  },
  {
    id: 'random', name: 'Región aleatoria',
    desc: 'Mapa generado por semilla. Comparte la semilla para jugar el mismo mapa que otra persona.',
    startCash: 250000, startYear: 1900,
    world: { name: 'Región', size: 128, cell: 300, coast: 'coast', relief: 1, cities: 7 },
  },
];

export function getScenario(id: string): Scenario { return SCENARIOS.find(s => s.id === id) ?? SCENARIOS[0]; }
