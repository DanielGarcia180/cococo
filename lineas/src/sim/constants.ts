/** Constantes de equilibrio. Ver docs/BALANCE.md. */
export const SAVE_VERSION = 3;
export const DT = 0.05;                 // s por paso de simulación
export const SEC_PER_DAY = 2;           // segundos reales por día de juego a 1×
export const DAYS_PER_MONTH = 30;
export const VSCALE = 20;               // escala de velocidad de vehículos (ver DISEÑO §2)
export const PAX_UNIT = 100;            // 1 unidad de pasajeros simulada = 100 viajeros mostrados
export const START_YEAR_DEFAULT = 1900;

export const BAL = {
  demandK: 4e-5,              // pax/día = K × (P×A)^demandExp / (1 + (d/D0)²)
  demandExp: 0.7,
  demandD0: 2600,             // m, escala de distancia del modelo gravitatorio
  minTripDist: 500,           // m, no se generan viajes más cortos
  waitDecay: 0.03,            // fracción de pasajeros que se va por día de espera
  catchment: { road: 900, rail: 1500, metro: 1400, hsr: 4500 } as Record<string, number>,
  stationCost: { road: 4000, rail: 12000, metro: 70000, hsr: 150000 } as Record<string, number>,
  stationDays: { road: 3, rail: 8, metro: 30, hsr: 40 } as Record<string, number>,
  stationUpkeep: { road: 120, rail: 350, metro: 900, hsr: 1800 } as Record<string, number>,
  waypointCost: 1500,
  stationCap: [0, 150, 400, 1000],      // pax esperando por nivel (máximo útil)
  levelCostMult: [0, 1, 1.6, 2.8],
  dwellDays: { bus: 0.12, tram: 0.15, train: 0.25, metro: 0.15, hsr: 0.3, truck: 0.3, freight: 0.5 } as Record<string, number>,
  fareKm: 0.25, fareBase: 0.6,
  cargoKm: 0.3,
  staffPerVehicle: 320,       // €/mes era 0; escala con era.fare/0.35
  fuelPrice: 1.0,             // multiplicador base sobre era.fuel
  wearPerKm: 0.00035,         // desgaste por km (≈1 a 2.800 km)
  breakdownBase: 1 / 60,      // prob. por km × (1−fiab) × (0,2 + desgaste)
  breakdownDays: [0.6, 2.2],
  maintenanceFactor: 1.0,
  depreciationYears: 20,
  loanRateBase: 0.08, loanRateMin: 0.03, loanYears: 10,
  fundMu: 0.065, fundSigma: 0.16, fundFee: 0.006,
  cycleAmp: 0.15, cyclePeriod: 9, fuelAmp: 0.3, fuelPeriod: 6,
  repDemand: [0.7, 0.6],      // multiplicador = a + b × rep/100
  cityGrowth: 0.006,          // por mes × (servicio − 0,3)
  cityDecline: 0.0008,
  prestigeThreshold: 5_000_000, prestigeBase: 2_500_000,
  offlineHoursMax: 8, offlineEff: 0.7,
  rushPerDay: 0.0025,         // fracción del coste por día acelerado
  researchSpeed: 1.0,
  contractMonths: 12,
  deliveryDays: 4,
};
