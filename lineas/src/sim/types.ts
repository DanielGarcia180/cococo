/** Modelos de datos. Todo es JSON plano para guardar/cargar. */

export type TrackKind = 'road' | 'rail' | 'metro' | 'hsr';
export type Cargo = 'ore' | 'steel' | 'goods' | 'wood' | 'planks' | 'food' | 'oil' | 'fuel';
export type ZoneKind = 'res' | 'com' | 'ind';
export type Weather = 'clear' | 'cloud' | 'rain' | 'fog' | 'snow';

export interface Vec2 { x: number; y: number }

export interface Zone extends Vec2 { kind: ZoneKind; w: number }

export interface City {
  id: number; name: string; x: number; y: number;
  pop: number; zones: Zone[];
  service: number;          // 0..1 fracción de demanda servida (media móvil)
  served: number;           // pax servidos este mes
  potential: number;        // pax potenciales este mes
  coastal: boolean;
}

export interface Industry {
  id: number; kind: string; name: string; x: number; y: number;
  input: Cargo | null; output: Cargo; rate: number;   // t/mes
  stock: number;           // producción acumulada sin recoger
  supplied: number;        // insumo recibido (t) este mes
  level: number;
}

export interface World {
  seed: number; name: string; size: number; cell: number; seaLevel: number;
  height: number[];        // size*size, metros
  cities: City[]; industries: Industry[];
}

export interface Node {
  id: number; name: string; x: number; y: number;
  kind: TrackKind; station: boolean;   // station=false ⇒ punto de paso
  cityId: number | null;
  doneDay: number;                     // día de fin de obra (<= day ⇒ operativa)
  waiting: Record<number, number>;     // pasajeros esperando por nodo destino
  cargo: Record<string, number>;       // "cargo:dest" ⇒ toneladas esperando
  lost: number;                        // pasajeros perdidos este mes
  boarded: number;                     // pax subidos este mes
  level: number;                       // 1..3 tamaño/capacidad
  cost: number;
  lastArrival?: Record<number, number>; // lineId → día de la última llegada
}

export interface Track {
  id: number; a: number; b: number; kind: TrackKind;
  path: Vec2[]; length: number; tunnel: number; bridge: number;
  cost: number; doneDay: number;
}

export interface LineMonth {
  month: number; revenue: number; cost: number; pax: number; cargo: number;
  load: number; punct: number; km: number;
}

export interface Line {
  id: number; name: string; color: number; kind: TrackKind;
  stops: number[]; circular: boolean; fare: number;   // fare: multiplicador 0.5..2
  vehicles: number[];
  cargoKind: Cargo | null;                            // línea de mercancías
  // caché de ruta (se regenera con network.rebuildLine)
  route: Vec2[]; cum: number[]; stopDist: number[]; stopNodes: number[]; total: number;
  valid: boolean;
  // estadística
  revenue: number; cost: number; pax: number; cargo: number; invested: number;
  mRevenue: number; mCost: number; mPax: number; mCargo: number; mLoadSum: number; mLoadN: number;
  mOnTime: number; mStops: number; mKm: number;
  history: LineMonth[];
  createdDay: number;
}

export interface Upgrades { engine: number; brakes: number; aero: number; comfort: number }

export interface Vehicle {
  id: number; model: string; lineId: number | null;
  dist: number; speed: number; dir: 1;
  state: 'moving' | 'dwell' | 'broken' | 'delivery' | 'idle';
  timer: number;                      // días restantes (dwell/broken/delivery)
  nextStop: number;                   // índice en line.stopDist
  load: Record<number, number>;       // pax por destino
  loadCount: number;
  cargoLoad: number;                  // toneladas
  cargoDest: number; cargoKm: number;
  km: number; wear: number; bought: number; price: number;
  upgrades: Upgrades; livery: number;
  breakdowns: number; trips: number;
  lastDepart: number;                 // día de la última salida (puntualidad)
}

export interface Loan { id: number; principal: number; rate: number; monthsLeft: number; payment: number }

export interface Fund { units: number; nav: number; invested: number; dca: number; history: number[] }

export interface LedgerMonth {
  month: number; year: number;
  fares: number; cargo: number; subsidies: number; fundGain: number; offline: number;
  running: number; maintenance: number; infra: number; staff: number; interest: number; depreciation: number;
  profit: number; cash: number; value: number; pax: number; pop: number;
}

export interface Research { id: string; startDay: number; doneDay: number; cost: number }

export interface Contract {
  id: number; cityId: number; pax: number; monthsLeft: number; reward: number; progress: number; active: boolean;
}

export interface Daily { id: string; target: number; progress: number; done: boolean; reward: number }

export interface Prestige {
  points: number; spent: Record<string, number>; runs: number; totalValue: number;
}

export interface Collection {
  owned: Record<string, { km: number; bought: number; variants: number[] }>;
}

export interface Medal { id: string; best: number; tier: 0 | 1 | 2 | 3 }

export interface Notification { day: number; text: string; kind: 'info' | 'warn' | 'good' }

export interface UndoOp { type: 'addNode' | 'addTrack' | 'delNode' | 'delTrack'; data: unknown; cost: number }

export interface GameState {
  version: number;
  seed: number; rng: number;
  scenario: string; weekly: boolean; weeklyWeek?: string;
  world: World;
  startYear: number; day: number; speed: number; paused: boolean;
  hour: number; weather: Weather; weatherLeft: number;
  nodes: Node[]; tracks: Track[]; lines: Line[]; vehicles: Vehicle[];
  nextId: number;
  cash: number; loans: Loan[]; fund: Fund; reputation: number;
  cycle: number; prevCycle: number; cyclePhase: number; cycleNoise: number; fuel: number; fuelPhase: number; monthPax: number;
  ledger: LedgerMonth[]; cur: LedgerMonth;
  era: number; researched: string[]; research: Research | null;
  contracts: Contract[];
  dailies: Daily[]; dailyDate: string; streak: number; lastPlayDate: string;
  prestige: Prestige; collection: Collection; medals: Medal[];
  notifications: Notification[];
  undo: UndoOp[]; redo: UndoOp[];
  lastRealTime: number;        // ms reales del último tick (para progreso offline)
  profitWindow: number[];      // beneficio diario últimos 30 días
  dayProfit: number; dayRevenue: number; dayPax: number; dayCargo: number; dayBuilt: number;
  monthBreakdowns: number; noBreakdownMonths: number;
  stats: { pax: number; cargo: number; km: number; built: number; vehiclesBought: number; maxValue: number; playSeconds: number; rushSpent?: number };
  tutorial: number;            // paso del tutorial contextual
  settings: { lang: 'es' | 'en'; textScale: number; colorblind: boolean };
  dailyHash: string; dailyKmBase?: number; weeklyReported?: boolean;
  offline: { days: number; profit: number; completed: string[] } | null;
}
