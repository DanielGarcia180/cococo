import type { GameState, Node, Track, Line, Vehicle, TrackKind, Cargo, Weather, UndoOp, Notification } from './types';
import { Rng, seedFromAny } from './rng';
import { generateWorld, nearestCity, isWater } from './world';
import { DATA, model, tech, eraOf, type VehicleModel } from './data';
import { routeTrack, TRACK_SPEC, type RouteResult } from './pathfind';
import { rebuildLine, buildRouting, lineUsesTrack, shortestPath, LINE_COLORS, type Routing } from './network';
import { rebuildDemand, generate, cityCoverage, type StationCover, type PairRate, type CargoPair } from './demand';
import { updateVehicles, bookValue, costMult } from './vehicles';
import { monthEnd, emptyMonth, companyValue, offerContract, yearOf, monthOf } from './economy';
import { BAL, DT, SEC_PER_DAY, DAYS_PER_MONTH, SAVE_VERSION } from './constants';
import { refreshDailies, dailyProgress, applyOffline, refoundPoints } from './progress';
import { updateMedals, weeklySeed, isoWeek, WEEKLY_YEARS, weeklyDone, variantsUnlocked } from './challenges';

export interface NewGameOpts {
  scenario: string; seed?: string | number; weekly?: boolean;
  carry?: { prestige: GameState['prestige']; collection: GameState['collection']; medals: GameState['medals']; settings: GameState['settings']; streak: number; dailyDate: string };
}

/** Contexto de simulación: estado + cachés derivadas (no se guardan). */
export interface Ctx {
  s: GameState; rng: Rng; routing: Routing;
  cover: Map<number, StationCover>; rates: Map<number, PairRate[]>; cargoPairs: CargoPair[];
  nodeMap: Map<number, Node>; lineMap: Map<number, Line>; cityCov: Map<number, number>;
  notify: (text: string, kind?: Notification['kind']) => void;
  afterOffline: () => void;
}

export type GameEvent = { type: string; [k: string]: unknown };

export class Game implements Ctx {
  s!: GameState; rng!: Rng;
  routing: Routing = { next: new Map(), reach: new Map() };
  cover = new Map<number, StationCover>(); rates = new Map<number, PairRate[]>(); cargoPairs: CargoPair[] = [];
  nodeMap = new Map<number, Node>(); lineMap = new Map<number, Line>(); cityCov = new Map<number, number>();
  dirty = true; private acc = 0; private lastDay = 0; private lastMonth = 0; private lastYear = 0;
  listeners: ((e: GameEvent) => void)[] = [];

  constructor(opts?: NewGameOpts) { if (opts) this.newGame(opts); }

  emit(e: GameEvent) { for (const l of this.listeners) l(e); }
  notify = (text: string, kind: Notification['kind'] = 'info') => {
    this.s.notifications.push({ day: this.s.day, text, kind });
    if (this.s.notifications.length > 60) this.s.notifications.shift();
    this.emit({ type: 'notify', text, kind });
  };
  afterOffline = () => { this.syncRng(); this.markDirty(); };

  // ───────────────────────── creación ─────────────────────────
  newGame(opts: NewGameOpts): void {
    const sc = DATA.scenarios.find(x => x.id === opts.scenario) ?? DATA.scenarios[0];
    let seed = opts.seed !== undefined && opts.seed !== '' ? seedFromAny(opts.seed) : (sc.world.seed ?? (Date.now() >>> 0));
    let weeklyWeek: string | undefined;
    if (opts.weekly) { weeklyWeek = isoWeek(); seed = weeklySeed(weeklyWeek); }
    const world = generateWorld({ ...sc.world, seed, name: sc.world.name });
    const rng = new Rng(seed ^ 0xabcdef);
    const carry = opts.carry;
    const cashMult = 1 + 0.25 * (carry?.prestige.spent.cash ?? 0);
    this.s = {
      version: SAVE_VERSION, seed, rng: 0, scenario: sc.id, weekly: !!opts.weekly, weeklyWeek, world,
      startYear: sc.startYear, day: 0, speed: 1, paused: false, hour: 8, weather: 'clear', weatherLeft: 3,
      nodes: [], tracks: [], lines: [], vehicles: [], nextId: 1,
      cash: Math.round(sc.startCash * cashMult), loans: [], fund: { units: 0, nav: 100, invested: 0, dca: 0, history: [100] }, reputation: 50,
      cycle: 1, prevCycle: 1, cyclePhase: rng.range(0, Math.PI * 2), cycleNoise: 0, fuel: 1, fuelPhase: rng.range(0, Math.PI * 2), monthPax: 0,
      ledger: [], cur: emptyMonth(0, sc.startYear),
      era: 0, researched: [], research: null, contracts: [],
      dailies: [], dailyDate: carry?.dailyDate ?? '', streak: carry?.streak ?? 0, lastPlayDate: '',
      prestige: carry?.prestige ?? { points: 0, spent: {}, runs: 0, totalValue: 0 },
      collection: carry?.collection ?? { owned: {} }, medals: carry?.medals ?? [],
      notifications: [], undo: [], redo: [], lastRealTime: 0, profitWindow: [], dayProfit: 0, dayRevenue: 0, dayPax: 0, dayCargo: 0, dayBuilt: 0,
      monthBreakdowns: 0, noBreakdownMonths: 0,
      stats: { pax: 0, cargo: 0, km: 0, built: 0, vehiclesBought: 0, maxValue: 0, playSeconds: 0 },
      tutorial: sc.tutorial ? 0 : 99,
      settings: carry?.settings ?? { lang: 'es', textScale: 1, colorblind: false },
      dailyHash: '', offline: null,
    };
    this.rng = rng; this.syncRng();
    this.lastDay = 0; this.lastMonth = 0; this.lastYear = 0;
    this.markDirty();
    this.notify(`Bienvenido a ${world.name}. Construye una estación para empezar.`, 'info');
  }

  load(state: GameState): void {
    this.s = state;
    this.rng = new Rng(state.rng);
    this.lastDay = Math.floor(state.day); this.lastMonth = Math.floor(state.day / DAYS_PER_MONTH); this.lastYear = Math.floor(state.day / 360);
    for (const l of this.s.lines) rebuildLine(this.s, l);
    this.markDirty(); this.refresh();
  }

  private syncRng() { this.s.rng = this.rng.s; }
  markDirty() { this.dirty = true; }

  /** Reconstruye cachés derivadas. */
  refresh(): void {
    const s = this.s;
    this.nodeMap = new Map(s.nodes.map(n => [n.id, n]));
    this.lineMap = new Map(s.lines.map(l => [l.id, l]));
    for (const l of s.lines) rebuildLine(s, l);
    this.routing = buildRouting(s);
    rebuildDemand(this);
    this.cityCov = new Map(s.world.cities.map(c => [c.id, cityCoverage(s, c)]));
    this.dirty = false;
    this.emit({ type: 'network' });
  }

  // ───────────────────────── bucle ─────────────────────────
  year(): number { return yearOf(this.s); }
  month(): number { return monthOf(this.s); }

  /** Avanza la simulación `realSeconds` de tiempo real, aplicando la velocidad de juego. */
  advance(realSeconds: number): void {
    const s = this.s;
    s.stats.playSeconds += realSeconds;
    if (s.paused) return;
    this.acc += Math.min(realSeconds, 0.25) * s.speed;
    let steps = 0;
    while (this.acc >= DT && steps < 400) { this.step(); this.acc -= DT; steps++; }
    if (steps >= 400) this.acc = 0;
  }

  /** Un paso fijo de DT segundos (a 1×). */
  step(): void {
    const s = this.s;
    if (this.dirty) this.refresh();
    const dtDays = DT / SEC_PER_DAY;
    s.day += dtDays;
    s.hour = (s.day % 1) * 24;
    generate(this, dtDays);
    updateVehicles(this, DT, dtDays);
    const day = Math.floor(s.day);
    if (day !== this.lastDay) { for (let d = this.lastDay; d < day; d++) this.dayEnd(); this.lastDay = day; }
    this.syncRng();
  }

  private dayEnd(): void {
    const s = this.s;
    s.profitWindow.push(s.dayProfit); if (s.profitWindow.length > 30) s.profitWindow.shift();
    const msgs = [...dailyProgress(s, 'pax', s.dayPax), ...dailyProgress(s, 'earn', s.dayRevenue), ...dailyProgress(s, 'cargo', s.dayCargo), ...dailyProgress(s, 'km', 0), ...dailyProgress(s, 'build', s.dayBuilt)];
    for (const m of msgs) this.notify(m, 'good');
    s.monthPax += s.dayPax;
    s.dayProfit = 0; s.dayRevenue = 0; s.dayPax = 0; s.dayCargo = 0; s.dayBuilt = 0;
    // km diario para objetivos
    const kmDaily = s.vehicles.reduce((a, v) => a + v.km, 0);
    const prevKm = s.dailyKmBase ?? kmDaily; s.dailyKmBase = kmDaily;
    for (const m of dailyProgress(s, 'km', kmDaily - prevKm)) this.notify(m, 'good');
    // clima
    s.weatherLeft -= 1;
    if (s.weatherLeft <= 0) {
      const month = this.month();
      const winter = month === 11 || month === 0 || month === 1;
      const r = this.rng.next();
      s.weather = r < 0.5 ? 'clear' : r < 0.72 ? 'cloud' : r < 0.88 ? 'rain' : r < 0.95 ? 'fog' : winter ? 'snow' : 'rain';
      s.weatherLeft = this.rng.int(2, 6);
    }
    // obras terminadas
    for (const n of s.nodes) if (n.doneDay > s.day - 1 && n.doneDay <= s.day) { this.markDirty(); if (n.station) this.notify(`${n.name} abre al público.`, 'good'); }
    for (const t of s.tracks) if (t.doneDay > s.day - 1 && t.doneDay <= s.day) this.markDirty();
    // investigación
    if (s.research && s.research.doneDay <= s.day) this.completeResearch();
    // mes
    const month = Math.floor(s.day / DAYS_PER_MONTH);
    if (month !== this.lastMonth) { this.lastMonth = month; this.monthEnd(); }
    const year = Math.floor(s.day / 360);
    if (year !== this.lastYear) { this.lastYear = year; this.yearEnd(); }
    if (this.dirty) this.refresh();
    this.emit({ type: 'day' });
  }

  private monthEnd(): void {
    const s = this.s;
    monthEnd(this);
    for (const m of updateMedals(s)) this.notify(m, 'good');
    if (s.vehicles.some(v => v.wear > 0.85)) this.notify('Hay vehículos muy desgastados: considera sustituirlos.', 'warn');
    this.markDirty();
    this.emit({ type: 'month' });
  }
  private yearEnd(): void {
    const s = this.s;
    if (this.rng.chance(0.6)) offerContract(this);
    if (s.weekly && s.day >= WEEKLY_YEARS * 360 && !s.weeklyReported) { s.weeklyReported = true; this.emit({ type: 'weeklyDone' }); }
    this.emit({ type: 'year' });
  }

  // ───────────────────────── construcción ─────────────────────────
  stationCost(kind: TrackKind, station: boolean): number { return station ? BAL.stationCost[kind] * (eraOf(this.s.era).fare / 16) ** 0.5 * costMult(this.s) : BAL.waypointCost; }
  stationDays(kind: TrackKind, station: boolean): number { return Math.max(1, Math.round((station ? BAL.stationDays[kind] : 1) * this.buildMult())); }
  buildMult(): number { return 1 - 0.08 * (this.s.prestige.spent.build ?? 0); }

  canPlaceNode(x: number, y: number, kind: TrackKind): string | null {
    const s = this.s;
    const ext = (s.world.size - 1) * s.world.cell;
    if (x < 0 || y < 0 || x > ext || y > ext) return 'Fuera del mapa';
    if (kind !== 'metro' && isWater(s.world, x, y)) return 'No se puede construir en el agua';
    if (s.nodes.some(n => Math.hypot(n.x - x, n.y - y) < 350)) return 'Demasiado cerca de otro nodo';
    if (kind === 'metro' && !s.researched.includes('metro')) return 'Investiga "Ferrocarril subterráneo"';
    if (kind === 'hsr' && !s.researched.includes('hsr')) return 'Investiga "Alta velocidad"';
    return null;
  }

  placeNode(x: number, y: number, kind: TrackKind, station = true, name?: string): Node | null {
    const s = this.s;
    if (this.canPlaceNode(x, y, kind)) return null;
    const cost = this.stationCost(kind, station);
    if (s.cash < cost) { this.notify('Caja insuficiente', 'warn'); return null; }
    const city = nearestCity(s.world, x, y);
    const count = s.nodes.filter(n => n.station && n.cityId === (city?.id ?? null)).length;
    const kindName = { road: 'Parada', rail: 'Estación', metro: 'Metro', hsr: 'Terminal AV' }[kind];
    const n: Node = {
      id: s.nextId++, name: name ?? (station ? `${kindName} ${city ? city.name : 'Campo'}${count ? ' ' + (count + 1) : ''}` : `Punto ${s.nextId}`),
      x, y, kind, station, cityId: city?.id ?? null, doneDay: s.day + this.stationDays(kind, station), waiting: {}, cargo: {}, lost: 0, boarded: 0, level: 1, cost,
    };
    s.nodes.push(n); s.cash -= cost; s.stats.built++; s.dayBuilt++;
    s.undo.push({ type: 'addNode', data: n.id, cost }); s.redo = []; if (s.undo.length > 30) s.undo.shift();
    if (s.tutorial === 0) s.tutorial = 1; else if (s.tutorial === 1 && s.nodes.filter(z => z.station).length >= 2) s.tutorial = 2;
    this.markDirty(); this.emit({ type: 'build' });
    return n;
  }

  previewTrack(a: number, b: number, kind: TrackKind): (RouteResult & { error?: string }) | null {
    const s = this.s; const na = this.nodeMap.get(a) ?? s.nodes.find(n => n.id === a), nb = this.nodeMap.get(b) ?? s.nodes.find(n => n.id === b);
    if (!na || !nb || a === b) return null;
    if (s.tracks.some(t => (t.a === a && t.b === b) || (t.a === b && t.b === a))) return null;
    const r = routeTrack(s.world, na, nb, kind); if (!r) return null;
    const cm = costMult(s); r.cost = Math.round(r.cost * cm); r.days = Math.max(1, Math.round(r.days * this.buildMult()));
    let error: string | undefined;
    if (kind === 'hsr' && !s.researched.includes('hsr')) error = 'Investiga "Alta velocidad"';
    if (kind === 'metro' && !s.researched.includes('metro')) error = 'Investiga "Ferrocarril subterráneo"';
    if (s.cash < r.cost) error = 'Caja insuficiente';
    return { ...r, error };
  }

  buildTrack(a: number, b: number, kind: TrackKind): Track | null {
    const s = this.s; const r = this.previewTrack(a, b, kind);
    if (!r || r.error) { if (r?.error) this.notify(r.error, 'warn'); return null; }
    const t: Track = { id: s.nextId++, a, b, kind, path: r.path, length: r.length, tunnel: r.tunnel, bridge: r.bridge, cost: r.cost, doneDay: s.day + r.days };
    s.tracks.push(t); s.cash -= r.cost; s.stats.built++; s.dayBuilt++;
    s.undo.push({ type: 'addTrack', data: t.id, cost: r.cost }); s.redo = [];
    if (s.tutorial === 2) s.tutorial = 3;
    this.markDirty(); this.emit({ type: 'build' });
    return t;
  }

  removeTrack(id: number, refund = 0.5): boolean {
    const s = this.s; const t = s.tracks.find(x => x.id === id); if (!t) return false;
    if (s.lines.some(l => lineUsesTrack(s, l, id))) { this.notify('Una línea usa esta vía', 'warn'); return false; }
    s.tracks = s.tracks.filter(x => x.id !== id); s.cash += t.cost * refund;
    s.undo.push({ type: 'delTrack', data: t, cost: -t.cost * refund }); s.redo = [];
    this.markDirty(); this.emit({ type: 'build' }); return true;
  }
  removeNode(id: number, refund = 0.5): boolean {
    const s = this.s; const n = s.nodes.find(x => x.id === id); if (!n) return false;
    if (s.tracks.some(t => t.a === id || t.b === id)) { this.notify('Elimina primero las vías conectadas', 'warn'); return false; }
    if (s.lines.some(l => l.stops.includes(id))) { this.notify('Una línea para aquí', 'warn'); return false; }
    s.nodes = s.nodes.filter(x => x.id !== id); s.cash += n.cost * refund;
    s.undo.push({ type: 'delNode', data: n, cost: -n.cost * refund }); s.redo = [];
    this.markDirty(); this.emit({ type: 'build' }); return true;
  }
  undo(): boolean {
    const s = this.s; const op = s.undo.pop(); if (!op) return false;
    const ok = this.applyInverse(op);
    if (ok) s.redo.push(op); return ok;
  }
  redo(): boolean {
    const s = this.s; const op = s.redo.pop(); if (!op) return false;
    const ok = this.applyForward(op); if (ok) s.undo.push(op); return ok;
  }
  private applyInverse(op: UndoOp): boolean {
    const s = this.s;
    if (op.type === 'addNode') { const n = s.nodes.find(x => x.id === op.data); if (!n || s.tracks.some(t => t.a === n.id || t.b === n.id) || s.lines.some(l => l.stops.includes(n.id))) return false; s.nodes = s.nodes.filter(x => x !== n); s.cash += op.cost; op.data = n; }
    else if (op.type === 'addTrack') { const t = s.tracks.find(x => x.id === op.data); if (!t || s.lines.some(l => lineUsesTrack(s, l, t.id))) return false; s.tracks = s.tracks.filter(x => x !== t); s.cash += op.cost; op.data = t; }
    else if (op.type === 'delNode') { const n = op.data as Node; s.nodes.push(n); s.cash += op.cost; }
    else if (op.type === 'delTrack') { const t = op.data as Track; if (!s.nodes.some(n => n.id === t.a) || !s.nodes.some(n => n.id === t.b)) return false; s.tracks.push(t); s.cash += op.cost; }
    this.markDirty(); this.emit({ type: 'build' }); return true;
  }
  private applyForward(op: UndoOp): boolean {
    const s = this.s;
    if (op.type === 'addNode') { const n = op.data as Node; if (s.cash < op.cost) return false; s.nodes.push(n); s.cash -= op.cost; }
    else if (op.type === 'addTrack') { const t = op.data as Track; if (s.cash < op.cost || !s.nodes.some(n => n.id === t.a) || !s.nodes.some(n => n.id === t.b)) return false; s.tracks.push(t); s.cash -= op.cost; }
    else if (op.type === 'delNode') { const n = op.data as Node; if (s.tracks.some(t => t.a === n.id || t.b === n.id)) return false; s.nodes = s.nodes.filter(x => x.id !== n.id); s.cash -= op.cost; }
    else if (op.type === 'delTrack') { const t = op.data as Track; if (s.lines.some(l => lineUsesTrack(s, l, t.id))) return false; s.tracks = s.tracks.filter(x => x.id !== t.id); s.cash -= op.cost; }
    this.markDirty(); this.emit({ type: 'build' }); return true;
  }

  upgradeStation(id: number): boolean {
    const s = this.s; const n = s.nodes.find(x => x.id === id); if (!n || !n.station || n.level >= 3) return false;
    if (n.level === 1 && !s.researched.includes('signals')) { this.notify('Investiga "Señalización mecánica"', 'warn'); return false; }
    if (n.level === 2 && !s.researched.includes('station3')) { this.notify('Investiga "Estaciones intermodales"', 'warn'); return false; }
    const cost = this.stationCost(n.kind, true) * BAL.levelCostMult[n.level + 1];
    if (s.cash < cost) { this.notify('Caja insuficiente', 'warn'); return false; }
    s.cash -= cost; n.cost += cost; n.level++; n.doneDay = Math.max(n.doneDay, s.day) + this.stationDays(n.kind, true);
    this.markDirty(); this.emit({ type: 'build' }); return true;
  }

  /** Coste de acelerar una obra hasta hoy. */
  rushCost(kind: 'node' | 'track' | 'research' | 'vehicle', id: number | string): number {
    const s = this.s;
    if (kind === 'node') { const n = s.nodes.find(x => x.id === id); return n && n.doneDay > s.day ? Math.ceil(n.cost * BAL.rushPerDay * (n.doneDay - s.day)) + 100 : 0; }
    if (kind === 'track') { const t = s.tracks.find(x => x.id === id); return t && t.doneDay > s.day ? Math.ceil(t.cost * BAL.rushPerDay * (t.doneDay - s.day)) + 100 : 0; }
    if (kind === 'vehicle') { const v = s.vehicles.find(x => x.id === id); return v && v.state === 'delivery' ? Math.ceil(v.price * BAL.rushPerDay * v.timer * 4) + 100 : 0; }
    if (kind === 'research' && s.research) return Math.ceil(s.research.cost * BAL.rushPerDay * (s.research.doneDay - s.day)) + 500;
    return 0;
  }
  rush(kind: 'node' | 'track' | 'research' | 'vehicle', id: number | string): boolean {
    const s = this.s; const cost = this.rushCost(kind, id); if (cost <= 0 || s.cash < cost) return false;
    s.cash -= cost; s.stats.rushSpent = (s.stats.rushSpent ?? 0) + cost;   // inversión (no pasa por la cuenta de resultados)
    if (kind === 'node') s.nodes.find(x => x.id === id)!.doneDay = s.day;
    else if (kind === 'track') s.tracks.find(x => x.id === id)!.doneDay = s.day;
    else if (kind === 'vehicle') { const v = s.vehicles.find(x => x.id === id)!; v.timer = 0; v.state = v.lineId === null ? 'idle' : 'moving'; }
    else if (s.research) this.completeResearch();
    this.markDirty(); this.emit({ type: 'build' }); return true;
  }

  // ───────────────────────── líneas ─────────────────────────
  createLine(kind: TrackKind, cargoKind: Cargo | null = null, name?: string): Line {
    const s = this.s; const idx = s.lines.length;
    const l: Line = {
      id: s.nextId++, name: name ?? (cargoKind ? `Carga ${idx + 1}` : `Línea ${idx + 1}`), color: LINE_COLORS[idx % LINE_COLORS.length], kind,
      stops: [], circular: false, fare: 1, vehicles: [], cargoKind,
      route: [], cum: [], stopDist: [], stopNodes: [], total: 0, valid: false,
      revenue: 0, cost: 0, pax: 0, cargo: 0, invested: 0, mRevenue: 0, mCost: 0, mPax: 0, mCargo: 0, mLoadSum: 0, mLoadN: 0, mOnTime: 0, mStops: 0, mKm: 0,
      history: [], createdDay: s.day,
    };
    s.lines.push(l); this.markDirty(); this.emit({ type: 'lines' }); return l;
  }
  lineAddStop(lineId: number, nodeId: number): string | null {
    const s = this.s; const l = s.lines.find(x => x.id === lineId); const n = s.nodes.find(x => x.id === nodeId);
    if (!l || !n) return 'No existe';
    if (l.stops[l.stops.length - 1] === nodeId) return 'Ya es la última parada';
    if (l.stops.length) {
      const prev = s.nodes.find(x => x.id === l.stops[l.stops.length - 1])!;
      const r = this.previewPath(prev.id, nodeId, l.kind);
      if (!r) return 'No hay vía de este tipo entre las paradas';
    }
    l.stops.push(nodeId); this.markDirty(); this.emit({ type: 'lines' });
    if (s.tutorial === 3 && l.stops.length >= 2) s.tutorial = 4;
    return null;
  }
  previewPath(a: number, b: number, kind: TrackKind) { return shortestPath(this.s, a, b, kind); }
  lineRemoveStop(lineId: number, index: number): void {
    const l = this.s.lines.find(x => x.id === lineId); if (!l) return;
    l.stops.splice(index, 1); this.markDirty(); this.emit({ type: 'lines' });
  }
  lineSet(lineId: number, patch: Partial<Pick<Line, 'name' | 'fare' | 'circular' | 'color'>>): void {
    const l = this.s.lines.find(x => x.id === lineId); if (!l) return;
    Object.assign(l, patch); if (patch.fare !== undefined) l.fare = Math.max(0.5, Math.min(2, patch.fare));
    this.markDirty(); this.emit({ type: 'lines' });
  }
  deleteLine(lineId: number): void {
    const s = this.s; const l = s.lines.find(x => x.id === lineId); if (!l) return;
    for (const vid of l.vehicles) { const v = s.vehicles.find(x => x.id === vid); if (v) { v.lineId = null; v.state = v.state === 'delivery' ? 'delivery' : 'idle'; v.load = {}; v.loadCount = 0; v.cargoLoad = 0; } }
    s.lines = s.lines.filter(x => x.id !== lineId); this.markDirty(); this.emit({ type: 'lines' });
  }

  // ───────────────────────── vehículos ─────────────────────────
  availableModels(): VehicleModel[] {
    const s = this.s;
    return DATA.vehicles.filter(m => (m.unlock === null || s.researched.includes(m.unlock)) && m.era <= s.era);
  }
  buyVehicle(modelId: string, lineId: number | null): Vehicle | null {
    const s = this.s; const m = model(modelId); if (!m) return null;
    if (!this.availableModels().some(x => x.id === modelId)) { this.notify('Modelo no disponible todavía', 'warn'); return null; }
    const price = Math.round(m.price * costMult(s));
    if (s.cash < price) { this.notify('Caja insuficiente', 'warn'); return null; }
    const line = lineId !== null ? s.lines.find(l => l.id === lineId) : null;
    if (line && !this.lineAccepts(line, m)) { this.notify('Este vehículo no puede circular por esa línea', 'warn'); return null; }
    const v: Vehicle = {
      id: s.nextId++, model: modelId, lineId: line ? line.id : null, dist: 0, speed: 0, dir: 1, state: 'delivery', timer: BAL.deliveryDays * this.buildMult() * (1 + m.era * 0.5),
      nextStop: 1, load: {}, loadCount: 0, cargoLoad: 0, cargoDest: -1, cargoKm: 0, km: 0, wear: 0, bought: s.day, price,
      upgrades: { engine: 0, brakes: 0, aero: 0, comfort: 0 }, livery: 0, breakdowns: 0, trips: 0, lastDepart: s.day,
    };
    s.vehicles.push(v); s.cash -= price; s.stats.vehiclesBought++;
    if (line) { line.vehicles.push(v.id); line.invested += price; }
    if (!s.collection.owned[modelId]) { s.collection.owned[modelId] = { km: 0, bought: s.day, variants: [0] }; this.notify(`Nuevo modelo en la colección: ${m.name}`, 'good'); }
    this.markDirty(); this.emit({ type: 'vehicles' });
    return v;
  }
  lineAccepts(line: Line, m: VehicleModel): boolean {
    if (!!line.cargoKind !== m.cargo) return false;
    if (m.track === line.kind) return true;
    if (m.track === 'rail' && line.kind === 'hsr') return true;
    return false;
  }
  assignVehicle(vehicleId: number, lineId: number | null): boolean {
    const s = this.s; const v = s.vehicles.find(x => x.id === vehicleId); if (!v) return false;
    const m = model(v.model)!;
    const line = lineId !== null ? s.lines.find(l => l.id === lineId) : null;
    if (lineId !== null && !line) return false;
    if (line && !this.lineAccepts(line, m)) return false;
    if (v.lineId !== null) { const old = s.lines.find(l => l.id === v.lineId); if (old) old.vehicles = old.vehicles.filter(x => x !== vehicleId); }
    v.lineId = line ? line.id : null; v.dist = 0; v.nextStop = 1; v.load = {}; v.loadCount = 0; v.cargoLoad = 0; v.speed = 0;
    if (v.state !== 'delivery' && v.state !== 'broken') v.state = line ? 'moving' : 'idle';
    if (line) line.vehicles.push(v.id);
    this.markDirty(); this.emit({ type: 'vehicles' }); return true;
  }
  sellVehicle(vehicleId: number): boolean {
    const s = this.s; const v = s.vehicles.find(x => x.id === vehicleId); if (!v) return false;
    const value = bookValue(v, s.day) * 0.7;
    if (v.lineId !== null) { const l = s.lines.find(x => x.id === v.lineId); if (l) l.vehicles = l.vehicles.filter(x => x !== vehicleId); }
    s.vehicles = s.vehicles.filter(x => x.id !== vehicleId); s.cash += value;
    this.markDirty(); this.emit({ type: 'vehicles' }); return true;
  }
  upgradeCost(v: Vehicle, part: keyof Vehicle['upgrades']): number { return Math.round(v.price * 0.12 * (v.upgrades[part] + 1)); }
  upgradeVehicle(vehicleId: number, part: keyof Vehicle['upgrades']): boolean {
    const s = this.s; const v = s.vehicles.find(x => x.id === vehicleId); if (!v) return false;
    if (!s.researched.includes('tuning')) { this.notify('Investiga "Taller de competición"', 'warn'); return false; }
    if (v.upgrades[part] >= 3) return false;
    const cost = this.upgradeCost(v, part); if (s.cash < cost) { this.notify('Caja insuficiente', 'warn'); return false; }
    s.cash -= cost; v.upgrades[part]++; v.price += cost * 0.5;   // inversión capitalizada en el vehículo
    if (v.lineId !== null) { const l = s.lines.find(x => x.id === v.lineId); if (l) { l.invested += cost; l.cost += 0; } }
    this.emit({ type: 'vehicles' }); return true;
  }
  setLivery(vehicleId: number, livery: number): boolean {
    const s = this.s; const v = s.vehicles.find(x => x.id === vehicleId); if (!v) return false;
    if (livery >= variantsUnlocked(s, v.model) * 2) return false;
    v.livery = livery; this.emit({ type: 'vehicles' }); return true;
  }

  // ───────────────────────── investigación ─────────────────────────
  researchAvailable(): { id: string; ok: boolean; why: string | null }[] {
    const s = this.s; const year = this.year();
    return DATA.tech.map(t => {
      if (s.researched.includes(t.id)) return { id: t.id, ok: false, why: 'Completada' };
      if (s.research?.id === t.id) return { id: t.id, ok: false, why: 'En curso' };
      const missing = t.req.filter(r => !s.researched.includes(r));
      if (missing.length) return { id: t.id, ok: false, why: `Requiere ${missing.map(m => tech(m)?.name ?? m).join(', ')}` };
      if (t.year && year < t.year - 10) return { id: t.id, ok: false, why: `Disponible a partir de ${t.year - 10}` };
      if (s.research) return { id: t.id, ok: false, why: 'Ya hay una investigación en curso' };
      if (s.cash < t.cost) return { id: t.id, ok: false, why: 'Caja insuficiente' };
      return { id: t.id, ok: true, why: null };
    });
  }
  startResearch(id: string): boolean {
    const s = this.s; const t = tech(id); if (!t) return false;
    const av = this.researchAvailable().find(a => a.id === id); if (!av?.ok) { if (av?.why) this.notify(av.why, 'warn'); return false; }
    const mult = BAL.researchSpeed * (1 - 0.08 * (s.prestige.spent.research ?? 0));
    s.cash -= t.cost; s.cur.infra += t.cost * 0;   // la investigación es inversión, no gasto operativo
    s.research = { id, startDay: s.day, doneDay: s.day + Math.max(1, Math.round(t.days * mult)), cost: t.cost };
    this.notify(`Investigando: ${t.name}`, 'info'); this.emit({ type: 'research' }); return true;
  }
  private completeResearch(): void {
    const s = this.s; if (!s.research) return;
    const t = tech(s.research.id); s.researched.push(s.research.id); s.research = null;
    if (t) {
      if (t.id.startsWith('era_') && t.era > s.era) { s.era = t.era; this.notify(`Nueva era: ${eraOf(s.era).name}. Nuevos vehículos disponibles.`, 'good'); }
      else this.notify(`Investigación completada: ${t.name}`, 'good');
    }
    this.markDirty(); this.emit({ type: 'research' });
  }

  // ───────────────────────── capital ─────────────────────────
  fundInvest(amount: number): boolean { const s = this.s; if (amount <= 0 || s.cash < amount) return false; s.cash -= amount; s.fund.units += amount / s.fund.nav; s.fund.invested += amount; this.emit({ type: 'fund' }); return true; }
  fundWithdraw(amount: number): boolean {
    const s = this.s; const value = s.fund.units * s.fund.nav; amount = Math.min(amount, value); if (amount <= 0) return false;
    const units = amount / s.fund.nav; s.fund.units -= units; s.fund.invested = Math.max(0, s.fund.invested - s.fund.invested * (units / (s.fund.units + units)));
    s.cash += amount; this.emit({ type: 'fund' }); return true;
  }
  setDca(amount: number): void { this.s.fund.dca = Math.max(0, Math.round(amount)); this.emit({ type: 'fund' }); }

  // ───────────────────────── prestigio ─────────────────────────
  /** Refunda la compañía: devuelve las opciones para un newGame con bonos permanentes. */
  refound(scenario: string, seed?: string | number): boolean {
    const s = this.s; const pts = refoundPoints(s); if (pts <= 0) return false;
    const value = companyValue(s);
    const carry = { prestige: { ...s.prestige, points: s.prestige.points + pts, runs: s.prestige.runs + 1, totalValue: s.prestige.totalValue + value }, collection: s.collection, medals: s.medals, settings: s.settings, streak: s.streak, dailyDate: s.dailyDate };
    this.newGame({ scenario, seed, carry });
    this.notify(`Compañía refundada. +${pts} puntos de prestigio.`, 'good');
    return true;
  }

  // ───────────────────────── sesión ─────────────────────────
  /** Al abrir una partida: progreso offline y objetivos diarios. */
  onSessionStart(now = Date.now()): void {
    const s = this.s;
    applyOffline(this, now);
    if (refreshDailies(s, now)) this.emit({ type: 'dailies' });
    s.lastPlayDate = this.todayKey(now);
  }
  todayKey(now: number): string { const d = new Date(now); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`; }
  touchRealTime(now = Date.now()): void { this.s.lastRealTime = now; }

  value(): number { return companyValue(this.s); }
  setSpeed(sp: number): void { this.s.speed = sp; this.s.paused = false; this.emit({ type: 'speed' }); }
  togglePause(): void { this.s.paused = !this.s.paused; this.emit({ type: 'speed' }); }
  weeklyFinished(): boolean { return weeklyDone(this.s); }
}

export { companyValue };
