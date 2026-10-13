import type { GameState, Node, City } from './types';
import { BAL } from './constants';
import { CITY_CARGO } from './world';
import { eraOf } from './data';
import type { Ctx } from './game';

export interface StationCover { P: number; A: number; cities: Map<number, number> }   // cities: cityId → fracción cubierta (suma de pesos)
export interface PairRate { dst: number; rate: number }
export interface CargoPair { lineId: number; src: number; dst: number; cargo: string; industry: number; toCity: boolean }

export function catchment(s: GameState, n: Node): number {
  const base = BAL.catchment[n.kind] * (s.researched.includes('station3') ? 1.2 : 1);
  return base * (1 + (n.level - 1) * 0.15);
}

/** Cobertura de un nodo: producción y atracción ponderadas por zonas en su radio. */
export function coverOf(s: GameState, n: Node): StationCover {
  const r = catchment(s, n);
  let P = 0, A = 0; const cities = new Map<number, number>();
  for (const c of s.world.cities) {
    if (Math.hypot(c.x - n.x, c.y - n.y) > r + 2000) continue;
    let frac = 0;
    for (const z of c.zones) {
      const d = Math.hypot(z.x - n.x, z.y - n.y);
      if (d > r) continue;
      const f = 1 - d / r * 0.6;
      const w = z.w * c.pop * f;
      if (z.kind === 'res') { P += w; A += w * 0.25; } else { A += w; P += w * 0.3; }
      frac += z.w * f;
    }
    if (frac > 0) cities.set(c.id, frac);
  }
  return { P, A, cities };
}

/** Fracción de la demanda potencial de una ciudad que está cubierta por estaciones operativas. */
export function cityCoverage(s: GameState, c: City): number {
  let cov = 0;
  for (const z of c.zones) {
    let best = 0;
    for (const n of s.nodes) {
      if (!n.station || n.doneDay > s.day) continue;
      const r = catchment(s, n);
      const d = Math.hypot(z.x - n.x, z.y - n.y);
      if (d < r) best = Math.max(best, 1 - d / r * 0.6);
    }
    cov += z.w * best;
  }
  return cov;
}

export function priceMult(fare: number): number { return Math.pow(1 / fare, 1.2); }

export function demandMultiplier(s: GameState): number {
  const rep = BAL.repDemand[0] + BAL.repDemand[1] * s.reputation / 100;
  const weather = s.weather === 'rain' ? 0.92 : s.weather === 'snow' ? 0.8 : s.weather === 'fog' ? 0.95 : 1;
  const prestige = 1 + 0.02 * (s.prestige.spent.demand ?? 0);
  return s.cycle * rep * weather * prestige;
}

/** Recalcula la matriz de demanda (pax/día) entre estaciones conectadas. */
export function rebuildDemand(ctx: Ctx): void {
  const s = ctx.s;
  ctx.cover.clear(); ctx.rates.clear();
  const stations = s.nodes.filter(n => n.station && n.doneDay <= s.day);
  for (const n of stations) ctx.cover.set(n.id, coverOf(s, n));
  // tarifa media de las líneas que sirven cada nodo (para elasticidad)
  const fareAt = new Map<number, number>();
  for (const l of s.lines) if (l.valid && !l.cargoKind) for (const id of l.stops) { const f = fareAt.get(id); fareAt.set(id, f === undefined ? l.fare : (f + l.fare) / 2); }
  const comfortAt = new Map<number, number>();
  for (const l of s.lines) if (l.valid && !l.cargoKind) {
    let c = 0, k = 0; for (const vid of l.vehicles) { const v = s.vehicles.find(x => x.id === vid); if (v) { c += v.upgrades.comfort; k++; } }
    const avg = k ? c / k : 0; for (const id of l.stops) comfortAt.set(id, Math.max(comfortAt.get(id) ?? 0, avg));
  }
  for (const a of stations) {
    const reach = ctx.routing.reach.get(a.id); if (!reach) continue;
    const ca = ctx.cover.get(a.id)!; if (ca.P <= 0) continue;
    const list: PairRate[] = [];
    for (const bId of reach) {
      const b = s.nodes.find(n => n.id === bId); if (!b || !b.station) continue;
      const cb = ctx.cover.get(bId); if (!cb || cb.A <= 0) continue;
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (d < BAL.minTripDist) continue;
      let rate = BAL.demandK * Math.pow(ca.P * cb.A, BAL.demandExp) / (1 + (d / BAL.demandD0) ** 2);
      rate *= priceMult(fareAt.get(a.id) ?? 1);
      rate *= 1 + 0.03 * (comfortAt.get(a.id) ?? 0);
      if (rate > 0.05) list.push({ dst: bId, rate });
    }
    if (list.length) ctx.rates.set(a.id, list);
  }
  // demanda potencial por ciudad (para el indicador de servicio): suma de generación con cobertura total
  rebuildCargoPairs(ctx);
}

export function rebuildCargoPairs(ctx: Ctx): void {
  const s = ctx.s; ctx.cargoPairs = [];
  for (const l of s.lines) {
    if (!l.valid || !l.cargoKind) continue;
    const stopsU = Array.from(new Set(l.stops));
    for (const src of stopsU) {
      const ns = s.nodes.find(n => n.id === src); if (!ns || !ns.station) continue;
      for (const ind of s.world.industries) {
        if (ind.output !== l.cargoKind) continue;
        if (Math.hypot(ind.x - ns.x, ind.y - ns.y) > 1800) continue;
        for (const dst of stopsU) {
          if (dst === src) continue;
          const nd = s.nodes.find(n => n.id === dst); if (!nd) continue;
          // destino: industria que consume ese cargo o ciudad (bienes finales)
          const accInd = s.world.industries.find(i => i.input === l.cargoKind && Math.hypot(i.x - nd.x, i.y - nd.y) < 1800);
          const city = CITY_CARGO.includes(l.cargoKind) && s.world.cities.some(c => Math.hypot(c.x - nd.x, c.y - nd.y) < 2000 + Math.sqrt(c.pop) * 4);
          if (accInd || city) ctx.cargoPairs.push({ lineId: l.id, src, dst, cargo: l.cargoKind, industry: ind.id, toCity: !accInd && city });
        }
      }
    }
  }
}

/** Genera pasajeros y carga durante dtDays. */
export function generate(ctx: Ctx, dtDays: number): void {
  const s = ctx.s;
  const mult = demandMultiplier(s);
  for (const [src, list] of ctx.rates) {
    const n = ctx.nodeMap.get(src); if (!n) continue;
    const cap = BAL.stationCap[n.level] * 1.5;
    let total = 0; for (const k in n.waiting) total += n.waiting[k];
    const room = Math.max(0, cap - total);
    let gen = 0;
    for (const pr of list) {
      const g = pr.rate * mult * dtDays;
      n.waiting[pr.dst] = (n.waiting[pr.dst] ?? 0) + g; gen += g;
    }
    // acumula demanda potencial por ciudad
    const cov = ctx.cover.get(src);
    if (cov) for (const [cid, frac] of cov.cities) { const c = s.world.cities[cid]; if (c) c.potential += gen * frac; }
    if (gen > room) { // saturación: los que no caben se pierden
      const over = gen - room; n.lost += over;
      for (const pr of list) n.waiting[pr.dst] = Math.max(0, n.waiting[pr.dst] - over * (pr.rate / Math.max(1e-9, list.reduce((a, b) => a + b.rate, 0))));
    }
    // abandono por espera
    const decay = 1 - BAL.waitDecay * dtDays;
    for (const k in n.waiting) { const before = n.waiting[k]; n.waiting[k] = before * decay; n.lost += before - n.waiting[k]; if (n.waiting[k] < 0.01) delete n.waiting[k]; }
  }
  // industrias: producción continua
  for (const ind of s.world.industries) {
    const primary = ind.input === null;
    const eff = primary ? 1 : Math.min(1.5, ind.supplied / Math.max(1, ind.rate * 0.6));
    ind.stock += ind.rate * ind.level * eff / 30 * dtDays;
    if (ind.stock > ind.rate * 3) ind.stock = ind.rate * 3;
  }
  // mueve stock a las estaciones con pares de carga
  if (ctx.cargoPairs.length) {
    const byInd = new Map<number, CargoPair[]>();
    for (const p of ctx.cargoPairs) (byInd.get(p.industry) ?? byInd.set(p.industry, []).get(p.industry)!).push(p);
    for (const [iid, pairs] of byInd) {
      const ind = s.world.industries[iid];
      const move = Math.min(ind.stock, ind.rate * ind.level / 30 * dtDays * 3);   // recogida hasta 3× ritmo si hay stock
      if (move <= 0) continue;
      ind.stock -= move;
      const share = move / pairs.length;
      for (const p of pairs) {
        const n = ctx.nodeMap.get(p.src); if (!n) continue;
        const key = `${p.cargo}:${p.dst}`;
        n.cargo[key] = Math.min(600, (n.cargo[key] ?? 0) + share);
      }
    }
  }
}

export function fareFor(s: GameState, km: number, fareMult: number, comfort: number): number {
  const era = eraOf(s.era);
  return era.fare * (BAL.fareBase + BAL.fareKm * km) * fareMult * (1 + 0.05 * comfort) * (1 + 0.04 * (s.prestige.spent.revenue ?? 0));
}
export function cargoFareFor(s: GameState, km: number): number {
  const era = eraOf(s.era);
  return era.cargoFare * (1 + BAL.cargoKm * km) * (1 + 0.04 * (s.prestige.spent.revenue ?? 0));
}
