import type { Vehicle, Line, Node } from './types';
import { BAL, VSCALE, SEC_PER_DAY } from './constants';
import { model, eraOf } from './data';
import { fareFor, cargoFareFor } from './demand';
import type { Ctx } from './game';

export function effSpeedKmh(v: Vehicle): number {
  const m = model(v.model); if (!m) return 0;
  return m.speed * (1 + 0.08 * v.upgrades.engine + 0.02 * v.upgrades.aero) * (1 - 0.25 * Math.max(0, v.wear - 0.6));
}
export function effAccel(v: Vehicle): number { const m = model(v.model); return (m?.accel ?? 0.8) * (1 + 0.15 * v.upgrades.brakes); }
export function effConsumption(v: Vehicle): number {
  const m = model(v.model); if (!m) return 0;
  return m.consumption * (1 + 0.08 * v.upgrades.engine - 0.06 * v.upgrades.aero + 0.02 * v.upgrades.comfort);
}
export function effReliability(v: Vehicle, workshop: boolean): number {
  const m = model(v.model); if (!m) return 0.9;
  return Math.min(0.995, m.reliability + 0.015 * v.upgrades.brakes + (workshop ? 0.02 : 0));
}
export function capacityOf(v: Vehicle): number { return model(v.model)?.capacity ?? 0; }

export function bookValue(v: Vehicle, day: number): number {
  const years = (day - v.bought) / 360;
  return Math.max(v.price * 0.1, v.price * (1 - years / BAL.depreciationYears));
}

/** Avanza todos los vehículos. dt en segundos reales (a 1×), dtDays en días. */
export function updateVehicles(ctx: Ctx, dt: number, dtDays: number): void {
  const s = ctx.s;
  const workshop = s.researched.includes('workshop');
  const fuel = eraOf(s.era).fuel * s.fuel * BAL.fuelPrice;
  for (const v of s.vehicles) {
    if (v.state === 'delivery' || v.state === 'broken') {
      v.timer -= dtDays;
      if (v.timer <= 0) { v.state = v.lineId === null ? 'idle' : 'moving'; v.timer = 0; }
      continue;
    }
    if (v.lineId === null) { v.state = 'idle'; continue; }
    const line = ctx.lineMap.get(v.lineId);
    if (!line || !line.valid) { v.state = 'idle'; continue; }
    if (v.state === 'idle') { v.state = 'moving'; v.dist = Math.min(v.dist, line.total); v.nextStop = nextStopFrom(line, v.dist); }
    if (v.state === 'dwell') {
      v.timer -= dtDays;
      if (v.timer <= 0) {
        v.timer = 0; v.state = 'moving';
        v.nextStop = (v.nextStop + 1) % (line.stopDist.length);
        if (v.nextStop === 0) { v.dist = 0; v.nextStop = 1; }
        v.lastDepart = s.day;
      }
      continue;
    }
    // moving
    const m = model(v.model); if (!m) continue;
    const vmax = effSpeedKmh(v) / 3.6 * VSCALE;
    const a = vmax * effAccel(v) / 3;
    const target = line.stopDist[v.nextStop] ?? line.total;
    const remaining = Math.max(0, target - v.dist);
    v.speed = Math.min(vmax, v.speed + a * dt, Math.sqrt(2 * a * remaining) + 0.3);
    let step = v.speed * dt;
    if (step >= remaining) { step = remaining; }
    v.dist += step;
    const km = step / 1000 / VSCALE;        // km "reales" de ficha (la escala visual no cuenta para costes)
    v.km += km; s.stats.km += km; line.mKm += km;
    v.wear = Math.min(1, v.wear + km * BAL.wearPerKm * (1 + 0.1 * v.upgrades.engine));
    const cost = (step / 1000) * effConsumption(v) / 100 * fuel * costMult(s);   // coste por km visual
    s.cash -= cost; line.cost += cost; line.mCost += cost; s.cur.running += cost; s.dayProfit -= cost;
    const col = s.collection.owned[v.model]; if (col) col.km += km;
    // avería
    const rel = effReliability(v, workshop);
    const p = km * BAL.breakdownBase * (1 - rel) * (0.2 + v.wear) * (1 - 0.05 * v.upgrades.brakes);
    if (p > 0 && ctx.rng.chance(p)) {
      v.state = 'broken'; v.timer = ctx.rng.range(BAL.breakdownDays[0], BAL.breakdownDays[1]) * (workshop ? 0.7 : 1);
      v.breakdowns++; s.monthBreakdowns++;
      s.reputation = Math.max(0, s.reputation - 0.4);
      ctx.notify(`Avería: ${m.name} en ${line.name}`, 'warn');
      continue;
    }
    if (step >= remaining - 1e-6) arrive(ctx, v, line);
  }
}

export function costMult(s: { prestige: { spent: Record<string, number> } }): number { return 1 - 0.03 * (s.prestige.spent.cost ?? 0); }

function nextStopFrom(line: Line, dist: number): number {
  for (let i = 1; i < line.stopDist.length; i++) if (line.stopDist[i] >= dist) return i;
  return 1;
}

function arrive(ctx: Ctx, v: Vehicle, line: Line): void {
  const s = ctx.s;
  const idx = v.nextStop;
  const nodeId = line.stopNodes[idx];
  const node = ctx.nodeMap.get(nodeId);
  const m = model(v.model)!;
  v.speed = 0; v.state = 'dwell'; v.trips++;
  v.timer = BAL.dwellDays[m.mode] * (s.researched.includes('dispatch') ? 0.8 : 1) * (node && node.level >= 2 ? 0.85 : 1);
  if (!node || !node.station) { v.timer = 0.01; return; }    // punto de paso
  const nextIdx = (idx + 1) % line.stopDist.length;
  const nextNode = line.stopNodes[nextIdx === 0 ? 1 : nextIdx];
  // puntualidad: intervalo objetivo = total / (nVeh) / velocidad; tolerancia 25 %
  const nVeh = Math.max(1, line.vehicles.length);
  const vmaxS = Math.max(1, effSpeedKmh(v) / 3.6 * VSCALE);
  const expected = (line.total / vmaxS + line.stopDist.length * (BAL.dwellDays[m.mode] * SEC_PER_DAY + vmaxS / (vmaxS * effAccel(v) / 3))) / nVeh;   // s por vuelta / nº vehículos
  const headway = (s.day - (node.lastArrival?.[line.id] ?? s.day)) * SEC_PER_DAY;    // s reales
  if (!node.lastArrival) node.lastArrival = {};
  if (node.lastArrival[line.id] !== undefined) { line.mStops++; if (headway <= expected * 1.3 + 1) line.mOnTime++; }
  node.lastArrival[line.id] = s.day;
  if (line.cargoKind) { arriveCargo(ctx, v, line, node, idx); return; }
  // descarga
  const next = ctx.routing.next.get(nodeId);
  const unloadKeys: number[] = [];
  for (const k in v.load) {
    const dest = Number(k); const n = v.load[k];
    if (dest === nodeId) { unloadKeys.push(dest); line.pax += n; line.mPax += n; s.stats.pax += n; s.dayPax += n; node.boarded += 0; creditCity(ctx, node, n); continue; }
    const hop = next?.get(dest);
    if (hop !== nextNode) { unloadKeys.push(dest); node.waiting[dest] = (node.waiting[dest] ?? 0) + n; }  // transbordo
  }
  for (const k of unloadKeys) { v.loadCount -= v.load[k]; delete v.load[k]; }
  if (v.loadCount < 0.01) v.loadCount = 0;
  // carga
  const cap = capacityOf(v);
  let room = cap - v.loadCount;
  if (next && room > 0.5) {
    for (const k in node.waiting) {
      if (room <= 0.5) break;
      const dest = Number(k);
      if (next.get(dest) !== nextNode) continue;
      const take = Math.min(room, node.waiting[k]);
      if (take < 0.5) continue;
      node.waiting[k] -= take; if (node.waiting[k] < 0.01) delete node.waiting[k];
      v.load[dest] = (v.load[dest] ?? 0) + take; v.loadCount += take; room -= take;
      node.boarded += take;
      // ingreso: distancia hasta el punto de bajada en esta línea
      const exitIdx = exitStop(ctx, line, idx, dest);
      const km = Math.abs((line.stopDist[exitIdx] ?? line.total) - line.stopDist[idx]) / 1000;
      const rev = take * fareFor(s, km, line.fare, v.upgrades.comfort);
      s.cash += rev; line.revenue += rev; line.mRevenue += rev; s.cur.fares += rev; s.dayProfit += rev; s.dayRevenue += rev;
      for (const c of s.contracts) if (c.active && node.cityId === c.cityId) c.progress += take;
    }
  }
  line.mLoadSum += v.loadCount / Math.max(1, cap); line.mLoadN++;
  if (s.tutorial === 4 && line.revenue > 0) s.tutorial = 5;
}

function exitStop(ctx: Ctx, line: Line, fromIdx: number, dest: number): number {
  let i = fromIdx;
  for (let k = 0; k < line.stopNodes.length; k++) {
    const ni = (i + 1) % line.stopNodes.length;
    const nodeId = line.stopNodes[ni === 0 ? 1 : ni];
    if (nodeId === dest) return ni === 0 ? 1 : ni;
    const hop = ctx.routing.next.get(nodeId)?.get(dest);
    const after = line.stopNodes[(ni + 1) % line.stopNodes.length === 0 ? 1 : (ni + 1) % line.stopNodes.length];
    if (hop !== after) return ni === 0 ? 1 : ni;
    i = ni;
  }
  return fromIdx;
}

function creditCity(ctx: Ctx, node: Node, n: number): void {
  const c = node.cityId !== null ? ctx.s.world.cities[node.cityId] : null;
  if (c) c.served += n;
}

function arriveCargo(ctx: Ctx, v: Vehicle, line: Line, node: Node, idx: number): void {
  const s = ctx.s; const cargo = line.cargoKind!;
  // descarga: carga cuyo destino es este nodo
  const destKey = node.id;
  if (v.cargoLoad > 0 && v.cargoDest === destKey) {
    const t = v.cargoLoad;
    const km = v.cargoKm;
    const rev = t * cargoFareFor(s, km);
    s.cash += rev; line.revenue += rev; line.mRevenue += rev; s.cur.cargo += rev; s.dayProfit += rev; s.dayRevenue += rev;
    line.cargo += t; line.mCargo += t; s.stats.cargo += t; s.dayCargo += t;
    // entrega: industria receptora o ciudad
    const acc = s.world.industries.find(i => i.input === cargo && Math.hypot(i.x - node.x, i.y - node.y) < 1800);
    if (acc) acc.supplied += t;
    else { const c = node.cityId !== null ? s.world.cities[node.cityId] : null; if (c) c.served += t * 2; }
    v.cargoLoad = 0; v.cargoDest = -1; v.cargoKm = 0;
  }
  if (v.cargoLoad > 0) { line.mLoadSum += v.cargoLoad / capacityOf(v); line.mLoadN++; return; }
  // carga: coge el destino con más toneladas esperando que esté en la línea
  let bestKey = '', best = 0;
  for (const k in node.cargo) { if (!k.startsWith(cargo + ':')) continue; if (node.cargo[k] > best) { best = node.cargo[k]; bestKey = k; } }
  if (bestKey && best >= 1) {
    const dest = Number(bestKey.split(':')[1]);
    const cap = capacityOf(v);
    const take = Math.min(cap, best);
    node.cargo[bestKey] -= take; if (node.cargo[bestKey] < 0.01) delete node.cargo[bestKey];
    v.cargoLoad = take; v.cargoDest = dest;
    const exitIdx = line.stopNodes.indexOf(dest, idx + 1);
    const ex = exitIdx >= 0 ? exitIdx : line.stopNodes.indexOf(dest);
    v.cargoKm = Math.abs((line.stopDist[ex] ?? line.total) - line.stopDist[idx]) / 1000;
  }
  line.mLoadSum += v.cargoLoad / Math.max(1, capacityOf(v)); line.mLoadN++;
}
