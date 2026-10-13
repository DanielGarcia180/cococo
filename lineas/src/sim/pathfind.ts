import type { World, TrackKind, Vec2 } from './types';
import { heightAt } from './world';

export interface TrackSpec { base: number; tunnel: number; bridge: number; kmPerDay: number; maxSlope: number; upkeep: number }
/** Coste €/m por tipo de vía y reglas. */
export const TRACK_SPEC: Record<TrackKind, TrackSpec> = {
  road:  { base: 9,   tunnel: 120, bridge: 70,  kmPerDay: 0.6, maxSlope: 0.12, upkeep: 0.5 },
  rail:  { base: 14,  tunnel: 160, bridge: 100, kmPerDay: 0.4, maxSlope: 0.04, upkeep: 0.9 },
  metro: { base: 200, tunnel: 200, bridge: 200, kmPerDay: 0.15, maxSlope: 1, upkeep: 2.5 },
  hsr:   { base: 55,  tunnel: 380, bridge: 260, kmPerDay: 0.3, maxSlope: 0.025, upkeep: 1.6 },
};

export interface RouteResult { path: Vec2[]; length: number; tunnel: number; bridge: number; cost: number; days: number }

/** Traza una vía entre dos puntos sobre la rejilla del mundo. El metro va recto bajo tierra. */
export function routeTrack(w: World, a: Vec2, b: Vec2, kind: TrackKind): RouteResult | null {
  const spec = TRACK_SPEC[kind];
  if (kind === 'metro') {
    const length = Math.hypot(b.x - a.x, b.y - a.y);
    const n = Math.max(2, Math.ceil(length / 400));
    const path: Vec2[] = [];
    for (let i = 0; i <= n; i++) path.push({ x: a.x + (b.x - a.x) * i / n, y: a.y + (b.y - a.y) * i / n });
    const cost = length * spec.base;
    return { path, length, tunnel: length, bridge: 0, cost, days: Math.ceil(length / 1000 / spec.kmPerDay) + 5 };
  }
  const s = w.size, cell = w.cell;
  const toCell = (p: Vec2) => ({ x: Math.max(0, Math.min(s - 1, Math.round(p.x / cell))), y: Math.max(0, Math.min(s - 1, Math.round(p.y / cell))) });
  const ca = toCell(a), cb = toCell(b);
  const start = ca.y * s + ca.x, goal = cb.y * s + cb.x;
  if (start === goal) {
    const length = Math.max(1, Math.hypot(b.x - a.x, b.y - a.y));
    return { path: [a, b], length, tunnel: 0, bridge: 0, cost: length * spec.base, days: 2 };
  }
  // A* con heap binario
  const g = new Float64Array(s * s).fill(Infinity);
  const came = new Int32Array(s * s).fill(-1);
  const closed = new Uint8Array(s * s);
  const heap: number[] = []; const hf: number[] = [];
  const push = (k: number, f: number) => { heap.push(k); hf.push(f); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (hf[p] <= hf[i]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; [hf[p], hf[i]] = [hf[i], hf[p]]; i = p; } };
  const pop = () => { const k = heap[0]; const lk = heap.pop()!, lf = hf.pop()!; if (heap.length) { heap[0] = lk; hf[0] = lf; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && hf[l] < hf[m]) m = l; if (r < heap.length && hf[r] < hf[m]) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; [hf[m], hf[i]] = [hf[i], hf[m]]; i = m; } } return k; };
  const H = w.height, sea = w.seaLevel;
  const heur = (k: number) => { const x = k % s, y = (k / s) | 0; return Math.hypot(x - cb.x, y - cb.y) * cell * spec.base * 0.95; };
  const segCost = (k1: number, k2: number, len: number) => {
    const h1 = H[k1], h2 = H[k2];
    const water = h2 <= sea;
    const slope = Math.abs(h2 - h1) / len;
    let c = spec.base;
    if (water) c = spec.bridge;
    else if (slope > spec.maxSlope * 2.5) c = spec.tunnel;
    else if (slope > spec.maxSlope) c = spec.base * (1 + (slope - spec.maxSlope) * 25);
    else c = spec.base * (1 + slope * 3);
    return c * len;
  };
  g[start] = 0; push(start, heur(start));
  const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
  let found = false, expanded = 0;
  while (heap.length) {
    const k = pop();
    if (closed[k]) continue;
    closed[k] = 1;
    if (k === goal) { found = true; break; }
    if (++expanded > s * s) break;
    const x = k % s, y = (k / s) | 0;
    for (const [dx, dy] of dirs) {
      const X = x + dx, Y = y + dy;
      if (X < 0 || Y < 0 || X >= s || Y >= s) continue;
      const k2 = Y * s + X;
      if (closed[k2]) continue;
      const len = (dx && dy) ? cell * Math.SQRT2 : cell;
      // esquinas: penaliza cambios de dirección para trazados suaves
      let turn = 0;
      const prev = came[k];
      if (prev >= 0) { const px = prev % s, py = (prev / s) | 0; const ddx = x - px, ddy = y - py; if (ddx !== dx || ddy !== dy) turn = spec.base * cell * 0.35; }
      const ng = g[k] + segCost(k, k2, len) + turn;
      if (ng < g[k2]) { g[k2] = ng; came[k2] = k; push(k2, ng + heur(k2)); }
    }
  }
  if (!found) return null;
  const cells: number[] = [];
  for (let k = goal; k >= 0; k = came[k]) cells.push(k);
  cells.reverse();
  let raw: Vec2[] = cells.map(k => ({ x: (k % s) * cell, y: ((k / s) | 0) * cell }));
  raw[0] = { x: a.x, y: a.y }; raw[raw.length - 1] = { x: b.x, y: b.y };
  raw = simplify(raw);
  const path = chaikin(chaikin(raw));
  path[0] = { x: a.x, y: a.y }; path[path.length - 1] = { x: b.x, y: b.y };
  // mide y clasifica
  let length = 0, tunnel = 0, bridge = 0, cost = 0;
  for (let i = 1; i < path.length; i++) {
    const p = path[i - 1], q = path[i];
    const len = Math.hypot(q.x - p.x, q.y - p.y);
    const h1 = heightAt(w, p.x, p.y), h2 = heightAt(w, q.x, q.y);
    const slope = Math.abs(h2 - h1) / Math.max(1, len);
    let c = spec.base;
    if (h2 <= sea || h1 <= sea) { c = spec.bridge; bridge += len; }
    else if (slope > spec.maxSlope * 2.5) { c = spec.tunnel; tunnel += len; }
    else if (slope > spec.maxSlope) c = spec.base * (1 + (slope - spec.maxSlope) * 25);
    else c = spec.base * (1 + slope * 3);
    length += len; cost += c * len;
  }
  const days = Math.ceil(length / 1000 / spec.kmPerDay) + 2 + Math.ceil((tunnel + bridge) / 1000 * 6);
  return { path, length, tunnel, bridge, cost: Math.round(cost), days };
}

function simplify(pts: Vec2[]): Vec2[] {
  if (pts.length < 3) return pts;
  const out = [pts[0]];
  for (let i = 1; i < pts.length - 1; i++) {
    const a = out[out.length - 1], b = pts[i], c = pts[i + 1];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (Math.abs(cross) > 1e-6) out.push(b);
  }
  out.push(pts[pts.length - 1]);
  return out;
}

export function chaikin(pts: Vec2[]): Vec2[] {
  if (pts.length < 3) return pts;
  const out: Vec2[] = [pts[0]];
  for (let i = 0; i < pts.length - 1; i++) {
    const p = pts[i], q = pts[i + 1];
    out.push({ x: p.x * 0.75 + q.x * 0.25, y: p.y * 0.75 + q.y * 0.25 });
    out.push({ x: p.x * 0.25 + q.x * 0.75, y: p.y * 0.25 + q.y * 0.75 });
  }
  out.push(pts[pts.length - 1]);
  return out;
}
