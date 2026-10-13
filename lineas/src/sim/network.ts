import type { GameState, Node, Track, Line, TrackKind, Vec2 } from './types';

/** Vías operativas que tocan un nodo. */
export function tracksAt(s: GameState, nodeId: number): Track[] { return s.tracks.filter(t => t.a === nodeId || t.b === nodeId); }

export function nodeById(s: GameState, id: number): Node | undefined { return s.nodes.find(n => n.id === id); }
export function lineById(s: GameState, id: number): Line | undefined { return s.lines.find(l => l.id === id); }

export function compatible(kind: TrackKind, track: TrackKind): boolean {
  if (kind === track) return true;
  if (kind === 'rail' && track === 'hsr') return true;   // trenes convencionales pueden usar vía AV
  return false;
}

/** Dijkstra sobre nodos usando vías del tipo adecuado (operativas o no: la línea se valida al operar). */
export function shortestPath(s: GameState, from: number, to: number, kind: TrackKind): { nodes: number[]; tracks: Track[] } | null {
  if (from === to) return { nodes: [from], tracks: [] };
  const dist = new Map<number, number>(); const prev = new Map<number, { n: number; t: Track }>();
  const open = new Set<number>([from]); dist.set(from, 0);
  const adj = new Map<number, Track[]>();
  for (const t of s.tracks) {
    if (!compatible(kind, t.kind)) continue;
    (adj.get(t.a) ?? adj.set(t.a, []).get(t.a)!).push(t);
    (adj.get(t.b) ?? adj.set(t.b, []).get(t.b)!).push(t);
  }
  const done = new Set<number>();
  while (open.size) {
    let u = -1, du = Infinity;
    for (const n of open) { const d = dist.get(n)!; if (d < du) { du = d; u = n; } }
    open.delete(u); done.add(u);
    if (u === to) break;
    for (const t of adj.get(u) ?? []) {
      const v = t.a === u ? t.b : t.a;
      if (done.has(v)) continue;
      const nd = du + t.length;
      if (nd < (dist.get(v) ?? Infinity)) { dist.set(v, nd); prev.set(v, { n: u, t }); open.add(v); }
    }
  }
  if (!dist.has(to)) return null;
  const nodes = [to]; const tracks: Track[] = [];
  let cur = to;
  while (cur !== from) { const p = prev.get(cur)!; tracks.push(p.t); cur = p.n; nodes.push(cur); }
  nodes.reverse(); tracks.reverse();
  return { nodes, tracks };
}

/** Reconstruye la geometría de una línea (bucle ida y vuelta o circular). */
export function rebuildLine(s: GameState, line: Line): void {
  line.route = []; line.cum = []; line.stopDist = []; line.stopNodes = []; line.total = 0; line.valid = false;
  const stops = line.stops.filter(id => nodeById(s, id));
  line.stops = stops;
  if (stops.length < 2) return;
  const seq = line.circular ? [...stops, stops[0]] : [...stops, ...stops.slice(0, -1).reverse()];
  const route: Vec2[] = []; const cum: number[] = []; const stopDist: number[] = []; const stopNodes: number[] = [];
  let acc = 0;
  const pushPt = (p: Vec2) => {
    if (route.length) { const q = route[route.length - 1]; const d = Math.hypot(p.x - q.x, p.y - q.y); if (d < 0.5) return; acc += d; }
    route.push({ x: p.x, y: p.y }); cum.push(acc);
  };
  for (let i = 0; i < seq.length - 1; i++) {
    const a = seq[i], b = seq[i + 1];
    const sp = shortestPath(s, a, b, line.kind);
    if (!sp) return;
    const na = nodeById(s, a)!;
    if (i === 0) pushPt(na);
    stopDist.push(acc); stopNodes.push(a);
    let cur = a;
    for (const t of sp.tracks) {
      const pts = t.a === cur ? t.path : [...t.path].reverse();
      for (let k = 1; k < pts.length; k++) pushPt(pts[k]);
      cur = t.a === cur ? t.b : t.a;
    }
  }
  // última parada (vuelta al origen) → bucle cerrado
  stopDist.push(acc); stopNodes.push(seq[seq.length - 1]);
  line.route = route; line.cum = cum; line.stopDist = stopDist; line.stopNodes = stopNodes; line.total = acc;
  line.valid = acc > 10;
}

export function lineUsesTrack(s: GameState, line: Line, trackId: number): boolean {
  if (!line.valid) return false;
  for (let i = 0; i < line.stops.length - 1; i++) {
    const sp = shortestPath(s, line.stops[i], line.stops[i + 1], line.kind);
    if (sp && sp.tracks.some(t => t.id === trackId)) return true;
  }
  return false;
}

/** Posición en el mundo a lo largo de la ruta de una línea. */
export function posAlong(line: Line, d: number): { x: number; y: number; ang: number } {
  const cum = line.cum, r = line.route;
  if (r.length < 2) return { x: r[0]?.x ?? 0, y: r[0]?.y ?? 0, ang: 0 };
  d = ((d % line.total) + line.total) % line.total;
  let lo = 0, hi = cum.length - 1;
  while (lo < hi - 1) { const m = (lo + hi) >> 1; if (cum[m] <= d) lo = m; else hi = m; }
  const a = r[lo], b = r[lo + 1] ?? r[0];
  const seg = (cum[lo + 1] ?? line.total) - cum[lo];
  const t = seg > 0 ? (d - cum[lo]) / seg : 0;
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, ang: Math.atan2(b.y - a.y, b.x - a.x) };
}

/**
 * Tabla de transbordos: para cada estación origen, el siguiente nodo hacia cada destino.
 * Grafo de paradas consecutivas de todas las líneas válidas (BFS ponderado por nº de saltos y
 * penalización por transbordo entre líneas).
 */
export interface Routing { next: Map<number, Map<number, number>>; reach: Map<number, Set<number>> }

export function buildRouting(s: GameState): Routing {
  const adj = new Map<number, Set<number>>();
  const add = (a: number, b: number) => { (adj.get(a) ?? adj.set(a, new Set()).get(a)!).add(b); };
  for (const l of s.lines) {
    if (!l.valid || l.cargoKind) continue;
    for (let i = 0; i < l.stopNodes.length - 1; i++) { add(l.stopNodes[i], l.stopNodes[i + 1]); add(l.stopNodes[i + 1], l.stopNodes[i]); }
  }
  const next = new Map<number, Map<number, number>>(); const reach = new Map<number, Set<number>>();
  for (const src of adj.keys()) {
    const prev = new Map<number, number>(); const q = [src]; prev.set(src, -1);
    for (let qi = 0; qi < q.length; qi++) { const u = q[qi]; for (const v of adj.get(u) ?? []) if (!prev.has(v)) { prev.set(v, u); q.push(v); } }
    const nx = new Map<number, number>(); const rs = new Set<number>();
    for (const dst of prev.keys()) {
      if (dst === src) continue;
      let cur = dst, p = prev.get(cur)!;
      while (p !== src) { cur = p; p = prev.get(cur)!; }
      nx.set(dst, cur); rs.add(dst);
    }
    next.set(src, nx); reach.set(src, rs);
  }
  return { next, reach };
}

export const LINE_COLORS = [0x1f6fd0, 0xd9452b, 0x2a9d5c, 0xe39b1a, 0x8a4fc7, 0x14a2a8, 0xc23a7e, 0x6b5b3e, 0x3d6b1f, 0x8c8c8c];
