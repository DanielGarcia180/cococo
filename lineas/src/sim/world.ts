import { Rng, valueNoise2D, fbm } from './rng';
import type { World, City, Industry, Zone, Cargo, ZoneKind } from './types';

export interface WorldParams {
  seed: number; name: string; size?: number; cell?: number;
  seaLevel?: number;              // 0..1 fracción del rango de ruido
  cities?: number; relief?: number; coast?: 'island' | 'coast' | 'inland';
  fixedCities?: { name: string; x: number; y: number; pop: number }[];   // coords en 0..1
  industries?: number;
  popRange?: [number, number]; bigPop?: [number, number];
}

const NAMES_ES = ['Alcalá', 'Ribera', 'Monforte', 'Valdés', 'Puente Nuevo', 'Santa Clara', 'Olivar', 'Las Rozas', 'Torrelago',
  'Cabrera', 'Miranda', 'Portocelo', 'Villafría', 'Castrillón', 'Benalúa', 'Arenales', 'Robledo', 'Fuencisla', 'Almenar', 'Serrada',
  'Lorquí', 'Noguera', 'Cardeña', 'Peñalba', 'Sotillo', 'Vallejera', 'Marmolejo', 'Aldeanueva', 'Calzada', 'Hontanar'];

const CHAINS: { kind: string; input: Cargo | null; output: Cargo; rate: number; name: string }[] = [
  { kind: 'mine', input: null, output: 'ore', rate: 120, name: 'Mina' },
  { kind: 'steel', input: 'ore', output: 'steel', rate: 90, name: 'Acería' },
  { kind: 'factory', input: 'steel', output: 'goods', rate: 80, name: 'Fábrica' },
  { kind: 'forest', input: null, output: 'wood', rate: 110, name: 'Bosque' },
  { kind: 'sawmill', input: 'wood', output: 'planks', rate: 90, name: 'Aserradero' },
  { kind: 'farm', input: null, output: 'food', rate: 100, name: 'Granja' },
  { kind: 'oilwell', input: null, output: 'oil', rate: 100, name: 'Pozo' },
  { kind: 'refinery', input: 'oil', output: 'fuel', rate: 80, name: 'Refinería' },
];
export const CHAIN_DEFS = CHAINS;
/** Cargos que consumen las ciudades (dan ingresos al entregarse en una estación urbana). */
export const CITY_CARGO: Cargo[] = ['goods', 'planks', 'food', 'fuel'];

export function heightAt(w: World, x: number, y: number): number {
  const gx = x / w.cell, gy = y / w.cell;
  const x0 = Math.max(0, Math.min(w.size - 2, Math.floor(gx)));
  const y0 = Math.max(0, Math.min(w.size - 2, Math.floor(gy)));
  const fx = Math.max(0, Math.min(1, gx - x0)), fy = Math.max(0, Math.min(1, gy - y0));
  const h = w.height;
  const a = h[y0 * w.size + x0], b = h[y0 * w.size + x0 + 1], c = h[(y0 + 1) * w.size + x0], d = h[(y0 + 1) * w.size + x0 + 1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}
export function isWater(w: World, x: number, y: number): boolean { return heightAt(w, x, y) <= w.seaLevel; }
export function worldExtent(w: World): number { return (w.size - 1) * w.cell; }

export function generateWorld(p: WorldParams): World {
  const size = p.size ?? 128, cell = p.cell ?? 300;
  const rng = new Rng(p.seed);
  const noise = valueNoise2D(p.seed ^ 0x9e3779b9);
  const noise2 = valueNoise2D(p.seed ^ 0x51f15e);
  const relief = p.relief ?? 1;
  const height = new Array(size * size).fill(0);
  const coast = p.coast ?? 'coast';
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const nx = x / size, ny = y / size;
    let h = fbm(noise, nx * 5, ny * 5, 5, 2, 0.5);           // 0..1
    const ridge = Math.abs(fbm(noise2, nx * 3 + 7, ny * 3 + 3, 3) - 0.5) * 2;
    h = h * 0.75 + ridge * 0.25 * relief;
    // forma de la masa de tierra
    const dx = nx - 0.5, dy = ny - 0.5;
    const r = Math.sqrt(dx * dx + dy * dy);
    if (coast === 'island') h -= Math.max(0, r - 0.22) * 1.6;
    else if (coast === 'coast') h -= Math.max(0, (nx - 0.35) * -1.1) * (ny < 0.5 ? 1.2 : 0.4) + Math.max(0, 0.08 - nx) * 2;
    else h += 0.08;
    height[y * size + x] = h;
  }
  // normaliza y escala a metros
  let mn = Infinity, mx = -Infinity;
  for (const v of height) { if (v < mn) mn = v; if (v > mx) mx = v; }
  const seaFrac = p.seaLevel ?? (coast === 'inland' ? 0.08 : 0.34);
  const maxH = 650 * relief;
  for (let i = 0; i < height.length; i++) {
    const t = (height[i] - mn) / (mx - mn);
    // curva: tierra baja extensa, montañas al final
    const s = t < seaFrac ? (t / seaFrac) * 0.3 - 0.3 : Math.pow((t - seaFrac) / (1 - seaFrac), 1.7);
    height[i] = Math.round(s * maxH);
  }
  const world: World = { seed: p.seed, name: p.name, size, cell, seaLevel: 0, height, cities: [], industries: [] };
  carveRivers(world, rng, coast === 'inland' ? 3 : 2);
  // ciudades
  const ext = worldExtent(world);
  const nCities = p.cities ?? 7;
  const cities: City[] = [];
  const tryPlace = (name: string, pop: number, fx?: number, fy?: number): boolean => {
    for (let attempt = 0; attempt < 200; attempt++) {
      const x = fx !== undefined ? fx * ext : rng.range(0.08, 0.92) * ext;
      const y = fy !== undefined ? fy * ext : rng.range(0.08, 0.92) * ext;
      const h = heightAt(world, x, y);
      if (fx === undefined && (h <= 2 || h > 260)) continue;
      if (fx === undefined && slopeAt(world, x, y) > 0.12) continue;
      if (fx === undefined && cities.some(c => Math.hypot(c.x - x, c.y - y) < ext * 0.16)) continue;
      if (fx !== undefined) flattenAround(world, x, y, 2);
      const coastal = [0, 60, 120, 180, 240, 300].some(a => isWater(world, x + Math.cos(a / 180 * Math.PI) * 1500, y + Math.sin(a / 180 * Math.PI) * 1500));
      cities.push({ id: cities.length, name, x, y, pop, zones: [], service: 0, served: 0, potential: 0, coastal });
      return true;
    }
    return false;
  };
  const names = [...NAMES_ES];
  for (let i = names.length - 1; i > 0; i--) { const j = rng.int(0, i); [names[i], names[j]] = [names[j], names[i]]; }
  if (p.fixedCities) for (const c of p.fixedCities) tryPlace(c.name, c.pop, c.x, c.y);
  else {
    const big = rng.int(1, 2);
    const pr = p.popRange ?? [6000, 40000], bp = p.bigPop ?? [60000, 140000];
    for (let i = 0; i < nCities; i++) {
      const pop = i < big ? rng.int(bp[0], bp[1]) : rng.int(pr[0], pr[1]);
      tryPlace(names[i % names.length], pop);
    }
  }
  for (const c of cities) c.zones = makeZones(c, rng, world);
  world.cities = cities;
  // industrias: cadenas completas cerca de ciudades distintas
  const nInd = p.industries ?? Math.max(4, Math.round(cities.length * 1.3));
  const inds: Industry[] = [];
  const chainSets = [['mine', 'steel', 'factory'], ['forest', 'sawmill'], ['farm'], ['oilwell', 'refinery']];
  let ci = 0;
  while (inds.length < nInd && ci < 40) {
    const chain = chainSets[ci % chainSets.length]; ci++;
    for (const kind of chain) {
      const def = CHAINS.find(d => d.kind === kind)!;
      for (let attempt = 0; attempt < 100; attempt++) {
        const x = rng.range(0.06, 0.94) * ext, y = rng.range(0.06, 0.94) * ext;
        const h = heightAt(world, x, y);
        if (h <= 2) continue;
        if (kind === 'mine' && h < 120) continue;
        if (kind !== 'mine' && h > 300) continue;
        if (cities.some(c => Math.hypot(c.x - x, c.y - y) < 2500)) continue;
        if (inds.some(c => Math.hypot(c.x - x, c.y - y) < 2500)) continue;
        flattenAround(world, x, y, 1);
        inds.push({ id: inds.length, kind, name: `${def.name} ${nameSuffix(inds.length)}`, x, y, input: def.input, output: def.output, rate: def.rate * rng.range(0.8, 1.3), stock: 0, supplied: 0, level: 1 });
        break;
      }
    }
  }
  world.industries = inds;
  return world;
}

function nameSuffix(i: number): string { return ['Norte', 'Sur', 'Este', 'Oeste', 'Alto', 'Bajo', 'Viejo', 'Nuevo', 'Mayor', 'Menor', 'Real', 'Chico'][i % 12]; }

function makeZones(c: City, rng: Rng, w: World): Zone[] {
  const zones: Zone[] = [];
  const n = Math.max(5, Math.min(14, Math.round(Math.sqrt(c.pop) / 25)));
  const radius = 600 + Math.sqrt(c.pop) * 6;
  zones.push({ x: c.x, y: c.y, kind: 'com', w: 0.18 });
  let wsum = 0.18;
  for (let i = 1; i < n; i++) {
    for (let attempt = 0; attempt < 20; attempt++) {
      const a = rng.range(0, Math.PI * 2), r = rng.range(250, radius);
      const x = c.x + Math.cos(a) * r, y = c.y + Math.sin(a) * r;
      if (isWater(w, x, y)) continue;
      const kind: ZoneKind = r < radius * 0.4 ? (rng.chance(0.5) ? 'com' : 'res') : (rng.chance(0.75) ? 'res' : 'ind');
      const wt = rng.range(0.4, 1);
      zones.push({ x, y, kind, w: wt }); wsum += wt;
      break;
    }
  }
  for (const z of zones) z.w /= wsum;
  return zones;
}

export function slopeAt(w: World, x: number, y: number): number {
  const d = w.cell;
  const hx = heightAt(w, x + d, y) - heightAt(w, x - d, y);
  const hy = heightAt(w, x, y + d) - heightAt(w, x, y - d);
  return Math.hypot(hx, hy) / (2 * d);
}

function flattenAround(w: World, x: number, y: number, r: number) {
  const gx = Math.round(x / w.cell), gy = Math.round(y / w.cell);
  const h0 = Math.max(4, heightAt(w, x, y));
  for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) {
    const X = gx + i, Y = gy + j;
    if (X < 0 || Y < 0 || X >= w.size || Y >= w.size) continue;
    const k = Y * w.size + X;
    w.height[k] = Math.round(w.height[k] * 0.3 + h0 * 0.7);
  }
}

function carveRivers(w: World, rng: Rng, n: number) {
  const s = w.size;
  for (let r = 0; r < n; r++) {
    // empieza en una celda alta
    let best = -1, bh = -1;
    for (let t = 0; t < 40; t++) { const k = rng.int(0, s * s - 1); if (w.height[k] > bh) { bh = w.height[k]; best = k; } }
    let x = best % s, y = Math.floor(best / s);
    for (let step = 0; step < s * 3; step++) {
      const k = y * s + x;
      if (w.height[k] <= w.seaLevel) break;
      w.height[k] = Math.min(w.height[k], w.seaLevel);   // el río es agua
      // vecino más bajo (con un poco de azar)
      let nx = x, ny = y, nh = Infinity;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const X = x + dx, Y = y + dy;
        if (X < 0 || Y < 0 || X >= s || Y >= s) continue;
        const h = w.height[Y * s + X] + rng.range(0, 25);
        if (h < nh) { nh = h; nx = X; ny = Y; }
      }
      if (nx === x && ny === y) break;
      if (w.height[ny * s + nx] > w.height[k]) w.height[ny * s + nx] = w.height[k];  // erosiona
      x = nx; y = ny;
    }
  }
}

export function nearestCity(w: World, x: number, y: number, maxDist = 3000): City | null {
  let best: City | null = null, bd = maxDist;
  for (const c of w.cities) {
    const r = 800 + Math.sqrt(c.pop) * 6;
    const d = Math.hypot(c.x - x, c.y - y) - r * 0.5;
    if (d < bd) { bd = d; best = c; }
  }
  return best;
}
