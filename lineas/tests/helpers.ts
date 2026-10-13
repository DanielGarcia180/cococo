import { Game } from '../src/sim/game';
import type { Node } from '../src/sim/types';

/** Construye una línea de tranvía de dos paradas dentro de la ciudad más grande y la pone en marcha. */
export function buildStarterLine(g: Game, modelId = 'tram_1900_0'): { a: Node; b: Node; lineId: number; vehicleId: number } {
  const s = g.s;
  const c0 = [...s.world.cities].sort((a, b) => b.pop - a.pop)[0];
  const zs = [...c0.zones].sort((a, b) => Math.hypot(b.x - c0.x, b.y - c0.y) - Math.hypot(a.x - c0.x, a.y - c0.y));
  const z1 = zs[0]; const z2 = zs.find(z => Math.hypot(z.x - z1.x, z.y - z1.y) > 900) ?? zs[1];
  const a = g.placeNode(z1.x, z1.y, 'rail')!; const b = g.placeNode(z2.x, z2.y, 'rail')!;
  const t = g.buildTrack(a.id, b.id, 'rail')!;
  const l = g.createLine('rail'); g.lineAddStop(l.id, a.id); g.lineAddStop(l.id, b.id);
  g.rush('node', a.id); g.rush('node', b.id); g.rush('track', t.id);
  const v = g.buyVehicle(modelId, l.id)!; g.rush('vehicle', v.id);
  g.refresh();
  return { a, b, lineId: l.id, vehicleId: v.id };
}

export function runSeconds(g: Game, seconds: number): void { const n = Math.round(seconds / 0.05); for (let i = 0; i < n; i++) g.advance(0.05); }
export function runDays(g: Game, days: number): void { runSeconds(g, days * 2); }
