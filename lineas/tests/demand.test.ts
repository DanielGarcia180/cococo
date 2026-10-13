import { describe, it, expect } from 'vitest';
import { Game } from '../src/sim/game';
import { buildStarterLine, runDays } from './helpers';
import { coverOf, priceMult, cityCoverage } from '../src/sim/demand';
import { buildRouting } from '../src/sim/network';

describe('demanda', () => {
  it('la cobertura de una estación en el centro es mayor que en el campo', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const s = g.s; const c = [...s.world.cities].sort((a, b) => b.pop - a.pop)[0];
    const center = g.placeNode(c.x, c.y, 'rail')!; const far = g.placeNode(c.x + 6000, c.y + 6000, 'rail') ?? g.placeNode(c.x - 6000, c.y - 6000, 'rail');
    const cc = coverOf(s, center); expect(cc.P + cc.A).toBeGreaterThan(1000);
    if (far) { const cf = coverOf(s, far); expect(cf.P + cf.A).toBeLessThan(cc.P + cc.A); }
    expect(cityCoverage(s, c)).toBe(0);   // en obras todavía
    g.rush('node', center.id); expect(cityCoverage(s, c)).toBeGreaterThan(0.2);
  });
  it('elasticidad: subir la tarifa reduce la demanda', () => {
    expect(priceMult(1)).toBe(1); expect(priceMult(2)).toBeLessThan(0.5); expect(priceMult(0.5)).toBeGreaterThan(2);
  });
  it('solo se generan viajes entre estaciones conectadas por líneas', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const { a, b } = buildStarterLine(g); const s = g.s;
    const c = s.world.cities[1]; const lonely = g.placeNode(c.x, c.y, 'rail')!; g.rush('node', lonely.id); g.refresh();
    expect(g.rates.has(a.id)).toBe(true); expect(g.rates.get(a.id)!.some(p => p.dst === b.id)).toBe(true);
    expect(g.rates.has(lonely.id)).toBe(false);
    runDays(g, 10);
    expect(Object.keys(lonely.waiting).length).toBe(0); expect(Object.values(a.waiting).reduce((x, y) => x + y, 0)).toBeGreaterThan(0);
  });
  it('tabla de transbordos: el siguiente salto conduce al destino', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const { a, b, lineId } = buildStarterLine(g); const s = g.s;
    const c = g.placeNode(b.x + 1200, b.y + 300, 'rail') ?? g.placeNode(b.x - 1200, b.y - 300, 'rail')!;
    const t = g.buildTrack(b.id, c.id, 'rail')!; g.rush('node', c.id); g.rush('track', t.id);
    const l2 = g.createLine('rail'); g.lineAddStop(l2.id, b.id); g.lineAddStop(l2.id, c.id); g.refresh();
    const r = buildRouting(s);
    expect(r.next.get(a.id)!.get(c.id)).toBe(b.id); expect(r.next.get(c.id)!.get(a.id)).toBe(b.id);
    expect(g.lineMap.get(lineId)!.valid).toBe(true);
  });
  it('los pasajeros abandonan la estación si nadie los recoge y la reputación baja', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const { vehicleId } = buildStarterLine(g); const s = g.s;
    g.sellVehicle(vehicleId); const rep = s.reputation; runDays(g, 90);
    expect(s.reputation).toBeLessThan(rep);
  });
  it('las ciudades con servicio crecen y las que no, decrecen', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); buildStarterLine(g); const s = g.s;
    const served = [...s.world.cities].sort((a, b) => b.pop - a.pop)[0]; const other = s.world.cities.find(c => c !== served)!;
    const p0 = served.pop, q0 = other.pop;
    for (let i = 0; i < 3; i++) g.buyVehicle('tram_1900_0', s.lines[0].id);
    runDays(g, 360 * 2);
    expect(served.pop).toBeGreaterThan(p0); expect(other.pop).toBeLessThan(q0);
  });
  it('cadena de mercancías: una línea de carga mueve toneladas de la mina a la acería', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const s = g.s;
    const mine = s.world.industries.find(i => i.kind === 'mine')!; const steel = s.world.industries.find(i => i.kind === 'steel')!;
    const a = g.placeNode(mine.x + 400, mine.y, 'rail')!; const b = g.placeNode(steel.x + 400, steel.y, 'rail')!;
    s.cash += 5_000_000; const t = g.buildTrack(a.id, b.id, 'rail')!; g.rush('node', a.id); g.rush('node', b.id); g.rush('track', t.id);
    const l = g.createLine('rail', 'ore'); g.lineAddStop(l.id, a.id); g.lineAddStop(l.id, b.id);
    const v = g.buyVehicle('freight_1900_0', l.id)!; g.rush('vehicle', v.id); g.refresh();
    expect(g.cargoPairs.length).toBeGreaterThan(0);
    runDays(g, 120);
    expect(l.cargo).toBeGreaterThan(0); expect(l.revenue).toBeGreaterThan(0); expect(steel.supplied).toBeGreaterThan(0);
  });
});
