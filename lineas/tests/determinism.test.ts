import { describe, it, expect } from 'vitest';
import { Game } from '../src/sim/game';
import { generateWorld } from '../src/sim/world';
import { serialize } from '../src/sim/save';
import { buildStarterLine, runDays } from './helpers';
import { Rng, seedFromAny, hashString } from '../src/sim/rng';

describe('determinismo por semilla', () => {
  it('mulberry32 produce la misma secuencia', () => {
    const a = new Rng(42), b = new Rng(42);
    for (let i = 0; i < 1000; i++) expect(a.next()).toBe(b.next());
    expect(new Rng(42).next()).not.toBe(new Rng(43).next());
  });
  it('seedFromAny: números y textos', () => {
    expect(seedFromAny('123')).toBe(123); expect(seedFromAny('dublin')).toBe(hashString('dublin')); expect(seedFromAny('dublin')).toBe(seedFromAny('dublin'));
  });
  it('misma semilla ⇒ mismo mundo', () => {
    const w1 = generateWorld({ seed: 99, name: 'x', cities: 5 }), w2 = generateWorld({ seed: 99, name: 'x', cities: 5 });
    expect(w1.height).toEqual(w2.height);
    expect(w1.cities.map(c => [c.name, c.x, c.y, c.pop])).toEqual(w2.cities.map(c => [c.name, c.x, c.y, c.pop]));
    expect(w1.industries.length).toBeGreaterThan(0);
    const w3 = generateWorld({ seed: 100, name: 'x', cities: 5 });
    expect(w3.height).not.toEqual(w1.height);
  });
  it('misma semilla y mismas acciones ⇒ mismo estado tras 200 días', () => {
    const run = () => { const g = new Game({ scenario: 'isla', seed: 7 }); buildStarterLine(g); runDays(g, 200); const s = g.s; s.lastRealTime = 0; s.notifications = []; return serialize(s); };
    const a = run(), b = run();
    expect(a).toBe(b);
  });
  it('escenarios prediseñados colocan sus ciudades', () => {
    const g = new Game({ scenario: 'dublin' });
    expect(g.s.world.cities.map(c => c.name)).toContain('Swords');
    expect(g.s.world.cities.map(c => c.name)).toContain('Aeropuerto');
    const e = new Game({ scenario: 'espana' });
    expect(e.s.world.cities.find(c => c.name === 'Madrid')!.pop).toBeGreaterThan(100000);
  });
});
