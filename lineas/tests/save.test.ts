import { describe, it, expect } from 'vitest';
import { Game } from '../src/sim/game';
import { buildStarterLine, runDays } from './helpers';
import { serialize, deserialize, migrate } from '../src/sim/save';
import { SAVE_VERSION, BAL, SEC_PER_DAY } from '../src/sim/constants';
import { applyOffline, refreshDailies, prestigeSpend, prestigeAvailable } from '../src/sim/progress';
import { applyMod, validateMod, DATA, resetData } from '../src/sim/data';
import { updateMedals } from '../src/sim/challenges';

describe('guardado y carga', () => {
  it('serializar → cargar → continuar produce el mismo resultado que seguir sin guardar', () => {
    const g1 = new Game({ scenario: 'isla', seed: 7 }); buildStarterLine(g1); runDays(g1, 50);
    const json = serialize(g1.s);
    const g2 = new Game(); g2.load(deserialize(json));
    runDays(g1, 50); runDays(g2, 50);
    expect(Math.round(g2.s.cash)).toBe(Math.round(g1.s.cash)); expect(g2.s.lines[0].pax).toBeCloseTo(g1.s.lines[0].pax, 3);
    expect(g2.s.vehicles[0].dist).toBeCloseTo(g1.s.vehicles[0].dist, 3);
  });
  it('rechaza archivos inválidos y migra versiones antiguas', () => {
    expect(() => deserialize('{"foo":1}')).toThrow();
    const g = new Game({ scenario: 'isla', seed: 7 }); const s = JSON.parse(serialize(g.s)); s.version = 1; delete s.dayRevenue; delete s.cur.offline;
    const m = migrate(s); expect(m.version).toBe(SAVE_VERSION); expect(m.dayRevenue).toBe(0); expect(m.cur.offline).toBe(0);
  });
  it('el guardado no incluye rutas (se regeneran) y pesa poco', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); buildStarterLine(g); runDays(g, 30);
    const json = serialize(g.s); expect(json.length).toBeLessThan(400_000);
    const g2 = new Game(); g2.load(deserialize(json)); expect(g2.s.lines[0].route.length).toBeGreaterThan(2); expect(g2.s.lines[0].valid).toBe(true);
  });
});

describe('progreso offline', () => {
  it('aplica beneficio medio × días × eficiencia, con tope de 8 h, y completa obras', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); buildStarterLine(g); runDays(g, 60);
    const s = g.s; const c = s.world.cities[1]; const n = g.placeNode(c.x, c.y, 'rail')!;
    const avg = s.profitWindow.reduce((a, b) => a + b, 0) / s.profitWindow.length; expect(avg).toBeGreaterThan(0);
    const t0 = 1_000_000_000_000; s.lastRealTime = t0; const cash = s.cash; const day = s.day;
    applyOffline(g, t0 + 3 * 3600 * 1000);      // 3 horas
    const days = 3 * 3600 / SEC_PER_DAY;
    expect(s.day).toBeCloseTo(day + days, 6); expect(s.cash).toBeCloseTo(cash + avg * days * BAL.offlineEff, 3);
    expect(s.offline!.completed).toContain(n.name);
    // tope 8 h
    s.lastRealTime = t0; const d2 = s.day; applyOffline(g, t0 + 30 * 3600 * 1000);
    expect(s.day - d2).toBeCloseTo(8 * 3600 / SEC_PER_DAY, 6);
    // menos de 2 minutos: nada
    s.lastRealTime = t0; const d3 = s.day; applyOffline(g, t0 + 60 * 1000); expect(s.day).toBe(d3);
  });
  it('objetivos diarios se regeneran por fecha y la racha cuenta días completos', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const s = g.s;
    const day1 = Date.UTC(2026, 9, 13, 12), day2 = day1 + 86400000, day4 = day1 + 3 * 86400000;
    expect(refreshDailies(s, day1)).toBe(true); expect(s.dailies.length).toBe(3); expect(refreshDailies(s, day1)).toBe(false);
    for (const d of s.dailies) d.done = true;
    refreshDailies(s, day2); expect(s.streak).toBe(1);
    for (const d of s.dailies) d.done = true;
    refreshDailies(s, day4); expect(s.streak).toBe(0);    // saltó un día
  });
  it('prestigio: puntos y árbol permanente', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const s = g.s;
    expect(g.refound('random')).toBe(false);
    s.cash = 12_000_000; expect(prestigeAvailable(s)).toBe(0);
    expect(g.refound('random', 5)).toBe(true);
    expect(g.s.prestige.points).toBe(2); expect(g.s.prestige.runs).toBe(1); expect(g.s.cash).toBe(250000);
    expect(prestigeSpend(g.s, 'cash')).toBe(true); expect(prestigeSpend(g.s, 'cash')).toBe(true); expect(prestigeSpend(g.s, 'cash')).toBe(false);
    expect(g.stationCost('rail', true)).toBeLessThanOrEqual(BAL.stationCost.rail);
  });
});

describe('mods y licencias', () => {
  it('valida y aplica un mod con vehículos nuevos', () => {
    resetData(); const n = DATA.vehicles.length;
    expect(validateMod({})).not.toBeNull(); expect(validateMod({ name: 'x', vehicles: [{ id: 'a' }] })).not.toBeNull();
    const mod = { name: 'Prueba', vehicles: [{ id: 'bus_test', name: 'Bus de prueba', mode: 'bus' as const, track: 'road' as const, price: 1000, speed: 50, capacity: 30 }] };
    expect(validateMod(mod)).toBeNull(); applyMod(mod);
    expect(DATA.vehicles.length).toBe(n + 1); expect(DATA.mods[0].name).toBe('Prueba');
    const g = new Game({ scenario: 'isla', seed: 7 }); expect(g.availableModels().some(m => m.id === 'bus_test')).toBe(true);
    resetData(); expect(DATA.vehicles.length).toBe(n);
  });
  it('las licencias conceden medallas al superar umbrales', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const s = g.s;
    s.cash = 3_000_000; const news = updateMedals(s);
    expect(news.some(x => x.includes('Licencia J'))).toBe(true); expect(s.medals.find(m => m.id === 'L10')!.tier).toBe(1);
  });
});
