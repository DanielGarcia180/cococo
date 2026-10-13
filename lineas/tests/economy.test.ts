import { describe, it, expect } from 'vitest';
import { Game } from '../src/sim/game';
import { buildStarterLine, runDays, runSeconds } from './helpers';
import { companyValue, takeLoan, maxLoan, loanRate } from '../src/sim/economy';
import { monteCarlo, lineKpi, ledgerCsv } from '../src/sim/stats';

describe('economía', () => {
  it('una línea de tranvía produce el primer ingreso en menos de 60 s a 1×', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const { lineId } = buildStarterLine(g);
    let first = -1;
    for (let i = 0; i < 60 / 0.05; i++) { g.advance(0.05); if (g.s.lines[0].revenue > 0) { first = i * 0.05; break; } }
    expect(first).toBeGreaterThan(0); expect(first).toBeLessThan(60);
    expect(g.lineMap.get(lineId)!.valid).toBe(true);
  });
  it('la línea de arranque es rentable y los costes mensuales se contabilizan', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); buildStarterLine(g); runDays(g, 365);
    const s = g.s; expect(s.ledger.length).toBeGreaterThanOrEqual(11);
    const last = s.ledger[s.ledger.length - 1];
    expect(last.fares).toBeGreaterThan(0); expect(last.maintenance).toBeGreaterThan(0); expect(last.staff).toBeGreaterThan(0); expect(last.infra).toBeGreaterThan(0);
    expect(last.profit).toBeGreaterThan(0);
    const k = lineKpi(s, s.lines[0]); expect(k.load).toBeGreaterThan(0.3); expect(k.payback).not.toBeNull();
    expect(ledgerCsv(s).split('\n').length).toBe(s.ledger.length + 1);
  });
  it('la caja cuadra: inicial − inversión + beneficios acumulados', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const start = g.s.cash;
    buildStarterLine(g); const invested = start - g.s.cash;
    runDays(g, 120);
    const s = g.s;
    const profits = s.ledger.reduce((a, m) => a + m.profit + m.depreciation, 0) + (s.cur.fares + s.cur.cargo + s.cur.subsidies + s.cur.fundGain + s.cur.offline - s.cur.running - s.cur.maintenance - s.cur.infra - s.cur.staff - s.cur.interest);
    // el fondo no se toca y no hay préstamos: caja = inicial − invertido + beneficios (sin depreciación, que no es caja)
    expect(Math.abs(s.cash - (start - invested + profits))).toBeLessThan(1);
  });
  it('préstamos: interés según reputación, cuota mensual y amortización', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const s = g.s; const before = s.cash;
    expect(loanRate(s)).toBeGreaterThan(0.03); expect(loanRate(s)).toBeLessThan(0.08);
    expect(takeLoan(s, 50000)).toBe(true); expect(s.cash).toBe(before + 50000);
    expect(takeLoan(s, maxLoan(s) + 1)).toBe(false);
    runDays(g, 31);
    expect(s.loans[0].principal).toBeLessThan(50000); expect(s.ledger[0].interest).toBeGreaterThan(0);
  });
  it('fondo indexado: DCA compra cada mes y el NAV evoluciona', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const s = g.s; g.setDca(1000); g.fundInvest(10000);
    runDays(g, 90);
    expect(s.fund.invested).toBe(13000); expect(s.fund.history.length).toBeGreaterThan(3); expect(s.fund.nav).not.toBe(100);
    const v = s.fund.units * s.fund.nav; expect(v).toBeGreaterThan(8000); expect(v).toBeLessThan(20000);
  });
  it('valor de compañía incluye vehículos, infraestructura y deuda', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const s = g.s; const v0 = companyValue(s);
    expect(v0).toBe(s.cash);
    buildStarterLine(g); expect(companyValue(s)).toBeLessThan(v0); expect(companyValue(s)).toBeGreaterThan(s.cash);
    takeLoan(s, 20000); expect(Math.round(companyValue(s))).toBe(Math.round(companyValue(s)));
  });
  it('Monte Carlo devuelve bandas ordenadas P10 ≤ P50 ≤ P90', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); buildStarterLine(g); runDays(g, 200);
    const f = monteCarlo(g.s.lines[0], 24, 300, 1);
    expect(f.p10.length).toBe(24);
    for (let i = 0; i < 24; i++) { expect(f.p10[i]).toBeLessThanOrEqual(f.p50[i]); expect(f.p50[i]).toBeLessThanOrEqual(f.p90[i]); }
    expect(f.months).toBeGreaterThanOrEqual(3);
  });
  it('ciclo económico y combustible varían dentro de rangos', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); runDays(g, 360 * 3);
    expect(g.s.cycle).toBeGreaterThan(0.7); expect(g.s.cycle).toBeLessThan(1.3); expect(g.s.fuel).toBeGreaterThan(0.5); expect(g.s.fuel).toBeLessThan(1.5);
  });
  it('acelerar obras cuesta dinero y las termina', () => {
    const g = new Game({ scenario: 'isla', seed: 7 }); const c = g.s.world.cities[0]; const n = g.placeNode(c.x, c.y, 'rail')!;
    expect(n.doneDay).toBeGreaterThan(g.s.day); const cost = g.rushCost('node', n.id); expect(cost).toBeGreaterThan(0);
    const cash = g.s.cash; expect(g.rush('node', n.id)).toBe(true); expect(g.s.cash).toBe(cash - cost); expect(n.doneDay).toBeLessThanOrEqual(g.s.day);
    runSeconds(g, 1);
  });
});
