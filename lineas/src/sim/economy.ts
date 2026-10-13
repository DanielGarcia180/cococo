import type { GameState, Loan, LedgerMonth } from './types';
import { BAL, DAYS_PER_MONTH } from './constants';
import { model, eraOf } from './data';
import { TRACK_SPEC } from './pathfind';
import { bookValue, costMult } from './vehicles';
import type { Ctx } from './game';

export function emptyMonth(month: number, year: number): LedgerMonth {
  return { month, year, fares: 0, cargo: 0, subsidies: 0, fundGain: 0, offline: 0, running: 0, maintenance: 0, infra: 0, staff: 0, interest: 0, depreciation: 0, profit: 0, cash: 0, value: 0, pax: 0, pop: 0 };
}

export function yearOf(s: GameState): number { return s.startYear + Math.floor(s.day / 360); }
export function monthOf(s: GameState): number { return Math.floor((s.day % 360) / DAYS_PER_MONTH); }

export function infraValue(s: GameState): number {
  let v = 0;
  for (const n of s.nodes) v += n.cost;
  for (const t of s.tracks) v += t.cost;
  return v;
}
export function vehiclesValue(s: GameState): number { let v = 0; for (const x of s.vehicles) v += bookValue(x, s.day); return v; }
export function debt(s: GameState): number { let d = 0; for (const l of s.loans) d += l.principal; return d; }
export function fundValue(s: GameState): number { return s.fund.units * s.fund.nav; }
export function companyValue(s: GameState): number { return s.cash + vehiclesValue(s) + infraValue(s) * 0.5 + fundValue(s) - debt(s); }

export function loanRate(s: GameState): number {
  const tech = s.researched.includes('loans2') ? 0.015 : 0;
  return Math.max(BAL.loanRateMin, BAL.loanRateBase - 0.04 * s.reputation / 100 - tech);
}
export function maxLoan(s: GameState): number {
  const base = 150000 * (s.researched.includes('loans2') ? 3 : 1) * (eraOf(s.era).fare / 16);
  return Math.max(0, Math.round(base + companyValue(s) * 0.5 - debt(s)));
}
export function takeLoan(s: GameState, amount: number): boolean {
  if (amount <= 0 || amount > maxLoan(s) || s.loans.length >= 3) return false;
  const r = loanRate(s) / 12, n = BAL.loanYears * 12;
  const payment = amount * r / (1 - Math.pow(1 + r, -n));
  s.loans.push({ id: s.nextId++, principal: amount, rate: loanRate(s), monthsLeft: n, payment });
  s.cash += amount;
  return true;
}
export function repayLoan(s: GameState, id: number): boolean {
  const l = s.loans.find(x => x.id === id); if (!l || s.cash < l.principal) return false;
  s.cash -= l.principal; s.loans = s.loans.filter(x => x.id !== id); return true;
}

/** Cierre de mes: costes fijos, préstamos, fondo, ciclo, ciudades, industrias, historial. */
export function monthEnd(ctx: Ctx): void {
  const s = ctx.s; const era = eraOf(s.era); const cm = costMult(s);
  const scale = era.fare / 16;
  // mantenimiento + personal por vehículo
  let maint = 0, staff = 0, depr = 0;
  const workshop = s.researched.includes('workshop') ? 0.8 : 1;
  const auto = s.researched.includes('autonomous') ? 0.6 : 1;
  for (const v of s.vehicles) {
    const m = model(v.model); if (!m) continue;
    maint += m.maintenance * workshop * BAL.maintenanceFactor * (1 + v.wear * 0.5);
    staff += BAL.staffPerVehicle * scale * auto;
    depr += v.price / BAL.depreciationYears / 12;
    v.wear = Math.max(0, v.wear - 0.03 * workshop);   // mantenimiento rutinario
  }
  let infra = 0;
  for (const n of s.nodes) infra += n.station ? BAL.stationUpkeep[n.kind] * BAL.levelCostMult[n.level] * scale * 0.6 : 20;
  for (const t of s.tracks) infra += TRACK_SPEC[t.kind].upkeep * t.length / 1000 * 30 * scale * 0.5;
  maint *= cm; staff *= cm; infra *= cm;
  s.cash -= maint + staff + infra;
  s.cur.maintenance += maint; s.cur.staff += staff; s.cur.infra += infra; s.cur.depreciation += depr;
  s.dayProfit -= maint + staff + infra;
  // reparte costes fijos entre líneas (por vehículos asignados)
  for (const l of s.lines) {
    const n = l.vehicles.length;
    let lm = 0; for (const vid of l.vehicles) { const v = s.vehicles.find(x => x.id === vid); const m = v && model(v.model); if (m) lm += m.maintenance * workshop * BAL.maintenanceFactor; }
    const c = (lm + n * BAL.staffPerVehicle * scale * auto) * cm + (infra * (s.vehicles.length ? n / s.vehicles.length : 0));
    l.cost += c; l.mCost += c;
  }
  // préstamos
  let interest = 0;
  for (const l of s.loans) {
    const i = l.principal * l.rate / 12; interest += i;
    const amort = Math.min(l.principal, l.payment - i);
    l.principal -= amort; l.monthsLeft--;
    s.cash -= i + amort;
  }
  s.loans = s.loans.filter(l => l.principal > 1 && l.monthsLeft > 0);
  s.cur.interest += interest; s.dayProfit -= interest;
  // fondo indexado
  const f = s.fund;
  const fee = BAL.fundFee * (s.researched.includes('fund') ? 0.5 : 1);
  const before = f.units * f.nav;
  f.nav *= Math.exp((BAL.fundMu - fee - BAL.fundSigma ** 2 / 2) / 12 + BAL.fundSigma / Math.sqrt(12) * ctx.rng.gauss());
  f.nav = Math.max(0.05, f.nav);
  const gain = f.units * f.nav - before;
  s.cur.fundGain += gain; f.history.push(Math.round(f.nav * 10000) / 10000);
  if (f.history.length > 600) f.history.shift();
  if (f.dca > 0 && s.cash >= f.dca) { s.cash -= f.dca; f.units += f.dca / f.nav; f.invested += f.dca; }
  // ciclo económico y combustible
  const t = s.day / 360;
  s.cycleNoise = s.cycleNoise * 0.8 + ctx.rng.gauss() * 0.02;
  s.cycle = 1 + BAL.cycleAmp * Math.sin(2 * Math.PI * t / BAL.cyclePeriod + s.cyclePhase) + s.cycleNoise;
  s.fuel = 1 + BAL.fuelAmp * Math.sin(2 * Math.PI * t / BAL.fuelPeriod + s.fuelPhase) + ctx.rng.gauss() * 0.03;
  if (s.cycle < 0.9 && s.prevCycle >= 0.9) ctx.notify('Recesión: la demanda cae y el crédito se encarece.', 'warn');
  if (s.cycle > 1.08 && s.prevCycle <= 1.08) ctx.notify('Expansión económica: la demanda sube.', 'good');
  s.prevCycle = s.cycle;
  // ciudades: servicio y crecimiento
  let pop = 0;
  for (const c of s.world.cities) {
    const ratio = c.potential > 1 ? Math.min(1, c.served / c.potential) : 0;
    c.service = c.service * 0.6 + ratio * 0.4;
    const cov = ctx.cityCov.get(c.id) ?? 0;
    const g = cov > 0.05 ? BAL.cityGrowth * (c.service * cov - 0.25) : -BAL.cityDecline;
    c.pop = Math.max(1500, Math.round(c.pop * (1 + g)));
    c.served = 0; c.potential = 0; pop += c.pop;
  }
  // industrias: nivel según suministro
  for (const ind of s.world.industries) {
    if (ind.input !== null) {
      if (ind.supplied > ind.rate * 0.9 && ind.level < 3 && ctx.rng.chance(0.15)) { ind.level++; ctx.notify(`${ind.name} amplía su producción (nivel ${ind.level}).`, 'good'); }
      ind.supplied = ind.supplied * 0.3;   // memoria corta
    } else if (ind.stock > ind.rate * 2.5 && ind.level < 3 && s.stats.cargo > 0 && ctx.rng.chance(0.05)) ind.level++;
  }
  // contratos
  for (const c of s.contracts) {
    if (!c.active) continue;
    c.monthsLeft--;
    if (c.progress >= c.pax) { s.cash += c.reward; s.cur.subsidies += c.reward; s.dayProfit += c.reward; c.active = false; ctx.notify(`Contrato cumplido: +${Math.round(c.reward)} € de ${s.world.cities[c.cityId].name}.`, 'good'); }
    else if (c.monthsLeft <= 0) { c.active = false; s.reputation = Math.max(0, s.reputation - 3); ctx.notify(`Contrato fallido con ${s.world.cities[c.cityId].name}.`, 'warn'); }
  }
  s.contracts = s.contracts.filter(c => c.active || c.monthsLeft > -6);
  // reputación
  let onTime = 0, stops = 0, lost = 0, boarded = 0;
  for (const l of s.lines) { onTime += l.mOnTime; stops += l.mStops; }
  for (const n of s.nodes) { lost += n.lost; boarded += n.boarded; }
  const punct = stops > 0 ? onTime / stops : 0.7;
  const lostRatio = boarded + lost > 0 ? lost / (boarded + lost) : 0;
  const breakdownPen = s.monthBreakdowns * 0.3;
  const active = boarded + lost > 0 ? 1 : 0;   // sin servicio ni demanda, la reputación no cambia
  s.reputation = Math.max(0, Math.min(100, s.reputation + active * ((punct - 0.7) * 6 - lostRatio * 2 + 0.4) - breakdownPen));
  s.noBreakdownMonths = s.monthBreakdowns === 0 && s.vehicles.length >= 5 ? s.noBreakdownMonths + 1 : 0;
  s.monthBreakdowns = 0;
  // historial por línea
  const month = Math.floor(s.day / DAYS_PER_MONTH);
  for (const l of s.lines) {
    l.history.push({ month, revenue: l.mRevenue, cost: l.mCost, pax: l.mPax, cargo: l.mCargo, load: l.mLoadN ? l.mLoadSum / l.mLoadN : 0, punct: l.mStops ? l.mOnTime / l.mStops : 1, km: l.mKm });
    if (l.history.length > 240) l.history.shift();
    l.mRevenue = 0; l.mCost = 0; l.mPax = 0; l.mCargo = 0; l.mLoadSum = 0; l.mLoadN = 0; l.mOnTime = 0; l.mStops = 0; l.mKm = 0;
  }
  for (const n of s.nodes) { n.lost = 0; n.boarded = 0; }
  // libro mayor
  const cur = s.cur;
  cur.profit = cur.fares + cur.cargo + cur.subsidies + cur.fundGain + cur.offline - cur.running - cur.maintenance - cur.infra - cur.staff - cur.interest - cur.depreciation;
  cur.cash = s.cash; cur.value = companyValue(s); cur.pax = s.monthPax; cur.pop = pop;
  s.ledger.push(cur);
  if (s.ledger.length > 600) s.ledger.shift();
  s.monthPax = 0;
  s.cur = emptyMonth(monthOf(s), yearOf(s));
  s.stats.maxValue = Math.max(s.stats.maxValue, cur.value);
  if (s.cash < 0 && s.cash < -maxLoan(s) * 0.5) ctx.notify('Caja negativa: pide un préstamo o vende vehículos.', 'warn');
}

/** Oferta de contratos de ciudad (al azar, una vez al año). */
export function offerContract(ctx: Ctx): void {
  const s = ctx.s;
  const served = s.world.cities.filter(c => (ctx.cityCov.get(c.id) ?? 0) > 0.1);
  if (!served.length || s.contracts.some(c => c.active)) return;
  const c = ctx.rng.pick(served);
  const pax = Math.round(c.pop * 0.08 * ctx.rng.range(0.8, 1.4) / 100) * 100;
  const reward = Math.round(pax * eraOf(s.era).fare * 1.5 / 100) * 100;
  s.contracts.push({ id: s.nextId++, cityId: c.id, pax, monthsLeft: BAL.contractMonths, reward, progress: 0, active: true });
  ctx.notify(`${c.name} ofrece una subvención: ${pax} pax en 12 meses por ${reward} €.`, 'info');
}
