import type { GameState } from './types';
import { DATA } from './data';
import { companyValue, fundValue } from './economy';
import { hashString } from './rng';
import { model } from './data';

/** Métricas para licencias (se evalúan cada mes). */
export function licenseMetrics(s: GameState): Record<string, number> {
  const m: Record<string, number> = {};
  let bestLine = 0, bestLoad3 = 0, punctSum = 0, punctN = 0, monthCargo = 0, hsrPax = 0, total = 0;
  for (const l of s.lines) {
    const h = l.history[l.history.length - 1]; if (!h) continue;
    bestLine = Math.max(bestLine, h.revenue - h.cost); total += h.revenue - h.cost;
    if (l.vehicles.length >= 3) bestLoad3 = Math.max(bestLoad3, h.load);
    punctSum += h.punct; punctN++; monthCargo += h.cargo;
    if (l.kind === 'hsr') hsrPax += h.pax;
  }
  m.bestLineMonthProfit = bestLine;
  m.monthProfit5 = s.nodes.filter(n => n.station).length >= 5 ? total : 0;
  m.bestLoad3 = bestLoad3;
  m.punctuality = punctN && s.vehicles.length >= 3 ? punctSum / punctN : 0;
  m.monthCargo = monthCargo;
  m.maxPop = Math.max(0, ...s.world.cities.map(c => c.pop));
  m.fundGain = fundValue(s) - s.fund.invested;
  m.models = Object.keys(s.collection.owned).length;
  m.noBreakdownMonths = s.noBreakdownMonths;
  m.value = companyValue(s);
  m.monthHsrPax = hsrPax;
  m.km = s.stats.km;
  return m;
}

export function updateMedals(s: GameState): string[] {
  const metrics = licenseMetrics(s); const news: string[] = [];
  for (const lic of DATA.licenses) {
    const v = metrics[lic.metric] ?? 0;
    let med = s.medals.find(x => x.id === lic.id);
    if (!med) { med = { id: lic.id, best: 0, tier: 0 }; s.medals.push(med); }
    if (v > med.best) med.best = v;
    let tier: 0 | 1 | 2 | 3 = 0;
    for (let i = 0; i < 3; i++) if (med.best >= lic.tiers[i]) tier = (i + 1) as 1 | 2 | 3;
    if (tier > med.tier) { med.tier = tier; news.push(`${lic.name}: medalla de ${['', 'bronce', 'plata', 'oro'][tier]}`); }
  }
  return news;
}

/** Reto semanal: semilla derivada de la semana ISO. */
export function isoWeek(d = new Date()): string {
  const date = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
  const dayNum = date.getUTCDay() || 7;
  date.setUTCDate(date.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(date.getUTCFullYear(), 0, 1));
  const week = Math.ceil(((date.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
  return `${date.getUTCFullYear()}W${String(week).padStart(2, '0')}`;
}
export const WEEKLY_YEARS = 15;
export function weeklySeed(week: string): number { return hashString('lineas-weekly-' + week); }
export function weeklyDone(s: GameState): boolean { return s.weekly && s.day >= WEEKLY_YEARS * 360; }
export function weeklyCode(s: GameState): string {
  const v = Math.round(companyValue(s));
  const week = s.weeklyWeek ?? isoWeek();
  const chk = (hashString(`${week}|${v}|${s.seed}`) % 46656).toString(36).padStart(3, '0').toUpperCase();
  return `LN-${week}-${v}-${chk}`;
}
export function verifyWeeklyCode(code: string): { week: string; value: number; ok: boolean } | null {
  const m = /^LN-(\d{4}W\d{2})-(\d+)-([0-9A-Z]{3})$/.exec(code.trim()); if (!m) return null;
  const seed = weeklySeed(m[1]);
  const chk = (hashString(`${m[1]}|${m[2]}|${seed}`) % 46656).toString(36).padStart(3, '0').toUpperCase();
  return { week: m[1], value: Number(m[2]), ok: chk === m[3] };
}

/** Variantes de librea desbloqueadas por km del modelo. */
export const VARIANT_KM = [0, 100, 500, 2000];
export function variantsUnlocked(s: GameState, modelId: string): number {
  const c = s.collection.owned[modelId]; if (!c) return 0;
  let n = 0; for (const k of VARIANT_KM) if (c.km >= k) n++;
  return n;
}
export function collectionSummary(s: GameState) {
  const total = DATA.vehicles.length;
  const owned = Object.keys(s.collection.owned).filter(id => model(id)).length;
  return { total, owned };
}
