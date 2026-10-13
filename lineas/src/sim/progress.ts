import type { GameState, Daily } from './types';
import { BAL, SEC_PER_DAY } from './constants';
import { hashString, Rng } from './rng';
import { companyValue } from './economy';
import { DATA, tech } from './data';
import type { Ctx } from './game';

export function todayKey(now = Date.now()): string { const d = new Date(now); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`; }

const DAILY_TYPES = [
  { id: 'pax', name: 'Transporta {n} pasajeros', base: 6 },
  { id: 'earn', name: 'Ingresa {n} €', base: 4000 },
  { id: 'build', name: 'Construye {n} elementos (estaciones o vías)', base: 2 },
  { id: 'cargo', name: 'Entrega {n} t de mercancías', base: 60 },
  { id: 'km', name: 'Recorre {n} km con la flota', base: 20 },
];
export function dailyName(d: Daily): string { const t = DAILY_TYPES.find(x => x.id === d.id)!; return t.name.replace('{n}', Math.round(d.target).toLocaleString('es-ES')); }

/** Regenera objetivos diarios si cambió la fecha real. Devuelve true si es un día nuevo. */
export function refreshDailies(s: GameState, now = Date.now()): boolean {
  const key = todayKey(now);
  if (s.dailyDate === key) return false;
  // racha: se mantiene si el último día jugado fue ayer y se completaron los 3
  const yesterday = todayKey(now - 86400000);
  const allDone = s.dailies.length === 3 && s.dailies.every(d => d.done);
  if (s.dailyDate === yesterday && allDone) s.streak++;
  else if (s.dailyDate !== key && s.dailyDate !== '') s.streak = allDone && s.dailyDate === yesterday ? s.streak : 0;
  const rng = new Rng(hashString('daily-' + key + '-' + s.seed));
  const scale = Math.max(1, Math.pow(Math.max(1, companyValue(s)) / 200000, 0.6));
  const types = [...DAILY_TYPES];
  const picked: Daily[] = [];
  const hasCargo = s.lines.some(l => l.cargoKind);
  while (picked.length < 3 && types.length) {
    const i = rng.int(0, types.length - 1); const t = types.splice(i, 1)[0];
    if (t.id === 'cargo' && !hasCargo) continue;
    const target = t.id === 'build' ? t.base : Math.round(t.base * scale * rng.range(0.8, 1.5) / 10) * 10;
    picked.push({ id: t.id, target, progress: 0, done: false, reward: Math.round(Math.max(2000, companyValue(s) * 0.01) * (1 + s.streak * 0.1) / 100) * 100 });
  }
  s.dailies = picked; s.dailyDate = key;
  return true;
}
export function dailyProgress(s: GameState, kind: string, n: number): string[] {
  const out: string[] = [];
  for (const d of s.dailies) {
    if (d.done || d.id !== kind) continue;
    d.progress += n;
    if (d.progress >= d.target) { d.done = true; s.cash += d.reward; s.cur.subsidies += d.reward; out.push(`Objetivo diario cumplido: +${d.reward} €`); }
  }
  return out;
}

/** Progreso offline: aplica el tiempo real transcurrido desde la última sesión. */
export function applyOffline(ctx: Ctx, now = Date.now()): void {
  const s = ctx.s;
  if (!s.lastRealTime) { s.lastRealTime = now; return; }
  const elapsed = (now - s.lastRealTime) / 1000;
  s.lastRealTime = now;
  if (elapsed < 120) return;
  const capped = Math.min(elapsed, BAL.offlineHoursMax * 3600);
  const days = capped / SEC_PER_DAY;
  const avg = s.profitWindow.length ? s.profitWindow.reduce((a, b) => a + b, 0) / s.profitWindow.length : 0;
  const eff = BAL.offlineEff + 0.05 * (s.prestige.spent.offline ?? 0);
  const profit = avg * days * eff;
  s.cash += profit; s.cur.offline += profit;
  const before = s.day; s.day += days;
  const completed: string[] = [];
  for (const n of s.nodes) if (n.doneDay > before && n.doneDay <= s.day) completed.push(n.name);
  for (const t of s.tracks) if (t.doneDay > before && t.doneDay <= s.day) completed.push('Vía ' + t.id);
  if (s.research && s.research.doneDay <= s.day) { completed.push(tech(s.research.id)?.name ?? s.research.id); }
  for (const v of s.vehicles) if (v.state === 'delivery') { v.timer -= days; if (v.timer <= 0) { v.timer = 0; v.state = v.lineId === null ? 'idle' : 'moving'; completed.push(DATA.vehicles.find(m => m.id === v.model)?.name ?? v.model); } }
  s.offline = { days, profit, completed };
  ctx.afterOffline();
}

/** Prestigio: refundar. */
export function canRefound(s: GameState): boolean { return companyValue(s) >= BAL.prestigeThreshold; }
export function refoundPoints(s: GameState): number { const v = companyValue(s); return v < BAL.prestigeThreshold ? 0 : Math.floor(Math.log2(v / BAL.prestigeBase)); }
export const PRESTIGE_TREE = [
  { id: 'revenue', name: 'Tarifas premium', desc: '+4 % de ingresos por punto', max: 5 },
  { id: 'cost', name: 'Compras centralizadas', desc: '−3 % de costes por punto', max: 5 },
  { id: 'build', name: 'Obra civil veterana', desc: '−8 % de plazo de obra por punto', max: 5 },
  { id: 'cash', name: 'Capital semilla', desc: '+25 % de caja inicial por punto', max: 5 },
  { id: 'offline', name: 'Dirección delegada', desc: '+5 % de eficiencia offline por punto', max: 5 },
  { id: 'research', name: 'Laboratorio propio', desc: '−8 % de tiempo de investigación por punto', max: 5 },
  { id: 'demand', name: 'Marca reconocida', desc: '+2 % de demanda por punto', max: 5 },
];
export function prestigeSpend(s: GameState, id: string): boolean {
  const node = PRESTIGE_TREE.find(p => p.id === id); if (!node) return false;
  const spent = Object.values(s.prestige.spent).reduce((a, b) => a + b, 0);
  if (spent >= s.prestige.points) return false;
  if ((s.prestige.spent[id] ?? 0) >= node.max) return false;
  s.prestige.spent[id] = (s.prestige.spent[id] ?? 0) + 1; return true;
}
export function prestigeAvailable(s: GameState): number { return s.prestige.points - Object.values(s.prestige.spent).reduce((a, b) => a + b, 0); }
