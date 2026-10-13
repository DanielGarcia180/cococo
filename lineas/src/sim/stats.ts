import type { GameState, Line } from './types';
import { Rng } from './rng';
import { companyValue } from './economy';

export interface LineKpi {
  revenue: number; cost: number; profit: number; pax: number; cargo: number; load: number; punct: number;
  roi: number; payback: number | null; paxPerHour: number; months: number; invested: number; vehicles: number;
}

export function lineKpi(s: GameState, l: Line): LineKpi {
  const h = l.history.slice(-12);
  const months = h.length;
  const sum = (k: keyof typeof h[number]) => h.reduce((a, b) => a + (b[k] as number), 0);
  const revenue = sum('revenue'), cost = sum('cost'), pax = sum('pax'), cargo = sum('cargo');
  const load = months ? sum('load') / months : 0;
  const punct = months ? sum('punct') / months : 1;
  const invested = l.invested;
  const profit = revenue - cost;
  const monthly = months ? profit / months : (l.mRevenue - l.mCost);
  const roi = invested > 0 ? (l.revenue - l.cost) / invested : 0;
  const payback = invested > 0 && monthly > 0 ? Math.max(0, (invested - (l.revenue - l.cost)) / monthly) : null;
  return { revenue, cost, profit, pax, cargo, load, punct, roi, payback, paxPerHour: months ? pax / months / 720 : (l.mPax / 720), months, invested, vehicles: l.vehicles.length };
}

export interface Forecast { p10: number[]; p50: number[]; p90: number[]; mean: number; sigma: number; months: number; cumulative: boolean }

/** Previsión Monte Carlo del beneficio mensual acumulado de una línea (N trayectorias). */
export function monteCarlo(l: Line, horizon = 24, n = 500, seed = 12345): Forecast {
  const h = l.history.slice(-24).map(m => m.revenue - m.cost);
  let mean: number, sigma: number;
  if (h.length >= 3) {
    mean = h.reduce((a, b) => a + b, 0) / h.length;
    const v = h.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, h.length - 1);
    sigma = Math.sqrt(v);
  } else {
    mean = h.length ? h.reduce((a, b) => a + b, 0) / h.length : l.mRevenue - l.mCost;
    sigma = Math.abs(mean) * 0.35 + 1;
  }
  const rng = new Rng(seed ^ l.id);
  const paths: number[][] = [];
  for (let i = 0; i < n; i++) {
    const p: number[] = []; let acc = 0;
    // deriva ligera (crecimiento urbano) + ruido AR(1)
    let noise = 0;
    for (let m = 0; m < horizon; m++) {
      noise = noise * 0.4 + rng.gauss() * sigma;
      acc += mean * (1 + 0.004 * m) + noise;
      p.push(acc);
    }
    paths.push(p);
  }
  const q = (m: number, pct: number) => { const col = paths.map(p => p[m]).sort((a, b) => a - b); return col[Math.min(n - 1, Math.floor(pct * n))]; };
  const p10: number[] = [], p50: number[] = [], p90: number[] = [];
  for (let m = 0; m < horizon; m++) { p10.push(q(m, 0.1)); p50.push(q(m, 0.5)); p90.push(q(m, 0.9)); }
  return { p10, p50, p90, mean, sigma, months: h.length, cumulative: true };
}

export function globalKpis(s: GameState) {
  const last = s.ledger.slice(-12);
  const sum = (k: 'profit' | 'fares' | 'cargo' | 'pax') => last.reduce((a, b) => a + b[k], 0);
  let onTime = 0, stops = 0;
  for (const l of s.lines) { const h = l.history.slice(-1)[0]; if (h) { onTime += h.punct; stops++; } }
  return {
    value: companyValue(s), profit12: sum('profit'), fares12: sum('fares'), cargo12: sum('cargo'), pax12: sum('pax'),
    punct: stops ? onTime / stops : 1, pop: s.world.cities.reduce((a, c) => a + c.pop, 0),
    vehicles: s.vehicles.length, lines: s.lines.length, stations: s.nodes.filter(n => n.station).length,
    km: s.stats.km, reputation: s.reputation,
  };
}

export function ledgerCsv(s: GameState): string {
  const head = ['año', 'mes', 'billetes', 'carga', 'subvenciones', 'fondo', 'offline', 'combustible', 'mantenimiento', 'infraestructura', 'personal', 'intereses', 'depreciacion', 'beneficio', 'caja', 'valor', 'pasajeros', 'poblacion'];
  const rows = s.ledger.map(m => [m.year, m.month + 1, m.fares, m.cargo, m.subsidies, m.fundGain, m.offline, m.running, m.maintenance, m.infra, m.staff, m.interest, m.depreciation, m.profit, m.cash, m.value, m.pax, m.pop].map(v => typeof v === 'number' ? Math.round(v * 100) / 100 : v).join(';'));
  return [head.join(';'), ...rows].join('\n');
}
export function linesCsv(s: GameState): string {
  const head = ['linea', 'mes', 'ingresos', 'costes', 'beneficio', 'pasajeros', 'carga_t', 'factor_carga', 'puntualidad', 'km'];
  const rows: string[] = [];
  for (const l of s.lines) for (const h of l.history) rows.push([l.name, h.month, h.revenue, h.cost, h.revenue - h.cost, h.pax, h.cargo, h.load, h.punct, h.km].map(v => typeof v === 'number' ? Math.round(v * 100) / 100 : `"${v}"`).join(';'));
  return [head.join(';'), ...rows].join('\n');
}
