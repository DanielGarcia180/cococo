/** Gráficos minimalistas sobre canvas. Un único acento gris, sin degradados. */
const INK = '#111', MUTED = '#8a8a8a', GRID = '#e6e6e6';

function setup(c: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  c.width = w * dpr; c.height = h * dpr; c.style.width = w + 'px'; c.style.height = h + 'px';
  const ctx = c.getContext('2d')!; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
  ctx.font = '11px system-ui, -apple-system, sans-serif'; return ctx;
}
function fmt(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(1) + 'M'; if (a >= 1e3) return (v / 1e3).toFixed(0) + 'k'; return v.toFixed(0);
}

export interface Series { name: string; values: number[]; color?: string; dashed?: boolean }

export function lineChart(c: HTMLCanvasElement, series: Series[], opts: { w?: number; h?: number; labels?: string[]; zero?: boolean; unit?: string; band?: { lo: number[]; hi: number[] } } = {}): void {
  const w = opts.w ?? (c.clientWidth || 360), hh = opts.h ?? 160;
  const ctx = setup(c, w, hh);
  const padL = 44, padR = 8, padT = 10, padB = 20;
  const all = series.flatMap(s => s.values).concat(opts.band ? [...opts.band.lo, ...opts.band.hi] : []);
  if (!all.length) { ctx.fillStyle = MUTED; ctx.fillText('Sin datos todavía', padL, hh / 2); return; }
  let mn = Math.min(...all), mx = Math.max(...all);
  if (opts.zero !== false) { mn = Math.min(0, mn); mx = Math.max(0, mx); }
  if (mx === mn) { mx += 1; mn -= 1; }
  const n = Math.max(2, ...series.map(s => s.values.length), opts.band?.lo.length ?? 0);
  const X = (i: number) => padL + (w - padL - padR) * i / (n - 1);
  const Y = (v: number) => padT + (hh - padT - padB) * (1 - (v - mn) / (mx - mn));
  // rejilla
  ctx.strokeStyle = GRID; ctx.lineWidth = 1; ctx.fillStyle = MUTED; ctx.textAlign = 'right';
  for (let k = 0; k <= 4; k++) { const v = mn + (mx - mn) * k / 4; const y = Y(v); ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); ctx.fillText(fmt(v), padL - 6, y + 4); }
  if (mn < 0 && mx > 0) { ctx.strokeStyle = '#bbb'; ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(w - padR, Y(0)); ctx.stroke(); }
  if (opts.labels) { ctx.textAlign = 'center'; const step = Math.ceil(opts.labels.length / 6); for (let i = 0; i < opts.labels.length; i += step) ctx.fillText(opts.labels[i], X(i), hh - 5); }
  if (opts.band) {
    ctx.fillStyle = 'rgba(0,0,0,0.08)'; ctx.beginPath();
    opts.band.lo.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)));
    for (let i = opts.band.hi.length - 1; i >= 0; i--) ctx.lineTo(X(i), Y(opts.band.hi[i]));
    ctx.closePath(); ctx.fill();
  }
  for (const s of series) {
    ctx.strokeStyle = s.color ?? INK; ctx.lineWidth = 1.6; ctx.setLineDash(s.dashed ? [4, 3] : []);
    ctx.beginPath(); s.values.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))); ctx.stroke(); ctx.setLineDash([]);
  }
  if (series.length > 1) { ctx.textAlign = 'left'; let x = padL; for (const s of series) { ctx.fillStyle = s.color ?? INK; ctx.fillRect(x, 2, 10, 2); ctx.fillStyle = MUTED; ctx.fillText(s.name, x + 14, 7); x += 14 + ctx.measureText(s.name).width + 14; } }
}

export function barChart(c: HTMLCanvasElement, values: number[], opts: { w?: number; h?: number; labels?: string[]; colorNeg?: string } = {}): void {
  const w = opts.w ?? (c.clientWidth || 360), hh = opts.h ?? 140;
  const ctx = setup(c, w, hh);
  const padL = 44, padR = 8, padT = 8, padB = 20;
  if (!values.length) { ctx.fillStyle = MUTED; ctx.fillText('Sin datos todavía', padL, hh / 2); return; }
  const mn = Math.min(0, ...values), mx = Math.max(0, ...values, 1);
  const Y = (v: number) => padT + (hh - padT - padB) * (1 - (v - mn) / (mx - mn));
  const bw = (w - padL - padR) / values.length;
  ctx.strokeStyle = GRID; ctx.fillStyle = MUTED; ctx.textAlign = 'right';
  for (let k = 0; k <= 3; k++) { const v = mn + (mx - mn) * k / 3; ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(w - padR, Y(v)); ctx.stroke(); ctx.fillText(fmt(v), padL - 6, Y(v) + 4); }
  values.forEach((v, i) => { ctx.fillStyle = v < 0 ? (opts.colorNeg ?? '#9a9a9a') : INK; const y0 = Y(0), y1 = Y(v); ctx.fillRect(padL + i * bw + bw * 0.15, Math.min(y0, y1), bw * 0.7, Math.max(1, Math.abs(y1 - y0))); });
  if (opts.labels) { ctx.textAlign = 'center'; ctx.fillStyle = MUTED; const step = Math.ceil(opts.labels.length / 6); for (let i = 0; i < opts.labels.length; i += step) ctx.fillText(opts.labels[i], padL + i * bw + bw / 2, hh - 5); }
}

export function sparkline(c: HTMLCanvasElement, values: number[], w = 90, hh = 24): void {
  const ctx = setup(c, w, hh);
  if (values.length < 2) return;
  const mn = Math.min(...values), mx = Math.max(...values);
  ctx.strokeStyle = INK; ctx.lineWidth = 1.2; ctx.beginPath();
  values.forEach((v, i) => { const x = i / (values.length - 1) * (w - 2) + 1; const y = hh - 2 - (mx === mn ? hh / 2 : (v - mn) / (mx - mn) * (hh - 4)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
}

/** Barra horizontal simple 0..1 (para factores de carga, progreso). */
export function meter(value: number, label?: string): HTMLElement {
  const el = document.createElement('div'); el.className = 'meter';
  const fill = document.createElement('div'); fill.className = 'meter-fill'; fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
  el.appendChild(fill);
  if (label) { const l = document.createElement('span'); l.className = 'meter-label'; l.textContent = label; el.appendChild(l); }
  return el;
}
