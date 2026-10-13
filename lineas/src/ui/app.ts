import type { Game } from '../sim/game';
import type { Renderer3D } from '../render/scene';
import type { TrackKind, Node, Track, Line, Vehicle, Vec2 } from '../sim/types';
import { h, clear, money, num, km, btn, row, section, pct, signed } from './dom';
import { t, monthName } from './i18n';
import { TRACK_SPEC } from '../sim/pathfind';
import { model, DATA, eraOf } from '../sim/data';
import { catchment } from '../sim/demand';
import { lineKpi, monteCarlo } from '../sim/stats';
import { lineChart, meter } from './charts';
import { effSpeedKmh, bookValue, capacityOf } from '../sim/vehicles';
import { PAX_UNIT } from '../sim/constants';
import { variantsUnlocked } from '../sim/challenges';
import { LIVERIES } from '../render/scene';
import { dailyName } from '../sim/progress';
import { renderSheet, type SheetId } from './panels';
import { LINE_COLORS } from '../sim/network';

export type Tool = 'select' | 'station' | 'track' | 'line' | 'waypoint' | 'bulldoze';
type Sel = { type: 'node'; id: number } | { type: 'track'; id: number } | { type: 'line'; id: number } | { type: 'vehicle'; id: number } | null;

const ICONS: Record<string, string> = {
  select: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 3l14 8-6 2-2 6z"/></svg>',
  station: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="9" width="16" height="10" rx="1"/><path d="M4 9l8-5 8 5M9 19v-4h6v4"/></svg>',
  track: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M7 20L17 4M11 20L21 4M6 9h9M4 14h9M3 19h8"/></svg>',
  line: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="5" cy="18" r="2"/><circle cx="12" cy="8" r="2"/><circle cx="19" cy="15" r="2"/><path d="M6.5 16.5l4-7M13.5 9.5l4 4"/></svg>',
  waypoint: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><path d="M12 3v4M12 17v4M3 12h4M17 12h4"/></svg>',
  bulldoze: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 20h16M6 20V9l6-5 6 5v11M9 20v-6h6v6"/><path d="M3 4l18 16" stroke-width="2"/></svg>',
  heat: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="10.5"/></svg>',
  undo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M9 14L4 9l5-5M4 9h10a6 6 0 010 12h-3"/></svg>',
  redo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M15 14l5-5-5-5M20 9H10a6 6 0 000 12h3"/></svg>',
};

export class App {
  game: Game; r: Renderer3D;
  tool: Tool = 'select'; kind: TrackKind = 'rail'; trackFrom: number | null = null; lineEditing: number | null = null; cargoLine = false;
  selected: Sel = null; sheet: SheetId | null = null; hoverNode: Node | null = null;
  els = {
    top: h('div', { id: 'top' }), tools: h('div', { id: 'tools' }), kinds: h('div', { id: 'kinds' }), side: h('div', { id: 'side' }),
    toasts: h('div', { id: 'toasts' }), tutorial: h('div', { id: 'tutorial', class: 'hidden' }), tip: h('div', { id: 'cursor-tip', class: 'hidden' }),
    sheet: h('div', { id: 'sheet', class: 'hidden' }), dailies: h('div', { id: 'dailies' }), photo: h('div', { id: 'photo', class: 'hidden' }),
  };
  private hudTimer = 0; private sideTimer = 0; private sheetTimer = 0; private lastHover: Vec2 | null = null;
  onMenu: () => void = () => {};
  onSave: () => void = () => {};

  constructor(game: Game, r: Renderer3D) {
    this.game = game; this.r = r;
    for (const el of Object.values(this.els)) document.body.appendChild(el);
    this.els.side.addEventListener('focusout', () => setTimeout(() => this.renderSide(), 50));
    this.buildTop(); this.buildTools();
    this.bindCanvas(); this.bindKeys();
    game.listeners.push(e => this.onEvent(e));
    this.refreshAll();
  }

  // ───────────── eventos del juego ─────────────
  onEvent(e: { type: string; text?: string; kind?: string }): void {
    if (e.type === 'notify') this.toast(e.text!, e.kind as 'info' | 'warn' | 'good');
    if (e.type === 'build' || e.type === 'network') { this.r.rebuildTracks(); this.r.rebuildStations(); if (this.r.heatOn) this.r.updateHeat(); this.renderSide(); }
    if (e.type === 'lines') { this.r.rebuildLines(); this.renderSide(); }
    if (e.type === 'vehicles' || e.type === 'research' || e.type === 'fund' || e.type === 'speed') { this.renderSide(); this.renderHud(); if (this.sheet) this.renderSheet(); }
    if (e.type === 'month') { this.r.checkCities(); if (this.sheet) this.renderSheet(); }
    if (e.type === 'day') this.renderDailies();
    if (e.type === 'weeklyDone') this.openSheet('challenges');
    if (e.type === 'dailies') this.renderDailies();
  }
  refreshAll(): void { this.renderHud(); this.renderTools(); this.renderSide(); this.renderTutorial(); this.renderDailies(); if (this.sheet) this.renderSheet(); }

  /** Llamar cada frame. */
  update(dt: number): void {
    this.hudTimer += dt; this.sideTimer += dt; this.sheetTimer += dt;
    if (this.hudTimer > 0.25) { this.hudTimer = 0; this.renderHud(); this.renderTutorial(); }
    if (this.sideTimer > 0.6) { this.sideTimer = 0; this.renderSideLive(); }
    if (this.sheet && this.sheetTimer > 2) { this.sheetTimer = 0; this.renderSheet(); }
  }

  toast(text: string, kind: 'info' | 'warn' | 'good' = 'info'): void {
    const el = h('div', { class: 'toast ' + kind }, text); this.els.toasts.appendChild(el);
    while (this.els.toasts.children.length > 5) this.els.toasts.removeChild(this.els.toasts.firstChild!);
    setTimeout(() => el.remove(), 8000);
  }

  // ───────────── HUD superior ─────────────
  private topRefs: Record<string, HTMLElement> = {};
  private buildTop(): void {
    const stat = (id: string, label: string) => { const v = h('span', { class: 'v mono' }, '—'); this.topRefs[id] = v; return h('div', { class: 'stat' }, h('span', { class: 'k' }, label), v); };
    const speedBtn = (sp: number, label: string) => { const b = h('button', { onClick: () => sp === 0 ? this.game.togglePause() : this.game.setSpeed(sp) }, label); this.topRefs['sp' + sp] = b; return b; };
    const nav = (id: SheetId, label: string) => { const b = h('button', { onClick: () => this.toggleSheet(id) }, label); this.topRefs['nav.' + id] = b; return b; };
    this.els.top.append(
      h('span', { class: 'brand' }, t('app.title')),
      stat('date', 'Fecha'), stat('cash', t('hud.cash')), stat('value', t('hud.value')), stat('rep', t('hud.rep')), stat('era', 'Era'), stat('weather', 'Clima'),
      h('span', { class: 'spacer' }),
      h('div', { class: 'speed' }, speedBtn(0, '⏸'), speedBtn(1, '1×'), speedBtn(2, '2×'), speedBtn(4, '4×'), speedBtn(8, '8×')),
      h('nav', null, nav('finance', t('menu.finance')), nav('data', t('menu.data')), nav('research', t('menu.research')), nav('vehicles', t('menu.vehicles')), nav('fund', t('menu.fund')), nav('challenges', t('menu.challenges')), nav('prestige', t('menu.prestige')), nav('settings', t('menu.settings'))),
      h('button', { class: 'btn sm', onClick: () => this.togglePhoto(), title: 'Modo foto (P)' }, '📷'),
      h('button', { class: 'btn sm', onClick: () => this.onMenu() }, t('menu.main')),
    );
  }
  renderHud(): void {
    const s = this.game.s; const r = this.topRefs;
    const day = Math.floor(s.day % 30) + 1;
    r.date.textContent = `${day} ${monthName(this.game.month())} ${this.game.year()}`;
    r.cash.textContent = money(s.cash); r.cash.className = 'v mono' + (s.cash < 0 ? ' neg' : '');
    r.value.textContent = money(this.game.value());
    r.rep.textContent = `${Math.round(s.reputation)}/100`;
    r.era.textContent = eraOf(s.era).name;
    const hr = this.r.clock; r.weather.textContent = `${t('weather.' + s.weather)} · ${String(Math.floor(hr)).padStart(2, '0')}:${String(Math.floor((hr % 1) * 60)).padStart(2, '0')}`;
    for (const sp of [0, 1, 2, 4, 8]) r['sp' + sp].classList.toggle('on', sp === 0 ? s.paused : (!s.paused && s.speed === sp));
    for (const k in r) if (k.startsWith('nav.')) r[k].classList.toggle('on', this.sheet === k.slice(4));
  }

  // ───────────── herramientas ─────────────
  private toolBtns: Record<string, HTMLButtonElement> = {};
  private buildTools(): void {
    const mk = (id: Tool | 'heat' | 'undo' | 'redo', key: string, label: string, onClick: () => void) => {
      const b = h('button', { html: ICONS[id] + `<span class="key">${key}</span>`, title: `${label} (${key})`, onClick }); this.toolBtns[id] = b; return b;
    };
    this.els.tools.append(
      mk('select', 'Q', t('tool.select'), () => this.setTool('select')),
      mk('station', '2', t('tool.station'), () => this.setTool('station')),
      mk('track', '3', t('tool.track'), () => this.setTool('track')),
      mk('line', '4', t('tool.line'), () => this.setTool('line')),
      mk('waypoint', '5', t('tool.waypoint'), () => this.setTool('waypoint')),
      mk('bulldoze', 'X', t('tool.bulldoze'), () => this.setTool('bulldoze')),
      h('hr'),
      mk('heat', 'H', t('tool.heat'), () => { this.r.toggleHeat(); this.renderTools(); }),
      mk('undo', 'Z', t('tool.undo'), () => { this.game.undo(); }),
      mk('redo', 'Y', t('tool.redo'), () => { this.game.redo(); }),
    );
  }
  setTool(tool: Tool): void {
    this.tool = tool; this.trackFrom = null; this.r.setPreview(null, this.kind);
    if (tool !== 'line') this.lineEditing = null;
    if (tool === 'line' && this.lineEditing === null) { this.selected = null; }
    this.renderTools(); this.renderSide(); this.hideTip();
  }
  renderTools(): void {
    for (const k in this.toolBtns) this.toolBtns[k].classList.toggle('on', k === this.tool || (k === 'heat' && this.r.heatOn));
    const showKinds = this.tool === 'station' || this.tool === 'track' || this.tool === 'waypoint' || (this.tool === 'line' && this.lineEditing === null);
    this.els.kinds.classList.toggle('hidden', !showKinds);
    clear(this.els.kinds);
    if (!showKinds) return;
    const s = this.game.s;
    for (const k of ['road', 'rail', 'metro', 'hsr'] as TrackKind[]) {
      const locked = (k === 'metro' && !s.researched.includes('metro')) || (k === 'hsr' && !s.researched.includes('hsr'));
      this.els.kinds.append(h('button', { class: k === this.kind ? 'on' : '', disabled: locked, title: locked ? 'Requiere investigación' : `${t('kind.' + k)} — ${TRACK_SPEC[k].base} €/m`, onClick: () => { this.kind = k; this.trackFrom = null; this.r.setPreview(null, k); this.renderTools(); } }, t('kind.' + k)));
    }
    if (this.tool === 'line') {
      this.els.kinds.append(h('label', { class: 'row small', style: { marginLeft: '8px' } }, h('input', { type: 'checkbox', checked: this.cargoLine, onChange: (e: Event) => { this.cargoLine = (e.target as HTMLInputElement).checked; } }), 'Mercancías'));
      this.els.kinds.append(btn('Nueva línea', () => this.newLine(), 'primary sm'));
    }
    if (this.tool === 'station') this.els.kinds.append(h('span', { class: 'small muted', style: { padding: '6px 8px' } }, `${money(this.game.stationCost(this.kind, true))} · ${this.game.stationDays(this.kind, true)} ${t('common.days')} · radio ${km(catchment(s, { kind: this.kind, level: 1 } as Node))}`));
  }
  newLine(): void {
    const l = this.game.createLine(this.kind, this.cargoLine ? 'ore' : null);
    this.lineEditing = l.id; this.selected = { type: 'line', id: l.id }; this.renderTools(); this.renderSide();
    this.toast('Haz clic en las estaciones en orden para añadir paradas.', 'info');
  }

  // ───────────── entrada ─────────────
  private bindCanvas(): void {
    const c = this.r.renderer.domElement; let downPos: [number, number] | null = null;
    c.addEventListener('pointerdown', e => { downPos = [e.clientX, e.clientY]; });
    c.addEventListener('pointerup', e => {
      if (!downPos) return; const moved = Math.hypot(e.clientX - downPos[0], e.clientY - downPos[1]); downPos = null;
      if (moved > 5) return;
      if (e.button === 2) { this.cancel(); return; }
      if (e.button === 0) this.click(e);
    });
    c.addEventListener('contextmenu', e => e.preventDefault());
    c.addEventListener('pointermove', e => this.move(e));
    c.addEventListener('pointerleave', () => { this.hideTip(); this.r.setHover(null); });
  }
  private ndc(e: PointerEvent): [number, number] { return [(e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1]; }
  private nodeAt(p: Vec2, maxDist = 450): Node | null {
    let best: Node | null = null, bd = maxDist;
    for (const n of this.game.s.nodes) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n; } }
    return best;
  }
  private trackAt(p: Vec2, maxDist = 180): Track | null {
    let best: Track | null = null, bd = maxDist;
    for (const t of this.game.s.tracks) for (let i = 1; i < t.path.length; i++) {
      const a = t.path[i - 1], b = t.path[i]; const d = distSeg(p, a, b); if (d < bd) { bd = d; best = t; }
    }
    return best;
  }
  cancel(): void {
    if (this.trackFrom !== null) { this.trackFrom = null; this.r.setPreview(null, this.kind); return; }
    if (this.lineEditing !== null) { this.lineEditing = null; this.setTool('select'); return; }
    if (this.tool !== 'select') { this.setTool('select'); return; }
    this.select(null);
  }
  private move(e: PointerEvent): void {
    if (this.r.photo.active) return;
    const [nx, ny] = this.ndc(e); const p = this.r.pickGround(nx, ny); this.lastHover = p;
    if (!p) { this.hideTip(); this.r.setHover(null); return; }
    const s = this.game.s;
    if (this.tool === 'station' || this.tool === 'waypoint') {
      const err = this.game.canPlaceNode(p.x, p.y, this.kind);
      this.r.setHover(p);
      const station = this.tool === 'station';
      this.showTip(e, err ?? `${station ? t('tool.station') : t('tool.waypoint')} · ${money(this.game.stationCost(this.kind, station))}\n${this.game.stationDays(this.kind, station)} ${t('common.days')} de obra`);
    } else if (this.tool === 'track') {
      const n = this.nodeAt(p); this.hoverNode = n; this.r.setHover(n ? n : null);
      if (this.trackFrom !== null && n && n.id !== this.trackFrom) {
        const pv = this.game.previewTrack(this.trackFrom, n.id, this.kind);
        if (pv) { this.r.setPreview(pv.path, this.kind, !pv.error); this.showTip(e, `${t('kind.' + this.kind)} · ${km(pv.length)}\n${money(pv.cost)} · ${pv.days} ${t('common.days')}${pv.tunnel > 50 ? `\nTúnel ${km(pv.tunnel)}` : ''}${pv.bridge > 50 ? `\nPuente ${km(pv.bridge)}` : ''}${pv.error ? '\n' + pv.error : ''}`); }
        else { this.r.setPreview(null, this.kind); this.showTip(e, 'Ya existe una vía entre estos nodos'); }
      } else if (this.trackFrom !== null) { this.r.setPreview(null, this.kind); this.showTip(e, 'Elige el nodo de destino'); }
      else this.showTip(e, n ? `Empezar vía en ${n.name}` : 'Haz clic en un nodo de origen');
    } else if (this.tool === 'line' && this.lineEditing !== null) {
      const n = this.nodeAt(p); this.r.setHover(n ? n : null);
      this.showTip(e, n ? `Añadir parada: ${n.name}` : 'Haz clic en una estación');
    } else if (this.tool === 'bulldoze') {
      const n = this.nodeAt(p, 300); const tr = n ? null : this.trackAt(p); this.r.setHover(n ? n : null);
      this.showTip(e, n ? `Demoler ${n.name} (devuelve 50 %)` : tr ? `Demoler vía (${km(tr.length)}, devuelve 50 %)` : '');
    } else {
      const n = this.nodeAt(p, 300); this.r.setHover(null);
      if (n) { const tot = Object.values(n.waiting).reduce((a, b) => a + b, 0); this.showTip(e, `${n.name}\n${n.station ? `${num(tot * PAX_UNIT)} esperando` : 'Punto de paso'}${n.doneDay > s.day ? `\nEn obras: ${Math.ceil(n.doneDay - s.day)} días` : ''}`); }
      else { const v = this.r.pickVehicle(nx, ny); if (v) { const m = model(v.model); this.showTip(e, `${m?.name}\n${v.state === 'broken' ? 'Averiado' : `${num(v.loadCount * PAX_UNIT)} / ${num(capacityOf(v) * PAX_UNIT)}`}`); } else this.hideTip(); }
    }
  }
  private click(e: PointerEvent): void {
    if (this.r.photo.active) return;
    const [nx, ny] = this.ndc(e); const p = this.r.pickGround(nx, ny); if (!p) return;
    const g = this.game;
    if (this.tool === 'station' || this.tool === 'waypoint') {
      const err = g.canPlaceNode(p.x, p.y, this.kind); if (err) { this.toast(err, 'warn'); return; }
      const n = g.placeNode(p.x, p.y, this.kind, this.tool === 'station'); if (n) this.select({ type: 'node', id: n.id }, false);
    } else if (this.tool === 'track') {
      const n = this.nodeAt(p); if (!n) return;
      if (this.trackFrom === null) { this.trackFrom = n.id; return; }
      if (n.id === this.trackFrom) { this.trackFrom = null; this.r.setPreview(null, this.kind); return; }
      const tr = g.buildTrack(this.trackFrom, n.id, this.kind);
      if (tr) { this.trackFrom = n.id; this.r.setPreview(null, this.kind); }
    } else if (this.tool === 'line' && this.lineEditing !== null) {
      const n = this.nodeAt(p); if (!n) return;
      const err = g.lineAddStop(this.lineEditing, n.id); if (err) this.toast(err, 'warn');
    } else if (this.tool === 'bulldoze') {
      const n = this.nodeAt(p, 300); if (n) { g.removeNode(n.id); return; }
      const tr = this.trackAt(p); if (tr) g.removeTrack(tr.id);
    } else {
      const v = this.r.pickVehicle(nx, ny); if (v) { this.select({ type: 'vehicle', id: v.id }); return; }
      const n = this.nodeAt(p, 300); if (n) { this.select({ type: 'node', id: n.id }); return; }
      const tr = this.trackAt(p); if (tr) { this.select({ type: 'track', id: tr.id }); return; }
      this.select(null);
    }
  }
  select(sel: Sel, closeSheet = true): void {
    this.selected = sel; if (closeSheet && sel) this.closeSheet(); this.renderSide(); this.updateSelectionRing();
  }
  private updateSelectionRing(): void {
    const s = this.game.s; const sel = this.selected;
    if (!sel) { this.r.setSelection(null); return; }
    if (sel.type === 'node') { const n = s.nodes.find(x => x.id === sel.id); this.r.setSelection(n ?? null, 1.2); }
    else if (sel.type === 'vehicle') { const v = s.vehicles.find(x => x.id === sel.id); const p = v && this.r.vehicleAt(v); this.r.setSelection(p ? { x: p.x * 100, y: p.z * 100 } : null, 0.8); }
    else this.r.setSelection(null);
  }
  private showTip(e: PointerEvent, text: string): void { if (!text) return this.hideTip(); this.els.tip.textContent = text; this.els.tip.classList.remove('hidden'); this.els.tip.style.left = e.clientX + 'px'; this.els.tip.style.top = e.clientY + 'px'; }
  private hideTip(): void { this.els.tip.classList.add('hidden'); }

  private bindKeys(): void {
    window.addEventListener('keydown', e => {
      const tag = (e.target as HTMLElement).tagName; if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (this.r.photo.active) { if (e.key === 'Escape' || e.key.toLowerCase() === 'p') this.togglePhoto(); return; }
      const k = e.key.toLowerCase();
      if (k === 'escape') { if (this.sheet) this.closeSheet(); else this.cancel(); }
      else if (k === ' ') { e.preventDefault(); this.game.togglePause(); }
      else if (k === '1' && !e.ctrlKey) this.game.setSpeed(1); else if (k === '2' && e.shiftKey) this.game.setSpeed(2);
      else if (k === 'q') this.setTool('select'); else if (k === '2') this.setTool('station'); else if (k === '3') this.setTool('track'); else if (k === '4') this.setTool('line'); else if (k === '5') this.setTool('waypoint'); else if (k === 'x') this.setTool('bulldoze');
      else if (k === 'h') { this.r.toggleHeat(); this.renderTools(); }
      else if (k === 'z' && (e.ctrlKey || e.metaKey) && e.shiftKey) this.game.redo(); else if (k === 'z') this.game.undo(); else if (k === 'y') this.game.redo();
      else if (k === 'p') this.togglePhoto();
      else if (k === 'f') this.toggleSheet('finance'); else if (k === 'd') this.toggleSheet('data'); else if (k === 'r') this.toggleSheet('research'); else if (k === 'v') this.toggleSheet('vehicles'); else if (k === 'i') this.toggleSheet('fund'); else if (k === 'c') this.toggleSheet('challenges');
      else if (k === 'n') this.newLine();
      else if (k === '+' || k === '=') this.game.setSpeed(Math.min(8, this.game.s.speed * 2)); else if (k === '-') this.game.setSpeed(Math.max(1, this.game.s.speed / 2));
      else if (k === 's' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); this.onSave(); }
      else if (k === 'tab') { e.preventDefault(); this.cycleRoadRail(); }
      else return;
    });
  }
  private cycleRoadRail(): void { const ks: TrackKind[] = ['road', 'rail', 'metro', 'hsr']; let i = ks.indexOf(this.kind); for (let k = 0; k < 4; k++) { i = (i + 1) % 4; const s = this.game.s; if ((ks[i] === 'metro' && !s.researched.includes('metro')) || (ks[i] === 'hsr' && !s.researched.includes('hsr'))) continue; this.kind = ks[i]; break; } this.renderTools(); }

  // ───────────── hojas ─────────────
  toggleSheet(id: SheetId): void { if (this.sheet === id) this.closeSheet(); else this.openSheet(id); }
  openSheet(id: SheetId): void { this.sheet = id; this.els.sheet.classList.remove('hidden'); this.els.sheet.classList.toggle('wide', !this.selected); this.renderSheet(); this.renderHud(); }
  closeSheet(): void { this.sheet = null; this.els.sheet.classList.add('hidden'); this.renderHud(); }
  renderSheet(): void { if (!this.sheet || this.typing(this.els.sheet)) return; const scroll = this.els.sheet.scrollTop; clear(this.els.sheet); this.els.sheet.appendChild(renderSheet(this, this.sheet)); this.els.sheet.scrollTop = scroll; }

  // ───────────── modo foto ─────────────
  togglePhoto(): void {
    const ph = this.r.photo; ph.active = !ph.active;
    document.body.classList.toggle('photo', ph.active); this.els.photo.classList.toggle('hidden', !ph.active);
    if (ph.active) { this.r.setSelection(null); this.r.setHover(null); this.r.setPreview(null, this.kind); this.renderPhoto(); }
    else { this.r.renderer.domElement.style.filter = ''; this.updateSelectionRing(); }
  }
  private renderPhoto(): void {
    const ph = this.r.photo; const el = this.els.photo; clear(el);
    const slider = (label: string, min: number, max: number, step: number, val: number, on: (v: number) => void) => h('div', null, h('label', null, label), h('input', { type: 'range', min, max, step, value: val, onInput: (e: Event) => on(Number((e.target as HTMLInputElement).value)) }));
    el.append(
      h('div', { class: 'row', style: { justifyContent: 'space-between' } }, h('b', null, t('menu.photo')), btn('✕', () => this.togglePhoto(), 'sm')),
      h('div', { class: 'small muted' }, 'Arrastra para orbitar, rueda para zoom. Esc para salir.'),
      h('label', null, 'Hora del día'),
      h('div', { class: 'row' }, h('input', { type: 'range', min: 0, max: 24, step: 0.1, value: ph.hour ?? this.game.s.hour, style: { flex: 1 }, onInput: (e: Event) => { ph.hour = Number((e.target as HTMLInputElement).value); } }), btn('Hora real', () => { ph.hour = null; this.renderPhoto(); }, 'sm')),
      h('div', { class: 'row small' }, btn('Hora dorada', () => { ph.hour = 17.6; this.renderPhoto(); }, 'sm'), btn('Mediodía', () => { ph.hour = 12.5; this.renderPhoto(); }, 'sm'), btn('Noche', () => { ph.hour = 23; this.renderPhoto(); }, 'sm')),
      slider('Apertura (profundidad de campo)', 0, 0.0004, 0.00001, ph.aperture, v => { ph.aperture = v; }),
      slider('Distancia de enfoque', 3, 300, 1, ph.focus, v => { ph.focus = v; }),
      slider('Viñeteado', 0, 1, 0.05, ph.vignette, v => { ph.vignette = v; }),
      h('label', null, 'Filtro'),
      h('div', { class: 'row' }, ...(['none', 'warm', 'cool', 'mono', 'film'] as const).map(f => btn({ none: 'Natural', warm: 'Cálido', cool: 'Frío', mono: 'B/N', film: 'Película' }[f], () => { ph.filter = f; this.renderPhoto(); }, 'sm' + (ph.filter === f ? ' primary' : '')))),
      h('div', { class: 'row', style: { marginTop: '12px' } }, btn('Capturar PNG (2×)', () => this.capture(2), 'primary'), btn('4×', () => this.capture(4), '')),
    );
  }
  capture(scale: number): void {
    const url = this.r.capture(scale);
    const a = document.createElement('a'); a.href = url; a.download = `lineas-${this.game.s.world.name.replace(/\s+/g, '_')}-${this.game.year()}.png`; a.click();
  }

  // ───────────── tutorial y diarios ─────────────
  renderTutorial(): void {
    const s = this.game.s; const el = this.els.tutorial;
    if (s.tutorial > 5 || (s.tutorial === 5 && s.stats.pax > 40)) { el.classList.add('hidden'); if (s.tutorial === 5) s.tutorial = 6; return; }
    el.classList.remove('hidden');
    const txt = t('tut.' + s.tutorial);
    if (el.dataset.step !== String(s.tutorial)) { el.dataset.step = String(s.tutorial); clear(el); el.append(h('b', null, `Paso ${s.tutorial + 1} de 6`), h('div', null, txt), h('div', { class: 'row', style: { justifyContent: 'flex-end', margin: '4px 0 0' } }, btn('Saltar tutorial', () => { s.tutorial = 99; this.renderTutorial(); }, 'sm'))); }
  }
  renderDailies(): void {
    const s = this.game.s; const el = this.els.dailies; clear(el);
    if (!s.dailies.length) { el.classList.add('hidden'); return; }
    el.classList.remove('hidden');
    el.append(h('div', { class: 'row', style: { justifyContent: 'space-between', margin: '0 0 4px' } }, h('b', null, 'Objetivos de hoy'), h('span', { class: 'muted' }, `Racha ${s.streak} 🔥`)));
    for (const d of s.dailies) el.append(h('div', { class: 'd' + (d.done ? ' done' : '') }, h('span', null, dailyName(d)), h('span', { class: 'mono muted' }, d.done ? '✓' : `${Math.min(100, Math.round(d.progress / d.target * 100))} % · ${money(d.reward)}`)));
  }

  // ───────────── panel lateral ─────────────
  private sideKey = '';
  renderSideLive(): void { if (this.selected || this.lineEditing !== null) this.renderSide(); }
  private typing(el: HTMLElement): boolean { const a = document.activeElement; return !!a && a !== document.body && el.contains(a) && (a.tagName === 'INPUT' || a.tagName === 'SELECT'); }
  renderSide(): void {
    const side = this.els.side; if (this.typing(side)) return;   // no interrumpir al usuario mientras escribe
    const scroll = side.scrollTop; clear(side);
    const s = this.game.s; const sel = this.selected;
    if (!sel) { this.sideKey = ''; this.els.sheet.classList.add('wide'); return; }
    this.els.sheet.classList.remove('wide');
    const close = btn('✕', () => this.select(null), 'sm icon');
    if (sel.type === 'node') { const n = s.nodes.find(x => x.id === sel.id); if (!n) return this.select(null); side.append(this.nodePanel(n, close)); }
    else if (sel.type === 'track') { const tr = s.tracks.find(x => x.id === sel.id); if (!tr) return this.select(null); side.append(this.trackPanel(tr, close)); }
    else if (sel.type === 'line') { const l = s.lines.find(x => x.id === sel.id); if (!l) return this.select(null); side.append(this.linePanel(l, close)); }
    else if (sel.type === 'vehicle') { const v = s.vehicles.find(x => x.id === sel.id); if (!v) return this.select(null); side.append(this.vehiclePanel(v, close)); }
    side.scrollTop = scroll;
    this.updateSelectionRing();
  }
  private nodePanel(n: Node, close: HTMLElement): HTMLElement {
    const s = this.game.s; const g = this.game;
    const waiting = Object.entries(n.waiting).map(([d, c]) => ({ d: Number(d), c })).sort((a, b) => b.c - a.c);
    const total = waiting.reduce((a, b) => a + b.c, 0);
    const lines = s.lines.filter(l => l.stops.includes(n.id));
    const cargo = Object.entries(n.cargo).sort((a, b) => b[1] - a[1]);
    const name = h('input', { type: 'text', value: n.name, style: { width: '100%' }, onChange: (e: Event) => { n.name = (e.target as HTMLInputElement).value; g.markDirty(); } });
    const building = n.doneDay > s.day;
    return h('div', null,
      h('h2', null, h('span', null, n.station ? t('kind.' + n.kind) : t('tool.waypoint')), close),
      name,
      building && h('div', { class: 'row' }, h('span', { class: 'warn' }, `En obras · ${Math.ceil(n.doneDay - s.day)} días`), btn(`${t('common.rush')} ${money(g.rushCost('node', n.id))}`, () => g.rush('node', n.id), 'sm')),
      n.station && section('Estación',
        row('Ciudad', n.cityId !== null ? s.world.cities[n.cityId].name : '—'),
        row('Nivel', `${n.level} / 3`), row('Radio de captación', km(catchment(s, n))),
        row('Esperando', `${num(total * PAX_UNIT)} pasajeros`), row('Subidos este mes', num(n.boarded * PAX_UNIT)), row('Perdidos este mes', num(n.lost * PAX_UNIT)),
        h('div', { class: 'row' }, n.level < 3 && btn(`Ampliar a nivel ${n.level + 1} · ${money(g.stationCost(n.kind, true) * [0, 1, 1.6, 2.8][n.level + 1])}`, () => g.upgradeStation(n.id), 'sm')),
      ),
      n.station && waiting.length > 0 && section('Destinos en espera', h('div', { class: 'list small' }, ...waiting.slice(0, 6).map(w => { const d = s.nodes.find(x => x.id === w.d); return h('div', { class: 'kv' }, h('span', { class: 'k' }, d?.name ?? '?'), h('span', { class: 'v' }, num(w.c * PAX_UNIT))); }))),
      n.station && cargo.length > 0 && section('Carga en espera', h('div', { class: 'list small' }, ...cargo.slice(0, 6).map(([k, v]) => { const [c, d] = k.split(':'); const dn = s.nodes.find(x => x.id === Number(d)); return h('div', { class: 'kv' }, h('span', { class: 'k' }, `${t('cargo.' + c)} → ${dn?.name ?? '?'}`), h('span', { class: 'v' }, `${num(v)} t`)); }))),
      section('Líneas', lines.length ? h('div', { class: 'list' }, ...lines.map(l => h('div', { class: 'item', onClick: () => this.select({ type: 'line', id: l.id }) }, h('span', { class: 'dot', style: { background: '#' + l.color.toString(16).padStart(6, '0') } }), h('span', { class: 'grow' }, l.name), h('span', { class: 'muted small' }, `${l.vehicles.length} veh.`)))) : h('div', { class: 'muted small' }, 'Ninguna línea para aquí todavía.')),
      h('div', { class: 'row' }, btn('Centrar', () => this.r.focusOn(n.x, n.y), 'sm'), btn(`${t('common.delete')} (devuelve ${money(n.cost * 0.5)})`, () => { if (g.removeNode(n.id)) this.select(null); }, 'sm danger')),
    );
  }
  private trackPanel(tr: Track, close: HTMLElement): HTMLElement {
    const s = this.game.s; const g = this.game; const a = s.nodes.find(n => n.id === tr.a), b = s.nodes.find(n => n.id === tr.b);
    const building = tr.doneDay > s.day;
    return h('div', null,
      h('h2', null, h('span', null, `Vía · ${t('kind.' + tr.kind)}`), close),
      h('div', { class: 'muted' }, `${a?.name ?? '?'} ↔ ${b?.name ?? '?'}`),
      building && h('div', { class: 'row' }, h('span', { class: 'warn' }, `En obras · ${Math.ceil(tr.doneDay - s.day)} días`), btn(`${t('common.rush')} ${money(g.rushCost('track', tr.id))}`, () => g.rush('track', tr.id), 'sm')),
      section('Datos', row('Longitud', km(tr.length)), row('Túnel', km(tr.tunnel)), row('Puente', km(tr.bridge)), row('Coste', money(tr.cost)), row('Mantenimiento', `${money(TRACK_SPEC[tr.kind].upkeep * tr.length / 1000 * 30 * 0.5)} / mes`)),
      h('div', { class: 'row' }, btn(`${t('common.delete')} (devuelve ${money(tr.cost * 0.5)})`, () => { if (g.removeTrack(tr.id)) this.select(null); }, 'sm danger')),
    );
  }
  private linePanel(l: Line, close: HTMLElement): HTMLElement {
    const s = this.game.s; const g = this.game; const k = lineKpi(s, l);
    const editing = this.lineEditing === l.id;
    const name = h('input', { type: 'text', value: l.name, style: { flex: 1 }, onChange: (e: Event) => g.lineSet(l.id, { name: (e.target as HTMLInputElement).value }) });
    const vehicles = l.vehicles.map(id => s.vehicles.find(v => v.id === id)).filter((v): v is Vehicle => !!v);
    const chart = h('canvas', { style: { width: '100%' } }); const mc = h('canvas', { style: { width: '100%' } });
    setTimeout(() => {
      const hist = l.history.slice(-24);
      lineChart(chart, [{ name: 'Beneficio', values: hist.map(x => x.revenue - x.cost) }, { name: 'Ingresos', values: hist.map(x => x.revenue), color: '#9a9a9a', dashed: true }], { h: 130, labels: hist.map(x => monthName(x.month % 12)) });
      const f = monteCarlo(l, 24, 400, s.seed);
      lineChart(mc, [{ name: 'P50', values: f.p50 }], { h: 130, band: { lo: f.p10, hi: f.p90 }, labels: f.p50.map((_, i) => `+${i + 1}`) });
    }, 0);
    const colors = s.settings.colorblind ? [0x0072b2, 0xd55e00, 0x009e73, 0xe69f00, 0xcc79a7, 0x56b4e9, 0xf0e442, 0x999999, 0x332288, 0x000000] : LINE_COLORS;
    return h('div', null,
      h('h2', null, h('span', { class: 'dot', style: { background: '#' + l.color.toString(16).padStart(6, '0'), width: '14px', height: '14px' } }), name, close),
      h('div', { class: 'row small muted' }, `${t('kind.' + l.kind)}${l.cargoKind ? ' · mercancías' : ''} · ${l.stops.length} paradas · ${km(l.total)} ida y vuelta`, !l.valid && l.stops.length >= 2 && h('span', { class: 'warn' }, 'Sin ruta: faltan vías')),
      h('div', { class: 'row' },
        btn(editing ? 'Terminar edición' : 'Añadir paradas', () => { if (editing) { this.lineEditing = null; this.setTool('select'); this.select({ type: 'line', id: l.id }); } else { this.tool = 'line'; this.lineEditing = l.id; this.renderTools(); this.renderSide(); } }, 'sm' + (editing ? ' primary' : '')),
        h('label', { class: 'row small' }, h('input', { type: 'checkbox', checked: l.circular, onChange: (e: Event) => g.lineSet(l.id, { circular: (e.target as HTMLInputElement).checked }) }), 'Circular'),
      ),
      section('Paradas', h('div', { class: 'list' }, ...l.stops.map((id, i) => { const n = s.nodes.find(x => x.id === id); return h('div', { class: 'item small' }, h('span', { class: 'muted' }, String(i + 1)), h('span', { class: 'grow', onClick: () => n && this.r.focusOn(n.x, n.y) }, n?.name ?? '?'), btn('✕', () => g.lineRemoveStop(l.id, i), 'sm icon')); }), l.stops.length === 0 && h('div', { class: 'muted small' }, editing ? 'Haz clic en las estaciones del mapa.' : 'Pulsa "Añadir paradas".'))),
      l.cargoKind && section('Mercancía', h('select', { onChange: (e: Event) => { l.cargoKind = (e.target as HTMLSelectElement).value as Line['cargoKind']; g.markDirty(); g.emit({ type: 'lines' }); } }, ...(['ore', 'steel', 'goods', 'wood', 'planks', 'food', 'oil', 'fuel'] as const).map(c => h('option', { value: c, selected: c === l.cargoKind }, t('cargo.' + c)))), h('div', { class: 'small muted' }, 'La línea recoge esta carga en estaciones junto a industrias productoras y la entrega donde se consume.')),
      !l.cargoKind && section('Tarifa', h('div', { class: 'row' }, h('input', { type: 'range', min: 0.5, max: 2, step: 0.05, value: l.fare, style: { flex: 1 }, onInput: (e: Event) => { l.fare = Number((e.target as HTMLInputElement).value); g.markDirty(); } }), h('span', { class: 'mono' }, `×${l.fare.toFixed(2)}`)), h('div', { class: 'small muted' }, `≈ ${(eraOf(s.era).fare * (0.6 + 0.25 * Math.pow(2, 0.75)) * l.fare / PAX_UNIT).toFixed(2)} € por viajero y 2 km. Elasticidad −1,2: subir precio reduce demanda.`)),
      section('Color', h('div', { class: 'row' }, ...colors.map(c => h('div', { class: 'swatch' + (c === l.color ? ' on' : ''), style: { background: '#' + c.toString(16).padStart(6, '0') }, onClick: () => g.lineSet(l.id, { color: c }) })))),
      section('Vehículos', h('div', { class: 'list' }, ...vehicles.map(v => { const m = model(v.model); return h('div', { class: 'item small', onClick: () => this.select({ type: 'vehicle', id: v.id }) }, h('span', { class: 'grow' }, m?.name ?? v.model), h('span', { class: 'muted' }, v.state === 'delivery' ? `entrega ${Math.ceil(v.timer)} d` : v.state === 'broken' ? 'avería' : `${Math.round(v.loadCount / Math.max(1, capacityOf(v)) * 100)} %`)); })), h('div', { class: 'row' }, btn('Comprar vehículo', () => this.openSheet('vehicles'), 'primary sm'))),
      section('Rendimiento (12 meses)', h('div', { class: 'grid2 small' },
        row('Ingresos', money(k.revenue)), row('Costes', money(k.cost)), row('Beneficio', h('span', { class: k.profit >= 0 ? 'ok' : 'neg' }, signed(k.profit))), row('ROI acumulado', pct(k.roi)),
        row('Recuperación', k.payback === null ? '—' : `${num(k.payback, 0)} meses`), row('Factor de carga', pct(k.load)), row('Puntualidad', pct(k.punct)), row('Pasajeros/h', num(k.paxPerHour * PAX_UNIT, 0)),
        row('Pax 12 m', num(k.pax * PAX_UNIT)), row('Carga 12 m', `${num(k.cargo)} t`), row('Invertido', money(k.invested)), row('Mes actual', signed(l.mRevenue - l.mCost))),
        meter(k.load, 'factor de carga')),
      section('Historial mensual', chart),
      section('Previsión Monte Carlo (beneficio acumulado, 24 meses, banda P10–P90)', mc, h('div', { class: 'small muted' }, k.months < 3 ? 'Con menos de 3 meses de historial se usa una volatilidad por defecto del 35 %.' : `Volatilidad estimada de ${k.months} meses de historial.`)),
      h('div', { class: 'row' }, btn(t('common.delete'), () => { g.deleteLine(l.id); this.select(null); }, 'sm danger')),
    );
  }
  private vehiclePanel(v: Vehicle, close: HTMLElement): HTMLElement {
    const s = this.game.s; const g = this.game; const m = model(v.model)!; const line = v.lineId !== null ? s.lines.find(l => l.id === v.lineId) : null;
    const canTune = s.researched.includes('tuning');
    const up = (part: keyof Vehicle['upgrades'], label: string, effect: string) => h('div', { class: 'row', style: { justifyContent: 'space-between' } }, h('span', null, `${label} `, h('span', { class: 'muted small' }, effect)), h('span', { class: 'row' }, h('span', { class: 'mono' }, '●'.repeat(v.upgrades[part]) + '○'.repeat(3 - v.upgrades[part])), btn(v.upgrades[part] >= 3 ? 'Máx.' : money(g.upgradeCost(v, part)), () => g.upgradeVehicle(v.id, part), 'sm')));
    const nVar = variantsUnlocked(s, v.model);
    const lines = s.lines.filter(l => g.lineAccepts(l, m));
    return h('div', null,
      h('h2', null, h('span', null, m.name), close),
      h('div', { class: 'muted small' }, `${t('mode.' + m.mode)} · ${m.year} · ${m.desc}`),
      v.state === 'delivery' && h('div', { class: 'row' }, h('span', { class: 'warn' }, `En entrega · ${Math.ceil(v.timer)} días`), btn(`${t('common.rush')} ${money(g.rushCost('vehicle', v.id))}`, () => g.rush('vehicle', v.id), 'sm')),
      v.state === 'broken' && h('div', { class: 'warn' }, `Averiado · ${v.timer.toFixed(1)} días para la reparación`),
      section('Ficha técnica', h('div', { class: 'grid2 small' },
        row('Velocidad', `${num(effSpeedKmh(v))} km/h`), row('Capacidad', m.cargo ? `${m.capacity} t` : `${num(m.capacity * PAX_UNIT)} pax`), row('Aceleración', `${(m.accel * (1 + 0.15 * v.upgrades.brakes)).toFixed(2)} m/s²`),
        row('Consumo', `${num(m.consumption * (1 + 0.08 * v.upgrades.engine - 0.06 * v.upgrades.aero), 1)} L/100 km eq.`), row('Fiabilidad', pct(m.reliability + 0.015 * v.upgrades.brakes, 1)), row('Mantenimiento', `${money(m.maintenance)}/mes`),
        row('Kilómetros', `${num(v.km)} km`), row('Desgaste', pct(v.wear)), row('Averías', String(v.breakdowns)), row('Valor contable', money(bookValue(v, s.day))), row('Viajes', String(v.trips)), row('Ocupación', m.cargo ? `${num(v.cargoLoad)} t` : `${num(v.loadCount * PAX_UNIT)} pax`)),
        meter(v.wear, 'desgaste')),
      section('Línea', h('select', { onChange: (e: Event) => { const val = (e.target as HTMLSelectElement).value; g.assignVehicle(v.id, val === '' ? null : Number(val)); } }, h('option', { value: '', selected: !line }, 'Sin asignar (depósito)'), ...lines.map(l => h('option', { value: l.id, selected: line?.id === l.id }, l.name)))),
      section('Tuning' + (canTune ? '' : ' (requiere "Taller de competición")'),
        up('engine', 'Motor', '+8 % velocidad, +8 % consumo, +10 % desgaste'), up('brakes', 'Frenos', '+15 % aceleración, +1,5 % fiabilidad'), up('aero', 'Aerodinámica', '−6 % consumo, +2 % velocidad'), up('comfort', 'Confort', '+5 % tarifa, +3 % demanda, +2 % consumo')),
      section(`Librea (${nVar * 2} de ${LIVERIES.length} desbloqueadas por km del modelo)`, h('div', { class: 'row' }, h('div', { class: 'swatch' + (v.livery === 0 ? ' on' : ''), style: { background: line ? '#' + line.color.toString(16).padStart(6, '0') : '#999' }, title: 'Color de línea', onClick: () => g.setLivery(v.id, 0) }), ...LIVERIES.slice(1).map((c, i) => h('div', { class: 'swatch' + (v.livery === i + 1 ? ' on' : ''), style: { background: '#' + c.toString(16).padStart(6, '0'), opacity: i + 1 < nVar * 2 ? '1' : '0.25' }, title: i + 1 < nVar * 2 ? 'Aplicar' : 'Bloqueada', onClick: () => g.setLivery(v.id, i + 1) })))),
      h('div', { class: 'row' }, btn('Seguir', () => { const p = this.r.vehicleAt(v); if (p) this.r.focusOn(p.x * 100, p.z * 100); }, 'sm'), btn(`${t('common.sell')} (${money(bookValue(v, s.day) * 0.7)})`, () => { g.sellVehicle(v.id); this.select(null); }, 'sm danger')),
    );
  }
}

function distSeg(p: Vec2, a: Vec2, b: Vec2): number {
  const dx = b.x - a.x, dy = b.y - a.y; const l2 = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / l2));
  return Math.hypot(p.x - (a.x + dx * t), p.y - (a.y + dy * t));
}
