import type { App } from './app';
import { h, money, num, pct, btn, row, section, signed, download, pickFile, km } from './dom';
import { t, monthName, setLang } from './i18n';
import { DATA, model, tech, eraOf, applyMod, validateMod, type VehicleModel } from '../sim/data';
import { lineKpi, globalKpis, ledgerCsv, linesCsv } from '../sim/stats';
import { companyValue, loanRate, maxLoan, takeLoan, repayLoan, debt, fundValue, vehiclesValue, infraValue } from '../sim/economy';
import { lineChart, barChart, meter } from './charts';
import { PAX_UNIT, BAL } from '../sim/constants';
import { licenseMetrics, isoWeek, weeklyCode, verifyWeeklyCode, weeklyDone, WEEKLY_YEARS, collectionSummary, VARIANT_KM } from '../sim/challenges';
import { PRESTIGE_TREE, prestigeSpend, prestigeAvailable, canRefound, refoundPoints } from '../sim/progress';
import { listSlots, saveSlot, loadSlot, deleteSlot, serialize, deserialize, setPref } from '../sim/save';
import { SCENARIOS } from '../sim/scenarios';

export type SheetId = 'finance' | 'data' | 'research' | 'vehicles' | 'fund' | 'challenges' | 'prestige' | 'settings' | 'collection';

export function renderSheet(app: App, id: SheetId): HTMLElement {
  const header = (title: string, ...extra: HTMLElement[]) => h('header', null, h('h2', null, title), h('div', { class: 'row' }, ...extra, btn('✕', () => app.closeSheet(), 'sm icon')));
  const body = h('div', null);
  switch (id) {
    case 'finance': body.append(header(t('menu.finance')), finance(app)); break;
    case 'data': body.append(header(t('menu.data'), btn('CSV libro mayor', () => download('lineas-libro-mayor.csv', ledgerCsv(app.game.s), 'text/csv'), 'sm'), btn('CSV líneas', () => download('lineas-lineas.csv', linesCsv(app.game.s), 'text/csv'), 'sm')), data(app)); break;
    case 'research': body.append(header(t('menu.research')), research(app)); break;
    case 'vehicles': body.append(header(t('menu.vehicles'), btn(t('menu.collection'), () => app.openSheet('collection'), 'sm')), vehicles(app)); break;
    case 'collection': body.append(header(t('menu.collection'), btn(t('menu.vehicles'), () => app.openSheet('vehicles'), 'sm')), collection(app)); break;
    case 'fund': body.append(header(t('menu.fund')), fund(app)); break;
    case 'challenges': body.append(header(t('menu.challenges')), challenges(app)); break;
    case 'prestige': body.append(header(t('menu.prestige')), prestige(app)); break;
    case 'settings': body.append(header(t('menu.settings')), settings(app)); break;
  }
  return body;
}

function kpi(k: string, v: string, s?: string): HTMLElement { return h('div', { class: 'kpi' }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v), s ? h('div', { class: 's' }, s) : null); }

// ───────────── Finanzas ─────────────
function finance(app: App): HTMLElement {
  const s = app.game.s; const g = app.game;
  const last = s.ledger.slice(-12); const cur = s.cur;
  const rowsDef: [string, (m: typeof cur) => number][] = [
    ['Billetes', m => m.fares], ['Mercancías', m => m.cargo], ['Subvenciones y objetivos', m => m.subsidies], ['Fondo (plusvalías)', m => m.fundGain], ['Progreso offline', m => m.offline],
    ['Combustible/energía', m => -m.running], ['Mantenimiento', m => -m.maintenance], ['Infraestructura', m => -m.infra], ['Personal', m => -m.staff], ['Intereses', m => -m.interest], ['Depreciación', m => -m.depreciation],
  ];
  const months = [...last, cur];
  const table = h('table', null,
    h('thead', null, h('tr', null, h('th', null, 'Cuenta de resultados'), ...months.map(m => h('th', null, `${monthName(m.month)} ${m.year}`)))),
    h('tbody', null,
      ...rowsDef.map(([name, f]) => h('tr', null, h('td', null, name), ...months.map(m => { const v = f(m); return h('td', { class: v < 0 ? 'muted' : '' }, v === 0 ? '·' : money(v)); }))),
      h('tr', { style: { fontWeight: '600' } }, h('td', null, 'Beneficio'), ...months.map((m, i) => { const p = i === months.length - 1 ? cur.fares + cur.cargo + cur.subsidies + cur.fundGain + cur.offline - cur.running - cur.maintenance - cur.infra - cur.staff - cur.interest - cur.depreciation : m.profit; return h('td', { class: p < 0 ? 'neg' : 'ok' }, signed(p)); })),
    ));
  const chart = h('canvas', { style: { width: '100%' } });
  setTimeout(() => barChart(chart, s.ledger.slice(-36).map(m => m.profit), { h: 140, labels: s.ledger.slice(-36).map(m => monthName(m.month)) }), 0);
  const value = companyValue(s);
  const loanInput = h('input', { type: 'number', value: Math.min(100000, maxLoan(s)), min: 1000, step: 1000, style: { width: '140px' } }) as HTMLInputElement;
  return h('div', null,
    h('div', { class: 'grid3', style: { marginBottom: '14px' } },
      kpi('Caja', money(s.cash)), kpi('Valor de compañía', money(value), 'caja + vehículos + 50 % infra + fondo − deuda'), kpi('Beneficio 12 meses', signed(last.reduce((a, m) => a + m.profit, 0))),
      kpi('Ciclo económico', `${pct(s.cycle - 1, 1)}`, s.cycle < 0.9 ? 'Recesión' : s.cycle > 1.08 ? 'Expansión' : 'Normal'), kpi('Precio del combustible', `${pct(s.fuel - 1, 0)}`, `base ${eraOf(s.era).fuel} €/u`), kpi('Reputación', `${Math.round(s.reputation)} / 100`, 'demanda × ' + (0.7 + 0.6 * s.reputation / 100).toFixed(2))),
    section('Beneficio mensual (36 meses)', chart),
    section('Cuenta de resultados', h('div', { style: { overflow: 'auto' } }, table)),
    section('Balance', h('div', { class: 'grid2' },
      h('div', null, row('Caja', money(s.cash)), row('Vehículos (valor contable)', money(vehiclesValue(s))), row('Infraestructura (coste)', money(infraValue(s))), row('Fondo indexado', money(fundValue(s)))),
      h('div', null, row('Deuda', money(debt(s))), row('Patrimonio', money(s.cash + vehiclesValue(s) + infraValue(s) + fundValue(s) - debt(s))), row('Valor máximo alcanzado', money(s.stats.maxValue))))),
    section('Préstamos', h('div', { class: 'small muted' }, `Tipo actual ${pct(loanRate(s), 1)} anual · máximo ${money(maxLoan(s))} · amortización en ${BAL.loanYears} años · hasta 3 préstamos`),
      h('div', { class: 'row' }, loanInput, btn('Pedir préstamo', () => { if (takeLoan(s, Number(loanInput.value))) { app.toast('Préstamo concedido', 'good'); app.renderSheet(); } else app.toast('No es posible: revisa el importe y el máximo', 'warn'); }, 'primary sm')),
      h('div', { class: 'list' }, ...s.loans.map(l => h('div', { class: 'item' }, h('span', { class: 'grow' }, `${money(l.principal)} pendiente · ${pct(l.rate, 1)} · ${l.monthsLeft} meses · cuota ${money(l.payment)}`), btn('Amortizar todo', () => { if (repayLoan(s, l.id)) app.renderSheet(); else app.toast('Caja insuficiente', 'warn'); }, 'sm'))))),
    section('Contratos', s.contracts.filter(c => c.active).length ? h('div', { class: 'list' }, ...s.contracts.filter(c => c.active).map(c => h('div', { class: 'item' }, h('span', { class: 'grow' }, `${s.world.cities[c.cityId].name}: ${num(c.progress * PAX_UNIT)} / ${num(c.pax * PAX_UNIT)} pasajeros · ${c.monthsLeft} meses · recompensa ${money(c.reward)}`), meter(c.progress / c.pax)))) : h('div', { class: 'muted small' }, 'Las ciudades ofrecen contratos cada año cuando tienen servicio.')),
    h('div', { class: 'small muted' }, `Ingresos totales de la compañía: ${money(g.s.lines.reduce((a, l) => a + l.revenue, 0))} · Pasajeros transportados: ${num(s.stats.pax * PAX_UNIT)} · Carga: ${num(s.stats.cargo)} t`),
  );
}

// ───────────── Datos ─────────────
function data(app: App): HTMLElement {
  const s = app.game.s; const k = globalKpis(s);
  const vchart = h('canvas', { style: { width: '100%' } }); const pchart = h('canvas', { style: { width: '100%' } });
  setTimeout(() => {
    const led = s.ledger.slice(-60);
    lineChart(vchart, [{ name: 'Valor', values: led.map(m => m.value) }, { name: 'Caja', values: led.map(m => m.cash), color: '#9a9a9a', dashed: true }], { h: 150, labels: led.map(m => `${monthName(m.month)} ${m.year}`) });
    lineChart(pchart, [{ name: 'Pasajeros/mes', values: led.map(m => m.pax * PAX_UNIT) }, { name: 'Población', values: led.map(m => m.pop), color: '#9a9a9a', dashed: true }], { h: 150, labels: led.map(m => monthName(m.month)) });
  }, 0);
  const sortKey = (app as unknown as { dataSort?: string }).dataSort ?? 'profit';
  const rows = s.lines.map(l => ({ l, k: lineKpi(s, l) })).sort((a, b) => (b.k as unknown as Record<string, number>)[sortKey] - (a.k as unknown as Record<string, number>)[sortKey]);
  const th = (label: string, key: string) => h('th', { style: { cursor: 'pointer' }, onClick: () => { (app as unknown as { dataSort?: string }).dataSort = key; app.renderSheet(); } }, label + (sortKey === key ? ' ▾' : ''));
  return h('div', null,
    h('div', { class: 'grid3', style: { marginBottom: '14px' } },
      kpi('Valor de compañía', money(k.value)), kpi('Beneficio 12 m', signed(k.profit12)), kpi('Pasajeros 12 m', num(k.pax12 * PAX_UNIT)),
      kpi('Puntualidad', pct(k.punct)), kpi('Población servida', num(k.pop)), kpi('Flota', `${k.vehicles} vehículos · ${k.lines} líneas · ${k.stations} estaciones`)),
    h('div', { class: 'grid2' }, section('Valor y caja (60 meses)', vchart), section('Pasajeros y población', pchart)),
    section('Líneas', h('table', null,
      h('thead', null, h('tr', null, h('th', null, 'Línea'), th('Beneficio 12 m', 'profit'), th('Ingresos', 'revenue'), th('Costes', 'cost'), th('Pax', 'pax'), th('Carga', 'cargo'), th('Carga %', 'load'), th('Puntual.', 'punct'), th('ROI', 'roi'), th('Recup. (m)', 'payback'), th('Veh.', 'vehicles'))),
      h('tbody', null, ...rows.map(({ l, k }) => h('tr', { class: 'clickable', onClick: () => { app.select({ type: 'line', id: l.id }, false); app.openSheet('data'); } },
        h('td', null, h('span', { class: 'dot', style: { background: '#' + l.color.toString(16).padStart(6, '0'), marginRight: '6px' } }), l.name),
        h('td', { class: k.profit < 0 ? 'neg' : 'ok' }, signed(k.profit)), h('td', null, money(k.revenue)), h('td', null, money(k.cost)), h('td', null, num(k.pax * PAX_UNIT)), h('td', null, `${num(k.cargo)} t`),
        h('td', null, pct(k.load)), h('td', null, pct(k.punct)), h('td', null, pct(k.roi)), h('td', null, k.payback === null ? '—' : num(k.payback)), h('td', null, String(k.vehicles))))),
    ), !s.lines.length && h('div', { class: 'muted small' }, 'Crea una línea para ver sus datos.')),
    section('Ciudades', h('table', null, h('thead', null, h('tr', null, h('th', null, 'Ciudad'), h('th', null, 'Población'), h('th', null, 'Cobertura'), h('th', null, 'Servicio'), h('th', null, 'Tendencia'))),
      h('tbody', null, ...[...s.world.cities].sort((a, b) => b.pop - a.pop).map(c => { const cov = app.game.cityCov.get(c.id) ?? 0; const g = cov > 0.05 ? BAL.cityGrowth * (c.service * cov - 0.25) : -BAL.cityDecline; return h('tr', { class: 'clickable', onClick: () => app.r.focusOn(c.x, c.y) }, h('td', null, c.name), h('td', null, num(c.pop)), h('td', null, pct(cov)), h('td', null, pct(c.service)), h('td', { class: g >= 0 ? 'ok' : 'neg' }, `${g >= 0 ? '+' : ''}${(g * 100).toFixed(2)} %/mes`)); })))),
    section('Industrias', h('table', null, h('thead', null, h('tr', null, h('th', null, 'Industria'), h('th', null, 'Produce'), h('th', null, 'Consume'), h('th', null, 'Nivel'), h('th', null, 'Stock'), h('th', null, 'Ritmo'))),
      h('tbody', null, ...s.world.industries.map(i => h('tr', { class: 'clickable', onClick: () => app.r.focusOn(i.x, i.y) }, h('td', null, i.name), h('td', null, t('cargo.' + i.output)), h('td', null, i.input ? t('cargo.' + i.input) : '—'), h('td', null, String(i.level)), h('td', null, `${num(i.stock)} t`), h('td', null, `${num(i.rate * i.level)} t/mes`)))))),
  );
}

// ───────────── Investigación ─────────────
function research(app: App): HTMLElement {
  const s = app.game.s; const g = app.game; const av = g.researchAvailable();
  const cur = s.research; const curT = cur && tech(cur.id);
  const byEra = new Map<number, typeof DATA.tech>();
  for (const n of DATA.tech) (byEra.get(n.era) ?? byEra.set(n.era, []).get(n.era)!).push(n);
  return h('div', null,
    cur && curT ? section('En curso', h('div', { class: 'item' }, h('div', { class: 'grow' }, h('b', null, curT.name), h('div', { class: 'small muted' }, `${Math.ceil(cur.doneDay - s.day)} días restantes`), meter((s.day - cur.startDay) / Math.max(1, cur.doneDay - cur.startDay))), btn(`${t('common.rush')} ${money(g.rushCost('research', cur.id))}`, () => { g.rush('research', cur.id); app.renderSheet(); }, 'sm'))) : h('div', { class: 'muted small', style: { marginBottom: '10px' } }, 'Ninguna investigación en curso. Solo puede haber una a la vez.'),
    h('div', { class: 'small muted', style: { marginBottom: '8px' } }, `Era actual: ${eraOf(s.era).name} (${eraOf(s.era).year}). Año ${g.year()}. Las eras se investigan desde 10 años antes de su fecha.`),
    ...[...byEra.entries()].sort((a, b) => a[0] - b[0]).map(([era, nodes]) => section(`${eraOf(era).name} · ${eraOf(era).year}`, h('div', { class: 'cards' }, ...nodes.map(n => {
      const a = av.find(x => x.id === n.id)!; const done = s.researched.includes(n.id);
      return h('div', { class: 'card' + (done || a.ok ? '' : ' locked') }, h('h4', null, n.name, done && h('span', { class: 'tag', style: { marginLeft: '6px' } }, 'Hecho')), h('div', { class: 'small muted' }, n.desc),
        h('div', { class: 'row small', style: { justifyContent: 'space-between', marginTop: '6px' } }, h('span', { class: 'mono' }, `${money(n.cost)} · ${n.days} d`), done ? null : a.ok ? btn('Investigar', () => { g.startResearch(n.id); app.renderSheet(); }, 'primary sm') : h('span', { class: 'muted' }, a.why)));
    })))),
  );
}

// ───────────── Vehículos ─────────────
function vehicles(app: App): HTMLElement {
  const s = app.game.s; const g = app.game; const st = app as unknown as { vehMode?: string; vehLine?: number };
  const sel = app.selected; const selLine = sel?.type === 'line' ? s.lines.find(l => l.id === sel.id) : null;
  const target = st.vehLine !== undefined ? s.lines.find(l => l.id === st.vehLine) ?? selLine : selLine;
  const modes = ['bus', 'tram', 'train', 'metro', 'hsr', 'truck', 'freight'];
  const mode = st.vehMode ?? (target ? (target.cargoKind ? (target.kind === 'road' ? 'truck' : 'freight') : target.kind === 'road' ? 'bus' : target.kind === 'rail' ? 'tram' : target.kind) : 'tram');
  const avail = g.availableModels();
  const list = DATA.vehicles.filter(m => m.mode === mode).sort((a, b) => a.year - b.year);
  const card = (m: VehicleModel) => {
    const ok = avail.some(x => x.id === m.id); const owned = s.collection.owned[m.id];
    const compatible = target ? g.lineAccepts(target, m) : true;
    return h('div', { class: 'card' + (ok ? '' : ' locked') },
      h('h4', null, m.name, owned && h('span', { class: 'tag', style: { marginLeft: '6px' } }, `${num(owned.km)} km`), m.mod && h('span', { class: 'tag', style: { marginLeft: '6px' } }, 'mod')),
      h('div', { class: 'small muted' }, `${t('mode.' + m.mode)} · ${m.year} · ${t('kind.' + m.track)}`),
      h('div', { class: 'specs' }, h('span', null, 'Velocidad ', h('b', null, `${m.speed} km/h`)), h('span', null, 'Capacidad ', h('b', null, m.cargo ? `${m.capacity} t` : num(m.capacity * PAX_UNIT))), h('span', null, 'Fiabilidad ', h('b', null, pct(m.reliability))), h('span', null, 'Mant. ', h('b', null, `${money(m.maintenance)}/mes`)), h('span', null, 'Consumo ', h('b', null, `${m.consumption} L/100`)), h('span', null, 'Vida útil ', h('b', null, `${m.lifespan} años`))),
      h('div', { class: 'row', style: { justifyContent: 'space-between' } }, h('b', { class: 'mono' }, money(m.price)), ok ? btn(target && compatible ? `Comprar para ${target.name}` : 'Comprar (depósito)', () => { const v = g.buyVehicle(m.id, target && compatible ? target.id : null); if (v) app.toast(`${m.name} comprado. Entrega en ${Math.ceil(v.timer)} días.`, 'good'); }, 'primary sm') : h('span', { class: 'small muted' }, m.unlock ? `Requiere: ${tech(m.unlock)?.name ?? m.unlock}` : `Era ${eraOf(m.era).year}`)));
  };
  return h('div', null,
    h('div', { class: 'row', style: { marginBottom: '8px' } }, h('span', { class: 'muted small' }, 'Destino:'), h('select', { onChange: (e: Event) => { const v = (e.target as HTMLSelectElement).value; st.vehLine = v === '' ? -1 : Number(v); app.renderSheet(); } }, h('option', { value: '', selected: !target }, 'Depósito (sin línea)'), ...s.lines.map(l => h('option', { value: l.id, selected: target?.id === l.id }, l.name))), h('span', { class: 'small muted' }, `Caja ${money(s.cash)}`)),
    h('div', { class: 'tabs' }, ...modes.map(m => h('button', { class: m === mode ? 'on' : '', onClick: () => { st.vehMode = m; app.renderSheet(); } }, t('mode.' + m)))),
    h('div', { class: 'cards' }, ...list.map(card)),
    s.vehicles.length > 0 && section('Flota', h('table', null, h('thead', null, h('tr', null, h('th', null, 'Vehículo'), h('th', null, 'Línea'), h('th', null, 'Estado'), h('th', null, 'km'), h('th', null, 'Desgaste'), h('th', null, 'Averías'), h('th', null, 'Valor'))),
      h('tbody', null, ...s.vehicles.map(v => { const m = model(v.model); const l = v.lineId !== null ? s.lines.find(x => x.id === v.lineId) : null; return h('tr', { class: 'clickable', onClick: () => app.select({ type: 'vehicle', id: v.id }) }, h('td', null, m?.name ?? v.model), h('td', null, l?.name ?? '—'), h('td', null, { moving: 'En ruta', dwell: 'En parada', broken: 'Averiado', delivery: 'En entrega', idle: 'Depósito' }[v.state]), h('td', null, num(v.km)), h('td', null, pct(v.wear)), h('td', null, String(v.breakdowns)), h('td', null, money(v.price * Math.max(0.1, 1 - (s.day - v.bought) / 360 / 20)))); })))),
  );
}

function collection(app: App): HTMLElement {
  const s = app.game.s; const sum = collectionSummary(s);
  const modes = ['bus', 'tram', 'train', 'metro', 'hsr', 'truck', 'freight'];
  return h('div', null,
    h('div', { class: 'grid3', style: { marginBottom: '14px' } }, kpi('Colección', `${sum.owned} / ${sum.total}`, 'modelos comprados alguna vez'), kpi('Kilómetros de flota', `${num(s.stats.km)} km`), kpi('Vehículos comprados', String(s.stats.vehiclesBought))),
    h('div', { class: 'small muted', style: { marginBottom: '10px' } }, `Cada modelo desbloquea libreas a los ${VARIANT_KM.slice(1).join(', ')} km recorridos. La colección se conserva al refundar.`),
    ...modes.map(mode => section(t('mode.' + mode), h('div', { class: 'cards' }, ...DATA.vehicles.filter(m => m.mode === mode).sort((a, b) => a.year - b.year).map(m => { const o = s.collection.owned[m.id]; const vars = o ? VARIANT_KM.filter(k => o.km >= k).length : 0; return h('div', { class: 'card' + (o ? '' : ' locked') }, h('h4', null, m.name), h('div', { class: 'small muted' }, `${m.year} · ${eraOf(m.era).name}`), o ? h('div', { class: 'small' }, `${num(o.km)} km · libreas ${vars}/4`, meter(Math.min(1, o.km / VARIANT_KM[3]))) : h('div', { class: 'small muted' }, 'Sin comprar')); })))),
  );
}

// ───────────── Fondo ─────────────
function fund(app: App): HTMLElement {
  const s = app.game.s; const g = app.game; const f = s.fund; const value = fundValue(s);
  const chart = h('canvas', { style: { width: '100%' } });
  setTimeout(() => lineChart(chart, [{ name: 'NAV', values: f.history.slice(-120) }], { h: 160, zero: false }), 0);
  const inv = h('input', { type: 'number', value: Math.min(50000, Math.max(0, Math.floor(s.cash / 1000) * 1000)), step: 1000, min: 0, style: { width: '140px' } }) as HTMLInputElement;
  const wd = h('input', { type: 'number', value: Math.floor(value / 1000) * 1000, step: 1000, min: 0, style: { width: '140px' } }) as HTMLInputElement;
  const dca = h('input', { type: 'number', value: f.dca, step: 500, min: 0, style: { width: '140px' } }) as HTMLInputElement;
  const gain = value - f.invested; const years = Math.max(0.1, s.day / 360);
  const fee = BAL.fundFee * (s.researched.includes('fund') ? 0.5 : 1);
  // simulación de interés compuesto (determinista, sin azar) para comparar con reinvertir
  const proj = (monthly: number, years: number, rate: number) => { let v = value; for (let m = 0; m < years * 12; m++) v = v * (1 + rate / 12) + monthly; return v; };
  return h('div', null,
    h('div', { class: 'grid3', style: { marginBottom: '14px' } }, kpi('Valor del fondo', money(value)), kpi('Aportado', money(f.invested)), kpi('Plusvalía', signed(gain), f.invested > 0 ? pct(gain / f.invested, 1) : '—'),
      kpi('NAV', num(f.nav, 2), `inicio 100 · ${pct(Math.pow(f.nav / 100, 1 / years) - 1, 1)} anualizado`), kpi('DCA mensual', money(f.dca), 'se compra el día 1 de cada mes si hay caja'), kpi('Comisión', pct(fee, 2) + ' anual', 'esperado 6,5 % · volatilidad 16 %')),
    section('Evolución del NAV (120 meses)', chart),
    h('div', { class: 'grid2' },
      section('Invertir', h('div', { class: 'row' }, inv, btn('Comprar participaciones', () => { if (g.fundInvest(Number(inv.value))) app.renderSheet(); else app.toast('Caja insuficiente', 'warn'); }, 'primary sm')),
        h('div', { class: 'row' }, wd, btn('Vender', () => { if (g.fundWithdraw(Number(wd.value))) app.renderSheet(); }, 'sm')),
        h('div', { class: 'row' }, dca, btn('Fijar DCA', () => { g.setDca(Number(dca.value)); app.toast('DCA actualizado', 'good'); app.renderSheet(); }, 'sm'))),
      section('Interés compuesto', h('div', { class: 'small' },
        row('En 10 años con el DCA actual (6,5 %)', money(proj(f.dca, 10, 0.065 - fee))), row('En 20 años', money(proj(f.dca, 20, 0.065 - fee))), row('En 30 años', money(proj(f.dca, 30, 0.065 - fee))),
        h('p', { class: 'muted' }, 'Compara con el ROI de tus líneas en el panel de Datos: una línea nueva suele rendir más que el fondo, pero el fondo no se avería ni necesita gestión. La decisión es tuya.')))),
  );
}

// ───────────── Retos ─────────────
function challenges(app: App): HTMLElement {
  const s = app.game.s; const metrics = licenseMetrics(s);
  const fmtv = (v: number, unit: string) => unit === '%' ? pct(v) : unit === '€' ? money(v) : `${num(v)} ${unit}`;
  const codeInput = h('input', { type: 'text', placeholder: 'LN-2026W42-...', style: { width: '260px' } }) as HTMLInputElement;
  const verify = h('span', { class: 'small muted' });
  const week = isoWeek();
  return h('div', null,
    section('Reto semanal', h('div', { class: 'card' },
      h('div', { class: 'row', style: { justifyContent: 'space-between' } }, h('div', null, h('b', null, `Semana ${week}`), h('div', { class: 'small muted' }, `Mapa aleatorio con semilla de la semana. Objetivo: máximo valor de compañía en ${WEEKLY_YEARS} años. Comparte el código con tus amigos: cada uno juega el mismo mapa.`)), btn('Jugar el reto de esta semana', () => app.onMenu(), 'primary sm')),
      s.weekly && h('div', { style: { marginTop: '8px' } }, weeklyDone(s) ? h('div', null, h('b', null, 'Reto completado. Tu código: '), h('code', { class: 'mono', style: { userSelect: 'all' } }, weeklyCode(s)), ' ', btn('Copiar', () => { navigator.clipboard?.writeText(weeklyCode(s)); app.toast('Código copiado', 'good'); }, 'sm')) : h('div', { class: 'small' }, `Partida de reto en curso (${s.weeklyWeek}): ${Math.floor(s.day / 360)} de ${WEEKLY_YEARS} años · valor actual ${money(companyValue(s))}`)),
      h('div', { class: 'row', style: { marginTop: '8px' } }, codeInput, btn('Verificar código', () => { const r = verifyWeeklyCode(codeInput.value); verify.textContent = !r ? 'Formato no válido' : r.ok ? `Válido: semana ${r.week}, valor ${money(r.value)}` : 'Código manipulado o incorrecto'; }, 'sm'), verify))),
    section('Objetivos diarios', h('div', { class: 'small muted' }, `Racha actual: ${s.streak} días. Completa los tres objetivos cada día para mantenerla; la recompensa crece con la racha.`)),
    section('Licencias', h('div', { class: 'cards' }, ...DATA.licenses.map(l => { const med = s.medals.find(m => m.id === l.id); const best = med?.best ?? 0; const cur = metrics[l.metric] ?? 0; const tier = med?.tier ?? 0; const next = tier < 3 ? l.tiers[tier] : null;
      return h('div', { class: 'card' }, h('h4', null, h('span', { class: 'medal t' + tier }), l.name), h('div', { class: 'small muted' }, l.desc),
        h('div', { class: 'small', style: { marginTop: '6px' } }, `Actual ${fmtv(cur, l.unit)} · mejor ${fmtv(best, l.unit)}`), next !== null ? meter(Math.min(1, best / next), `siguiente: ${fmtv(next, l.unit)}`) : h('div', { class: 'small ok' }, 'Oro conseguido'),
        h('div', { class: 'row small muted', style: { gap: '12px' } }, h('span', null, h('span', { class: 'medal t1' }), fmtv(l.tiers[0], l.unit)), h('span', null, h('span', { class: 'medal t2' }), fmtv(l.tiers[1], l.unit)), h('span', null, h('span', { class: 'medal t3' }), fmtv(l.tiers[2], l.unit)))); }))),
  );
}

// ───────────── Prestigio ─────────────
function prestige(app: App): HTMLElement {
  const s = app.game.s; const value = companyValue(s); const avail = prestigeAvailable(s);
  const scen = h('select', null, ...SCENARIOS.map(sc => h('option', { value: sc.id, selected: sc.id === 'random' }, sc.name))) as HTMLSelectElement;
  const seed = h('input', { type: 'text', placeholder: 'Semilla (opcional)', style: { width: '160px' } }) as HTMLInputElement;
  return h('div', null,
    h('div', { class: 'grid3', style: { marginBottom: '14px' } }, kpi('Valor de compañía', money(value), `umbral de refundación ${money(BAL.prestigeThreshold)}`), kpi('Puntos de prestigio', `${avail} disponibles / ${s.prestige.points}`), kpi('Refundaciones', String(s.prestige.runs), `valor acumulado ${money(s.prestige.totalValue)}`)),
    meter(Math.min(1, value / BAL.prestigeThreshold), `${pct(Math.min(1, value / BAL.prestigeThreshold))} del umbral`),
    section('Refundar', h('div', { class: 'small muted' }, `Al refundar empiezas de cero en otra región con bonos permanentes. Puntos que ganarías ahora: ${refoundPoints(s)} (= ⌊log₂(valor / 2,5 M€)⌋). La colección, las licencias y la racha se conservan.`),
      h('div', { class: 'row' }, scen, seed, btn(canRefound(s) ? `Refundar (+${refoundPoints(s)} pts)` : `Necesitas ${money(BAL.prestigeThreshold)}`, () => { if (!canRefound(s)) return; if (!confirm('¿Refundar la compañía? Perderás la red y la caja actuales.')) return; app.game.refound(scen.value, seed.value || undefined); app.r.setWorld(); app.select(null); app.closeSheet(); app.refreshAll(); }, 'primary' + (canRefound(s) ? '' : ' ')))),
    section('Árbol permanente', h('div', { class: 'cards' }, ...PRESTIGE_TREE.map(p => { const lvl = s.prestige.spent[p.id] ?? 0; return h('div', { class: 'card' }, h('h4', null, p.name), h('div', { class: 'small muted' }, p.desc), h('div', { class: 'row', style: { justifyContent: 'space-between', marginTop: '6px' } }, h('span', { class: 'mono' }, '●'.repeat(lvl) + '○'.repeat(p.max - lvl)), btn('Mejorar', () => { if (prestigeSpend(s, p.id)) app.renderSheet(); else app.toast('Sin puntos disponibles o nivel máximo', 'warn'); }, 'sm' + (avail > 0 && lvl < p.max ? ' primary' : '')))); }))),
  );
}

// ───────────── Ajustes ─────────────
function settings(app: App): HTMLElement {
  const s = app.game.s; const g = app.game;
  const slots = listSlots();
  const applyScale = (v: number) => { s.settings.textScale = v; document.documentElement.style.setProperty('--scale', String(v)); setPref('textScale', v); };
  const modsList = h('div', { class: 'list' }, ...DATA.mods.map(m => h('div', { class: 'item small' }, h('span', { class: 'grow' }, `${m.name}${m.version ? ' v' + m.version : ''}${m.author ? ' · ' + m.author : ''}`), h('span', { class: 'muted' }, Object.entries(m.counts).map(([k, v]) => `${v} ${k}`).join(', ')))), !DATA.mods.length && h('div', { class: 'muted small' }, 'Ningún mod cargado. Formato en README.md; ejemplos en la carpeta mods/.'));
  return h('div', null,
    h('div', { class: 'grid2' },
      section('Interfaz',
        h('div', { class: 'row' }, 'Idioma', h('select', { onChange: (e: Event) => { const v = (e.target as HTMLSelectElement).value as 'es' | 'en'; s.settings.lang = v; setLang(v); setPref('lang', v); app.toast(v === 'en' ? 'Language set to English (partial translation). Reload to apply everywhere.' : 'Idioma: español', 'info'); } }, h('option', { value: 'es', selected: s.settings.lang === 'es' }, 'Español'), h('option', { value: 'en', selected: s.settings.lang === 'en' }, 'English (parcial)'))),
        h('div', { class: 'row' }, 'Tamaño de texto', h('input', { type: 'range', min: 0.85, max: 1.4, step: 0.05, value: s.settings.textScale, style: { width: '160px' }, onInput: (e: Event) => applyScale(Number((e.target as HTMLInputElement).value)) })),
        h('label', { class: 'row' }, h('input', { type: 'checkbox', checked: s.settings.colorblind, onChange: (e: Event) => { s.settings.colorblind = (e.target as HTMLInputElement).checked; setPref('colorblind', s.settings.colorblind); document.body.classList.toggle('cb-pattern', s.settings.colorblind); app.toast('Paleta de líneas para daltonismo ' + (s.settings.colorblind ? 'activada: elige colores en cada línea' : 'desactivada'), 'info'); } }), 'Paleta de líneas apta para daltonismo (Okabe-Ito)'),
      ),
      section('Atajos', h('div', { class: 'small', style: { columns: '2' } }, ...[['Espacio', 'Pausa'], ['1 / + / −', 'Velocidad'], ['Q', 'Seleccionar'], ['2', 'Estación'], ['3', 'Vía'], ['4', 'Línea'], ['5', 'Punto de paso'], ['X', 'Demoler'], ['Tab', 'Cambiar tipo de vía'], ['N', 'Nueva línea'], ['H', 'Mapa de demanda'], ['Z / Y', 'Deshacer / rehacer'], ['F D R V I C', 'Paneles'], ['P', 'Modo foto'], ['Ctrl+S', 'Guardar'], ['Esc / clic derecho', 'Cancelar']].map(([k, v]) => h('div', null, h('kbd', null, k), ' ', v))))),
    section('Partidas', h('div', { class: 'small muted' }, 'Autoguardado cada minuto en el navegador. Cinco ranuras manuales. Exporta a archivo para hacer copia o compartir.'),
      h('div', { class: 'list' }, ...slots.map((info, i) => h('div', { class: 'item' }, h('span', { class: 'grow' }, `Ranura ${i + 1}: `, info ? `${info.name} · ${money(info.value)} · ${new Date(info.savedAt).toLocaleString()}` : h('span', { class: 'muted' }, 'vacía')),
        btn('Guardar', () => { if (saveSlot(s, i, { value: companyValue(s) })) { app.toast(`Guardado en ranura ${i + 1}`, 'good'); app.renderSheet(); } else app.toast('No se pudo guardar (¿almacenamiento lleno?)', 'warn'); }, 'sm'),
        info && btn('Cargar', () => { const st = loadSlot(i); if (st) { g.load(st); g.onSessionStart(); app.r.setWorld(); app.select(null); app.closeSheet(); app.refreshAll(); } }, 'sm'),
        info && btn('Borrar', () => { deleteSlot(i); app.renderSheet(); }, 'sm danger')))),
      h('div', { class: 'row' }, btn('Exportar partida a archivo', () => download(`lineas-${s.world.name.replace(/\s+/g, '_')}.json`, serialize(s), 'application/json'), 'sm'), btn('Importar partida desde archivo', async () => { const txt = await pickFile('.json'); if (!txt) return; try { const st = deserialize(txt); g.load(st); g.onSessionStart(); app.r.setWorld(); app.select(null); app.closeSheet(); app.refreshAll(); app.toast('Partida importada', 'good'); } catch (e) { app.toast('Archivo no válido: ' + (e as Error).message, 'warn'); } }, 'sm'))),
    section('Mods (JSON)', modsList, h('div', { class: 'row' }, btn('Importar mod', async () => { const txt = await pickFile('.json'); if (!txt) return; try { const m = JSON.parse(txt); const err = validateMod(m); if (err) throw new Error(err); applyMod(m); const mods = JSON.parse(localStorage.getItem('lineas.v1.mods') ?? '[]'); mods.push(m); localStorage.setItem('lineas.v1.mods', JSON.stringify(mods)); app.toast(`Mod "${m.name}" cargado`, 'good'); app.renderSheet(); } catch (e) { app.toast('Mod no válido: ' + (e as Error).message, 'warn'); } }, 'sm'), btn('Quitar todos los mods', () => { localStorage.removeItem('lineas.v1.mods'); app.toast('Mods eliminados: recarga la página', 'info'); }, 'sm danger'))),
    section('Acerca de', h('div', { class: 'small muted' }, `LÍNEAS · semilla ${s.seed} · escenario ${s.scenario} · ${km(s.world.size * s.world.cell)} de lado · ${s.world.cities.length} ciudades · tiempo jugado ${num(s.stats.playSeconds / 60)} min`)),
  );
}
