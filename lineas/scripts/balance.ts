/**
 * Bot de equilibrio: juega de forma sencilla durante N horas de tiempo real equivalente
 * (a velocidad media 3×) y registra la curva de progresión. Uso: npx tsx scripts/balance.ts [horas] [escenario] [semilla]
 */
import { Game } from '../src/sim/game';
import { companyValue } from '../src/sim/economy';
import { lineKpi } from '../src/sim/stats';
import { coverOf } from '../src/sim/demand';
import { PAX_UNIT } from '../src/sim/constants';
import type { Node } from '../src/sim/types';

const hours = Number(process.argv[2] ?? 10); const scenario = process.argv[3] ?? 'random'; const seed = process.argv[4] ?? '2026';
const SPEED = 3;                       // velocidad media de juego del bot
const g = new Game({ scenario, seed }); const s = g.s;
const log: string[] = [];
const fmt = (v: number) => Math.round(v).toLocaleString('en-US');

function candidates(): { x: number; y: number; score: number; cityId: number }[] {
  const out: { x: number; y: number; score: number; cityId: number }[] = [];
  for (const c of s.world.cities) for (const z of c.zones) {
    if (s.nodes.some(n => Math.hypot(n.x - z.x, n.y - z.y) < 900)) continue;
    const fake = { x: z.x, y: z.y, kind: 'rail', level: 1, station: true } as Node;
    const cov = coverOf(s, fake); out.push({ x: z.x, y: z.y, score: cov.P + cov.A, cityId: c.id });
  }
  return out.sort((a, b) => b.score - a.score);
}

/** Estrategia: si hay caja, (1) añade vehículos a líneas saturadas, (2) abre una nueva estación+línea cerca de la red, (3) investiga, (4) invierte el excedente. */
function decide(): void {
  const cash = s.cash; const models = g.availableModels().filter(m => !m.cargo);
  // 1. vehículos para líneas con factor de carga alto
  for (const l of s.lines) {
    if (l.cargoKind || !l.valid) continue;
    const k = lineKpi(s, l); const load = l.mLoadN ? l.mLoadSum / l.mLoadN : k.load;
    const m = models.filter(x => g.lineAccepts(l, x)).sort((a, b) => b.capacity / b.price - a.capacity / a.price)[0];
    if (!m) continue;
    if (load > 0.8 && l.vehicles.length < 12 && cash > m.price * 2) { const v = g.buyVehicle(m.id, l.id); if (v) g.rush('vehicle', v.id); return; }
    if (load < 0.25 && l.vehicles.length > 1 && l.history.length > 2) { g.sellVehicle(l.vehicles[l.vehicles.length - 1]); return; }
  }
  // 2. nueva línea: dos mejores zonas no cubiertas de una ciudad (o conexión entre ciudades)
  const cand = candidates();
  if (cand.length >= 2) {
    const kind = (s.researched.includes('hsr') && cash > 20_000_000) ? 'hsr' : 'rail';
    const a = cand[0]; const near = cand.find(c => Math.hypot(c.x - a.x, c.y - a.y) > 1200 && Math.hypot(c.x - a.x, c.y - a.y) < 4000);
    const b = near ?? cand.find(c => Math.hypot(c.x - a.x, c.y - a.y) > 1200 && Math.hypot(c.x - a.x, c.y - a.y) < (kind === 'hsr' ? 60000 : 25000));
    if (b) {
      const est = Math.hypot(a.x - b.x, a.y - b.y) * (kind === 'hsr' ? 60 : 16) + 2 * g.stationCost(kind, true) + 30000;
      if (cash > est * 1.3) {
        const na = g.placeNode(a.x, a.y, kind); const nb = na && g.placeNode(b.x, b.y, kind);
        if (na && nb) {
          const t = g.buildTrack(na.id, nb.id, kind);
          if (t) {
            const l = g.createLine(kind); g.lineAddStop(l.id, na.id); g.lineAddStop(l.id, nb.id);
            const m = models.filter(x => g.lineAccepts(l, x)).sort((a, b) => b.capacity / b.price - a.capacity / a.price)[0];
            if (m) g.buyVehicle(m.id, l.id);
            return;
          }
          g.removeNode(nb.id); g.removeNode(na.id);
        } else if (na) g.removeNode(na.id);
      }
    }
  }
  // 3. investigación
  if (!s.research) { const av = g.researchAvailable().filter(a => a.ok); const pick = av.sort((a, b) => (a.id.startsWith('era_') ? -1 : 1) - (b.id.startsWith('era_') ? -1 : 1))[0]; if (pick && cash > 200000) g.startResearch(pick.id); }
  // 4. fondo: excedente por encima de 500 k
  if (cash > 800000 && s.fund.dca === 0) g.setDca(Math.round(cash * 0.02));
}

const totalSeconds = hours * 3600;
let simSeconds = 0; let nextLog = 0; let nextDecide = 0;
const t0 = Date.now();
log.push('hora_real;año;mes;caja;valor;beneficio_mes;lineas;vehiculos;estaciones;pax_mes;era;rep;poblacion');
while (simSeconds < totalSeconds * SPEED) {
  g.advance(0.25 * 4); simSeconds += 1;            // 1 s de simulación por iteración
  if (simSeconds >= nextDecide) { nextDecide += 20; decide(); }
  if (simSeconds >= nextLog) {
    nextLog += 1800;                                 // cada 10 min reales (0,5 h de juego a 3×)
    const last = s.ledger[s.ledger.length - 1];
    log.push([(simSeconds / SPEED / 3600).toFixed(2), g.year(), g.month() + 1, fmt(s.cash), fmt(companyValue(s)), last ? fmt(last.profit) : '0', s.lines.length, s.vehicles.length, s.nodes.filter(n => n.station).length, last ? fmt(last.pax * PAX_UNIT) : 0, s.era, Math.round(s.reputation), fmt(s.world.cities.reduce((a, c) => a + c.pop, 0))].join(';'));
  }
}
console.log(log.join('\n'));
console.error(`simulado ${hours} h (${(Date.now() - t0) / 1000}s reales) · año ${g.year()} · valor ${fmt(companyValue(s))} · líneas ${s.lines.length} · vehículos ${s.vehicles.length} · investigado ${s.researched.join(',')}`);
