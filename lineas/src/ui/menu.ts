import { h, clear, money, num, btn, pickFile, section } from './dom';
import { DATA } from '../sim/data';
import { listSlots, loadSlot, slotInfo, deserialize } from '../sim/save';
import type { GameState } from '../sim/types';
import { isoWeek, WEEKLY_YEARS } from '../sim/challenges';
import { t } from './i18n';

export interface MenuResult { kind: 'continue' | 'new' | 'load' | 'import' | 'weekly'; scenario?: string; seed?: string; state?: GameState }

/** Menú principal (modal). Resuelve con la acción elegida. */
export function showMenu(hasGame: boolean): Promise<MenuResult> {
  return new Promise(resolve => {
    const modal = h('div', { id: 'modal' }); const box = h('div', { class: 'box' }); modal.appendChild(box); document.body.appendChild(modal);
    const done = (r: MenuResult) => { modal.remove(); resolve(r); };
    let scenario = 'isla'; const seed = h('input', { type: 'text', placeholder: 'Semilla (opcional: número o texto)', style: { width: '280px' } }) as HTMLInputElement;
    const cards = h('div', { class: 'cards' });
    const renderCards = () => { clear(cards); for (const sc of DATA.scenarios) cards.append(h('div', { class: 'scen' + (sc.id === scenario ? ' on' : ''), onClick: () => { scenario = sc.id; renderCards(); } }, h('h4', null, sc.name), h('div', { class: 'small muted' }, sc.desc), h('div', { class: 'small', style: { marginTop: '4px' } }, `Caja inicial ${money(sc.startCash)}`))); };
    renderCards();
    const auto = slotInfo('auto'); const slots = listSlots();
    box.append(
      h('h1', null, t('app.title')), h('div', { class: 'muted', style: { marginBottom: '16px' } }, 'Tycoon de redes de transporte · 1900 → 2060 · sin servidor, todo en tu navegador'),
      h('div', { class: 'row', style: { marginBottom: '12px' } },
        hasGame && btn('Volver a la partida', () => done({ kind: 'continue' }), 'primary'),
        !hasGame && auto && btn(`Continuar (${auto.name}, ${money(auto.value)})`, () => { const st = loadSlot('auto'); if (st) done({ kind: 'load', state: st }); }, 'primary'),
        btn(`Reto semanal ${isoWeek()}`, () => done({ kind: 'weekly' }), ''),
        btn('Importar partida', async () => { const txt = await pickFile('.json'); if (!txt) return; try { done({ kind: 'import', state: deserialize(txt) }); } catch (e) { alert('Archivo no válido: ' + (e as Error).message); } }, ''),
      ),
      section('Nueva partida', cards, h('div', { class: 'row', style: { marginTop: '10px' } }, seed, btn('Empezar', () => done({ kind: 'new', scenario, seed: seed.value }), 'primary'))),
      h('div', null, slots.some(x => x) ? section('Cargar ranura', h('div', { class: 'list' }, ...slots.map((info, i) => info && h('div', { class: 'item' }, h('span', { class: 'grow' }, `Ranura ${i + 1}: ${info.name} · ${money(info.value)} · día ${num(info.day)} · ${new Date(info.savedAt).toLocaleString()}`), btn('Cargar', () => { const st = loadSlot(i); if (st) done({ kind: 'load', state: st }); }, 'sm'))))) : null),
      h('div', { class: 'small muted', style: { marginTop: '12px' } }, `El reto semanal usa la semilla de la semana ISO actual y mide el valor de la compañía a los ${WEEKLY_YEARS} años. Misma semilla = mismo mapa.`),
    );
  });
}

export function showOffline(o: { days: number; profit: number; completed: string[] }, onClose: () => void): void {
  const modal = h('div', { id: 'modal' });
  modal.append(h('div', { class: 'box', style: { width: 'min(520px, 94vw)' } },
    h('h2', null, 'Mientras no estabas'),
    h('p', null, `Han pasado ${num(o.days)} días de juego (${num(o.days / 30, 1)} meses).`),
    h('p', null, 'Tus líneas ganaron ', h('b', { class: o.profit >= 0 ? 'ok' : 'neg' }, money(o.profit)), o.profit <= 0 ? ' — sin beneficio reciente no hay progreso offline.' : '.'),
    o.completed.length ? h('div', null, h('b', null, 'Completado: '), o.completed.join(', ')) : null,
    h('div', { class: 'small muted', style: { marginTop: '8px' } }, 'El progreso offline se calcula con la media de beneficio diario de los últimos 30 días, al 70 % de eficiencia (mejorable con prestigio), hasta 8 horas.'),
    h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '12px' } }, btn('Continuar', () => { modal.remove(); onClose(); }, 'primary')),
  ));
  document.body.appendChild(modal);
}
