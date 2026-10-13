import './ui/styles.css';
import { Game } from './sim/game';
import { Renderer3D } from './render/scene';
import { App } from './ui/app';
import { showMenu, showOffline } from './ui/menu';
import { saveSlot, getPref } from './sim/save';
import { applyMod, validateMod } from './sim/data';
import { setLang } from './ui/i18n';
import { companyValue } from './sim/economy';

// mods guardados en el navegador
try { const mods = JSON.parse(localStorage.getItem('lineas.v1.mods') ?? '[]'); for (const m of mods) if (!validateMod(m)) applyMod(m); } catch { /* sin mods */ }
const lang = getPref<'es' | 'en'>('lang', 'es'); setLang(lang);
document.documentElement.style.setProperty('--scale', String(getPref('textScale', 1)));
if (getPref('colorblind', false)) document.body.classList.add('cb-pattern');

const canvas = document.getElementById('gl') as HTMLCanvasElement;
const game = new Game({ scenario: 'isla', seed: 7 });
const r = new Renderer3D(canvas, game);
const app = new App(game, r);
window.addEventListener('resize', () => r.resize());
let running = false;

function startFrom(result: Awaited<ReturnType<typeof showMenu>>): void {
  if (result.kind === 'continue') { running = true; return; }
  if (result.kind === 'new') game.newGame({ scenario: result.scenario!, seed: result.seed });
  else if (result.kind === 'weekly') game.newGame({ scenario: 'random', weekly: true });
  else if (result.state) game.load(result.state);
  game.s.settings.lang = lang; game.s.settings.textScale = getPref('textScale', 1); game.s.settings.colorblind = getPref('colorblind', false);
  r.setWorld(); app.select(null); app.closeSheet(); app.refreshAll();
  game.onSessionStart();
  running = true;
  if (game.s.offline && game.s.offline.days > 1) { running = false; showOffline(game.s.offline, () => { running = true; }); game.s.offline = null; }
}

app.onMenu = async () => { running = false; const res = await showMenu(true); startFrom(res); };
app.onSave = () => { if (saveSlot(game.s, 'auto', { value: companyValue(game.s) })) app.toast('Partida guardada (autoguardado)', 'good'); };

let last = performance.now(); let saveTimer = 0;
function frame(now: number): void {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  r.tickClock(dt);
  if (running) { game.advance(dt); app.update(dt); saveTimer += dt; if (saveTimer > 60) { saveTimer = 0; game.touchRealTime(); saveSlot(game.s, 'auto', { value: companyValue(game.s) }); } }
  r.render();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
window.addEventListener('beforeunload', () => { game.touchRealTime(); saveSlot(game.s, 'auto', { value: companyValue(game.s) }); });
document.addEventListener('visibilitychange', () => { if (document.hidden) { game.touchRealTime(); saveSlot(game.s, 'auto', { value: companyValue(game.s) }); } });

showMenu(false).then(startFrom);
// acceso para depuración y tests e2e
(window as unknown as { lineas: unknown }).lineas = { game, r, app };
