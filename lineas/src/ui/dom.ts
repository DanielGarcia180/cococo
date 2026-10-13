import { getLang } from './i18n';

type Child = Node | string | number | null | undefined | false | Child[];
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, unknown> | null = null, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (attrs) for (const k in attrs) {
    const v = attrs[k]; if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = String(v);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v as Record<string, string>);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v as EventListener);
    else if (k === 'html') el.innerHTML = String(v);
    else if (k in el && k !== 'list' && k !== 'form') (el as unknown as Record<string, unknown>)[k] = v;
    else el.setAttribute(k, String(v));
  }
  append(el, children);
  return el;
}
function append(el: HTMLElement, children: Child[]) {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
}
export function clear(el: HTMLElement): void { while (el.firstChild) el.removeChild(el.firstChild); }

const locale = () => getLang() === 'en' ? 'en-IE' : 'es-ES';
export function money(v: number, compact = true): string {
  const abs = Math.abs(v); const sign = v < 0 ? '−' : '';
  if (compact && abs >= 1e9) return `${sign}${(abs / 1e9).toLocaleString(locale(), { maximumFractionDigits: 2 })} mil M€`;
  if (compact && abs >= 1e6) return `${sign}${(abs / 1e6).toLocaleString(locale(), { maximumFractionDigits: 2 })} M€`;
  if (compact && abs >= 1e4) return `${sign}${(abs / 1e3).toLocaleString(locale(), { maximumFractionDigits: 1 })} k€`;
  return `${sign}${Math.round(abs).toLocaleString(locale())} €`;
}
export function num(v: number, digits = 0): string { return v.toLocaleString(locale(), { maximumFractionDigits: digits, minimumFractionDigits: digits }); }
export function pct(v: number, digits = 0): string { return `${(v * 100).toLocaleString(locale(), { maximumFractionDigits: digits })} %`; }
export function km(m: number): string { return m >= 1000 ? `${(m / 1000).toLocaleString(locale(), { maximumFractionDigits: 1 })} km` : `${Math.round(m)} m`; }
export function signed(v: number): string { return (v >= 0 ? '+' : '') + money(v); }

export function download(name: string, content: string | Blob, type = 'text/plain'): void {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
export function pickFile(accept: string): Promise<string | null> {
  return new Promise(res => {
    const i = document.createElement('input'); i.type = 'file'; i.accept = accept;
    i.onchange = () => { const f = i.files?.[0]; if (!f) return res(null); const r = new FileReader(); r.onload = () => res(String(r.result)); r.readAsText(f); };
    i.click();
  });
}
export function btn(label: string, onClick: () => void, cls = ''): HTMLButtonElement { return h('button', { class: 'btn ' + cls, onClick }, label); }
export function row(label: string, value: Child, cls = ''): HTMLElement { return h('div', { class: 'kv ' + cls }, h('span', { class: 'k' }, label), h('span', { class: 'v' }, value)); }
export function section(title: string, ...children: Child[]): HTMLElement { return h('section', { class: 'sec' }, h('h3', null, title), ...children); }
export function tip(el: HTMLElement, text: string): HTMLElement { el.title = text; el.setAttribute('data-tip', text); return el; }
