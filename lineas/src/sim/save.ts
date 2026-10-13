import type { GameState } from './types';
import { SAVE_VERSION } from './constants';

export const SLOTS = 5;
const KEY = 'lineas.v1.';

export interface SlotInfo { slot: number; name: string; day: number; cash: number; value: number; savedAt: number; scenario: string }

/** Serializa el estado (sin cachés). */
export function serialize(s: GameState): string {
  // Las rutas de línea se regeneran al cargar; no las guardamos (ahorra espacio).
  const lines = s.lines.map(l => ({ ...l, route: [], cum: [], stopDist: [], stopNodes: [] }));
  return JSON.stringify({ ...s, lines });
}

export function deserialize(json: string): GameState {
  const s = JSON.parse(json) as GameState;
  if (typeof s.version !== 'number' || !s.world || !Array.isArray(s.nodes)) throw new Error('Archivo de partida no válido');
  return migrate(s);
}

export function migrate(s: GameState): GameState {
  if (s.version < SAVE_VERSION) {
    // migraciones aditivas: rellena campos nuevos con valores por defecto
    s.dayRevenue ??= 0; s.dayPax ??= 0; s.dayCargo ??= 0; s.dayBuilt ??= 0; s.monthBreakdowns ??= 0; s.noBreakdownMonths ??= 0; s.monthPax ??= 0; s.prevCycle ??= 1;
    s.cur.offline ??= 0; for (const m of s.ledger) m.offline ??= 0;
    for (const v of s.vehicles) { v.cargoDest ??= -1; v.cargoKm ??= 0; }
    s.version = SAVE_VERSION;
  }
  return s;
}

function storage(): Storage | null { try { return typeof localStorage !== 'undefined' ? localStorage : null; } catch { return null; } }

export function saveSlot(s: GameState, slot: number | 'auto', meta: { value: number }): boolean {
  const st = storage(); if (!st) return false;
  try {
    st.setItem(KEY + 'slot.' + slot, serialize(s));
    const info: SlotInfo = { slot: slot === 'auto' ? -1 : slot, name: s.world.name, day: s.day, cash: s.cash, value: meta.value, savedAt: Date.now(), scenario: s.scenario };
    st.setItem(KEY + 'meta.' + slot, JSON.stringify(info));
    return true;
  } catch { return false; }
}
export function loadSlot(slot: number | 'auto'): GameState | null {
  const st = storage(); if (!st) return null;
  const raw = st.getItem(KEY + 'slot.' + slot); if (!raw) return null;
  try { return deserialize(raw); } catch { return null; }
}
export function slotInfo(slot: number | 'auto'): SlotInfo | null {
  const st = storage(); if (!st) return null;
  const raw = st.getItem(KEY + 'meta.' + slot); if (!raw) return null;
  try { return JSON.parse(raw) as SlotInfo; } catch { return null; }
}
export function deleteSlot(slot: number | 'auto'): void { const st = storage(); if (!st) return; st.removeItem(KEY + 'slot.' + slot); st.removeItem(KEY + 'meta.' + slot); }
export function listSlots(): (SlotInfo | null)[] { return Array.from({ length: SLOTS }, (_, i) => slotInfo(i)); }

export function getPref<T>(key: string, def: T): T { const st = storage(); if (!st) return def; try { const r = st.getItem(KEY + 'pref.' + key); return r ? JSON.parse(r) as T : def; } catch { return def; } }
export function setPref(key: string, v: unknown): void { const st = storage(); if (!st) return; try { st.setItem(KEY + 'pref.' + key, JSON.stringify(v)); } catch { /* sin espacio */ } }
