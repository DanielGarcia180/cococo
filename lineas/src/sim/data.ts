import vehiclesJson from './data/vehicles.json';
import techJson from './data/tech.json';
import erasJson from './data/eras.json';
import licensesJson from './data/licenses.json';
import { SCENARIOS, type Scenario } from './scenarios';
import type { TrackKind } from './types';

export type VehicleMode = 'bus' | 'tram' | 'train' | 'metro' | 'hsr' | 'truck' | 'freight';
export interface VehicleModel {
  id: string; name: string; mode: VehicleMode; track: TrackKind; cargo: boolean; era: number; year: number;
  price: number; speed: number; capacity: number; accel: number; consumption: number; reliability: number;
  maintenance: number; lifespan: number; unlock: string | null; desc: string; mod?: string;
}
export interface TechNode { id: string; name: string; era: number; cost: number; days: number; req: string[]; desc: string; year?: number; mod?: string }
export interface Era { index: number; year: number; name: string; fare: number; cargoFare: number; fuel: number }
export interface License { id: string; name: string; desc: string; metric: string; tiers: number[]; unit: string }

export interface ModFile {
  name: string; version?: string; author?: string;
  vehicles?: (Partial<VehicleModel> & Pick<VehicleModel, 'id' | 'name' | 'mode' | 'track' | 'price' | 'speed' | 'capacity'>)[];
  tech?: (Partial<TechNode> & Pick<TechNode, 'id' | 'name' | 'cost' | 'days'>)[]; eras?: Era[]; scenarios?: Scenario[];
}

/** Registro de datos. Se puede extender con mods (JSON). */
export const DATA = {
  vehicles: [...(vehiclesJson as VehicleModel[])],
  tech: [...(techJson as TechNode[])],
  eras: [...(erasJson as Era[])],
  licenses: licensesJson as License[],
  scenarios: [...SCENARIOS],
  mods: [] as { name: string; version?: string; author?: string; counts: Record<string, number> }[],
};

const BASE = { vehicles: DATA.vehicles.slice(), tech: DATA.tech.slice(), eras: DATA.eras.slice(), scenarios: DATA.scenarios.slice() };

export function resetData(): void {
  DATA.vehicles = BASE.vehicles.slice(); DATA.tech = BASE.tech.slice(); DATA.eras = BASE.eras.slice(); DATA.scenarios = BASE.scenarios.slice();
  DATA.mods = [];
}

export function validateMod(m: unknown): string | null {
  if (!m || typeof m !== 'object') return 'El mod debe ser un objeto JSON';
  const mod = m as ModFile;
  if (typeof mod.name !== 'string' || !mod.name) return 'Falta "name"';
  for (const v of mod.vehicles ?? []) {
    for (const k of ['id', 'name', 'mode', 'track', 'price', 'speed', 'capacity'] as const) if (v[k] === undefined) return `Vehículo sin "${k}"`;
    if (!['bus', 'tram', 'train', 'metro', 'hsr', 'truck', 'freight'].includes(v.mode)) return `Modo desconocido: ${v.mode}`;
    if (!['road', 'rail', 'metro', 'hsr'].includes(v.track)) return `Vía desconocida: ${v.track}`;
  }
  for (const t of mod.tech ?? []) for (const k of ['id', 'name', 'cost', 'days'] as const) if (t[k] === undefined) return `Investigación sin "${k}"`;
  for (const s of mod.scenarios ?? []) for (const k of ['id', 'name', 'world'] as const) if (s[k] === undefined) return `Escenario sin "${k}"`;
  return null;
}

/** Aplica un mod: sustituye por id si ya existe, si no añade. */
export function applyMod(mod: ModFile): void {
  const counts: Record<string, number> = {};
  const merge = <T extends { id: string; mod?: string }>(list: T[], items: T[] | undefined, key: string) => {
    if (!items) return;
    for (const it of items) {
      const copy = { ...it, mod: mod.name };
      const i = list.findIndex(x => x.id === it.id);
      if (i >= 0) list[i] = copy; else list.push(copy);
    }
    counts[key] = items.length;
  };
  merge(DATA.vehicles, mod.vehicles?.map(v => ({ accel: 0.8, consumption: 30, reliability: 0.9, maintenance: Math.round(v.price * 0.01), lifespan: 20, unlock: null, desc: '', cargo: false, era: 0, year: 1900, ...v } as VehicleModel)), 'vehicles');
  merge(DATA.tech, mod.tech?.map(t => ({ era: 0, req: [], desc: '', ...t } as TechNode)), 'tech');
  if (mod.eras) { for (const e of mod.eras) { const i = DATA.eras.findIndex(x => x.index === e.index); if (i >= 0) DATA.eras[i] = e; else DATA.eras.push(e); } DATA.eras.sort((a, b) => a.index - b.index); counts.eras = mod.eras.length; }
  merge(DATA.scenarios as (Scenario & { mod?: string })[], mod.scenarios, 'scenarios');
  DATA.mods = DATA.mods.filter(m => m.name !== mod.name);
  DATA.mods.push({ name: mod.name, version: mod.version, author: mod.author, counts });
}

export function model(id: string): VehicleModel | undefined { return DATA.vehicles.find(v => v.id === id); }
export function eraOf(index: number): Era { return DATA.eras[Math.min(index, DATA.eras.length - 1)]; }
export function tech(id: string): TechNode | undefined { return DATA.tech.find(t => t.id === id); }
