import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { BokehPass } from 'three/addons/postprocessing/BokehPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import type { Game } from '../sim/game';
import type { GameState, Node, Track, Line, Vehicle, Vec2, TrackKind } from '../sim/types';
import { heightAt, worldExtent } from '../sim/world';
import { posAlong } from '../sim/network';
import { model } from '../sim/data';
import { catchment } from '../sim/demand';
import { Rng, valueNoise2D } from '../sim/rng';

export const S = 1 / 100;          // metros → unidades de render
const HX = 2.2 * S;                // exageración vertical
export const LIVERIES = [0x2c2c2c, 0xf2f2f2, 0xb8352c, 0x1f5fa8, 0x2f8a4a, 0xd9a024, 0x6a3fb0, 0x19a0a6];

const KIND_COLOR: Record<TrackKind, number> = { road: 0x5a5a5a, rail: 0x3a3732, metro: 0x7a5aa8, hsr: 0x2b2b2b };
const KIND_WIDTH: Record<TrackKind, number> = { road: 0.32, rail: 0.2, metro: 0.22, hsr: 0.34 };
const MODE_SIZE: Record<string, [number, number, number]> = { bus: [0.55, 0.22, 0.24], tram: [0.8, 0.24, 0.24], train: [1.6, 0.3, 0.28], metro: [1.5, 0.28, 0.28], hsr: [2.4, 0.3, 0.3], truck: [0.6, 0.26, 0.26], freight: [1.8, 0.3, 0.3] };

export interface PhotoSettings { active: boolean; hour: number | null; aperture: number; focus: number; filter: 'none' | 'warm' | 'cool' | 'mono' | 'film'; vignette: number }

export class Renderer3D {
  renderer: THREE.WebGLRenderer; scene = new THREE.Scene(); camera: THREE.PerspectiveCamera; controls: MapControls;
  sun: THREE.DirectionalLight; hemi: THREE.HemisphereLight; fog: THREE.Fog;
  terrain!: THREE.Mesh; water!: THREE.Mesh;
  buildings!: THREE.InstancedMesh; industries!: THREE.InstancedMesh; trees!: THREE.InstancedMesh;
  tracksMesh: THREE.Mesh | null = null; linesMesh: THREE.Mesh | null = null; previewMesh: THREE.Mesh | null = null;
  stations!: THREE.InstancedMesh; waypoints!: THREE.InstancedMesh; stationRings!: THREE.InstancedMesh;
  vehicleMeshes = new Map<string, THREE.InstancedMesh>();
  heat: THREE.Mesh; heatTex: THREE.DataTexture; heatOn = false;
  selectRing: THREE.Mesh; hoverMarker: THREE.Mesh;
  weatherPts: THREE.Points; weatherMat: THREE.PointsMaterial;
  composer: EffectComposer; bokeh: BokehPass;
  photo: PhotoSettings = { active: false, hour: null, aperture: 0.00008, focus: 40, filter: 'none', vignette: 0 };
  game: Game; w: GameState['world']; nodeIndex: Node[] = []; lastPop = 0; cityHash = '';
  /** Reloj visual (horas): avanza a ritmo constante real, independiente de la velocidad de juego. */
  clock = 9; clockSpeed = 24 / 240;
  private tmpM = new THREE.Matrix4(); private tmpQ = new THREE.Quaternion(); private tmpV = new THREE.Vector3(); private tmpS = new THREE.Vector3(); private tmpC = new THREE.Color();
  vignetteEl: HTMLDivElement;

  constructor(canvas: HTMLCanvasElement, game: Game) {
    this.game = game; this.w = game.s.world;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true; this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1.0;
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.5, 3000);
    this.controls = new MapControls(this.camera, canvas);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.12; this.controls.minDistance = 4; this.controls.maxDistance = 900;
    this.controls.maxPolarAngle = Math.PI * 0.46; this.controls.screenSpacePanning = false; this.controls.zoomToCursor = true;
    this.fog = new THREE.Fog(0xdfe4ea, 200, 900); this.scene.fog = this.fog;
    this.hemi = new THREE.HemisphereLight(0xcfd9e6, 0x6b6a5a, 0.9); this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xffffff, 2.2); this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048); this.sun.shadow.bias = -0.0005; this.sun.shadow.normalBias = 0.02;
    const sc = this.sun.shadow.camera; sc.near = 1; sc.far = 1200; sc.left = -220; sc.right = 220; sc.top = 220; sc.bottom = -220;
    this.scene.add(this.sun); this.scene.add(this.sun.target);
    // heatmap
    const n = 128; this.heatTex = new THREE.DataTexture(new Uint8Array(n * n * 4), n, n, THREE.RGBAFormat); this.heatTex.needsUpdate = true;
    this.heatTex.magFilter = THREE.LinearFilter; this.heatTex.minFilter = THREE.LinearFilter; this.heatTex.colorSpace = THREE.SRGBColorSpace;
    this.heat = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: this.heatTex, transparent: true, depthWrite: false, opacity: 0.85, polygonOffset: true, polygonOffsetFactor: -3 }));
    this.heat.visible = false; this.scene.add(this.heat);
    this.selectRing = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.1, 40), new THREE.MeshBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthWrite: false }));
    this.selectRing.rotation.x = -Math.PI / 2; this.selectRing.visible = false; this.scene.add(this.selectRing);
    this.hoverMarker = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.65, 32), new THREE.MeshBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false }));
    this.hoverMarker.rotation.x = -Math.PI / 2; this.hoverMarker.visible = false; this.scene.add(this.hoverMarker);
    // clima: partículas
    const pg = new THREE.BufferGeometry(); const parr = new Float32Array(3000 * 3);
    for (let i = 0; i < 3000; i++) { parr[i * 3] = Math.random() * 80 - 40; parr[i * 3 + 1] = Math.random() * 40; parr[i * 3 + 2] = Math.random() * 80 - 40; }
    pg.setAttribute('position', new THREE.BufferAttribute(parr, 3));
    this.weatherMat = new THREE.PointsMaterial({ color: 0x9aa4b0, size: 0.12, transparent: true, opacity: 0.6, depthWrite: false });
    this.weatherPts = new THREE.Points(pg, this.weatherMat); this.weatherPts.visible = false; this.scene.add(this.weatherPts);
    // post
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bokeh = new BokehPass(this.scene, this.camera, { focus: 40, aperture: 0.00008, maxblur: 0.012 });
    this.composer.addPass(this.bokeh); this.composer.addPass(new OutputPass());
    this.vignetteEl = document.createElement('div'); this.vignetteEl.className = 'vignette'; document.body.appendChild(this.vignetteEl);
    this.buildWorld();
    this.resize();
  }

  // ───────────── mundo ─────────────
  setWorld(): void { this.w = this.game.s.world; this.disposeWorld(); this.buildWorld(); }
  private disposeWorld(): void {
    for (const m of [this.terrain, this.water, this.buildings, this.industries, this.trees, this.stations, this.waypoints, this.stationRings]) if (m) { this.scene.remove(m); m.geometry.dispose(); }
    if (this.tracksMesh) { this.scene.remove(this.tracksMesh); this.tracksMesh.geometry.dispose(); this.tracksMesh = null; }
    if (this.linesMesh) { this.scene.remove(this.linesMesh); this.linesMesh.geometry.dispose(); this.linesMesh = null; }
    for (const m of this.vehicleMeshes.values()) { this.scene.remove(m); m.geometry.dispose(); }
    this.vehicleMeshes.clear();
  }
  private buildWorld(): void {
    const w = this.w; const ext = worldExtent(w) * S;
    const geo = new THREE.PlaneGeometry(ext, ext, w.size - 1, w.size - 1); geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position as THREE.BufferAttribute; const colors = new Float32Array(pos.count * 3);
    const c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const gx = i % w.size, gy = Math.floor(i / w.size);
      const h = w.height[gy * w.size + gx];
      pos.setX(i, gx * w.cell * S); pos.setZ(i, gy * w.cell * S); pos.setY(i, Math.max(h, w.seaLevel - 6) * HX);
      // color por altura/pendiente
      const hx = w.height[gy * w.size + Math.min(w.size - 1, gx + 1)] - w.height[gy * w.size + Math.max(0, gx - 1)];
      const hy = w.height[Math.min(w.size - 1, gy + 1) * w.size + gx] - w.height[Math.max(0, gy - 1) * w.size + gx];
      const slope = Math.hypot(hx, hy) / (2 * w.cell);
      if (h <= w.seaLevel) c.setHex(0xa8a58a);                            // fondo/arena
      else if (h < 8) c.setHex(0xcfc7a3);                                 // playa
      else if (h < 180) c.setHex(0x9db07a).lerp(new THREE.Color(0x8aa06c), h / 180);
      else if (h < 380) c.setHex(0x8aa06c).lerp(new THREE.Color(0xa09e8a), (h - 180) / 200);
      else c.setHex(0xa09e8a).lerp(new THREE.Color(0xe6e6e2), Math.min(1, (h - 380) / 260));
      if (slope > 0.2) c.lerp(new THREE.Color(0x928d80), Math.min(1, (slope - 0.2) * 3));
      colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3)); geo.computeVertexNormals();
    this.terrain = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0, flatShading: false }));
    this.terrain.receiveShadow = true; this.terrain.castShadow = false; this.scene.add(this.terrain);
    this.water = new THREE.Mesh(new THREE.PlaneGeometry(ext * 1.6, ext * 1.6), new THREE.MeshStandardMaterial({ color: 0x86a7bb, roughness: 0.3, metalness: 0.05, transparent: true, opacity: 0.9 }));
    this.water.rotation.x = -Math.PI / 2; this.water.position.set(ext / 2, (w.seaLevel + 0.5) * HX, ext / 2); this.water.receiveShadow = true; this.scene.add(this.water);
    this.heat.geometry.dispose(); this.heat.geometry = geo; this.heat.position.set(0, 0.12, 0);
    this.buildCities(); this.buildIndustries(); this.buildTrees();
    // estaciones
    const sg = new THREE.BoxGeometry(1, 1, 1); sg.translate(0, 0.5, 0);
    this.stations = new THREE.InstancedMesh(sg, new THREE.MeshStandardMaterial({ roughness: 0.7 }), 1024); this.stations.castShadow = true; this.stations.count = 0; this.scene.add(this.stations);
    this.waypoints = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.12, 0.12, 0.3, 8), new THREE.MeshStandardMaterial({ color: 0x444444 }), 1024); this.waypoints.count = 0; this.scene.add(this.waypoints);
    this.stationRings = new THREE.InstancedMesh(new THREE.RingGeometry(0.96, 1, 48).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x222222, transparent: true, opacity: 0.25, depthWrite: false, side: THREE.DoubleSide }), 1024); this.stationRings.count = 0; this.scene.add(this.stationRings);
    for (const mode of Object.keys(MODE_SIZE)) {
      const [l, h, wd] = MODE_SIZE[mode];
      const g = new THREE.BoxGeometry(l, h, wd); g.translate(0, h / 2 + 0.03, 0);
      if (mode === 'hsr') { const p = g.attributes.position as THREE.BufferAttribute; for (let i = 0; i < p.count; i++) if (Math.abs(p.getX(i)) > l * 0.45) p.setY(i, p.getY(i) * 0.7 + 0.03); }
      const m = new THREE.InstancedMesh(g, new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.2 }), 2000);
      m.castShadow = true; m.count = 0; m.frustumCulled = false; this.scene.add(m); this.vehicleMeshes.set(mode, m);
    }
    const big = [...w.cities].sort((a, b) => b.pop - a.pop)[0];
    const cx = big ? big.x * S : ext / 2, cz = big ? big.y * S : ext / 2; const cy = big ? this.terrainY(big.x, big.y) : 0;
    this.controls.target.set(cx, cy, cz); this.camera.position.set(cx, cy + 34, cz + 42); this.controls.update();
    this.rebuildTracks(); this.rebuildStations();
  }

  private buildCities(): void {
    const w = this.w; const cities = w.cities;
    const rng = new Rng(w.seed ^ 0x3333);
    const list: { x: number; y: number; h: number; wd: number; d: number; rot: number; col: number }[] = [];
    for (const c of cities) {
      const n = Math.max(12, Math.min(320, Math.round(c.pop / 350)));
      for (let i = 0; i < n; i++) {
        const z = c.zones[Math.floor(rng.next() * c.zones.length)];
        const a = rng.next() * Math.PI * 2, r = Math.pow(rng.next(), 0.6) * (z.kind === 'com' ? 320 : 420);
        const x = z.x + Math.cos(a) * r, y = z.y + Math.sin(a) * r;
        const hgt = heightAt(w, x, y); if (hgt <= w.seaLevel + 1) continue;
        const tall = z.kind === 'com' ? rng.range(0.5, 1.6) * (0.4 + Math.min(2, c.pop / 60000)) : z.kind === 'ind' ? rng.range(0.25, 0.5) : rng.range(0.2, 0.55);
        list.push({ x, y, h: hgt, wd: rng.range(0.25, 0.6), d: rng.range(0.25, 0.6), rot: rng.next() * Math.PI, col: z.kind === 'com' ? 0xd9d6cf : z.kind === 'ind' ? 0xb9b4a8 : 0xe3ddd0 });
        list[list.length - 1].h = tall;
        (list[list.length - 1] as unknown as { gy: number }).gy = hgt;
      }
    }
    const g = new THREE.BoxGeometry(1, 1, 1); g.translate(0, 0.5, 0);
    this.buildings = new THREE.InstancedMesh(g, new THREE.MeshStandardMaterial({ roughness: 0.85, flatShading: true }), Math.max(1, list.length));
    this.buildings.castShadow = true; this.buildings.receiveShadow = true;
    list.forEach((b, i) => {
      const gy = (b as unknown as { gy: number }).gy;
      this.tmpM.compose(new THREE.Vector3(b.x * S, gy * HX, b.y * S), new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), b.rot), new THREE.Vector3(b.wd, b.h, b.d));
      this.buildings.setMatrixAt(i, this.tmpM); this.buildings.setColorAt(i, this.tmpC.setHex(b.col));
    });
    this.buildings.count = list.length; this.buildings.instanceMatrix.needsUpdate = true; if (this.buildings.instanceColor) this.buildings.instanceColor.needsUpdate = true;
    this.scene.add(this.buildings);
    this.lastPop = cities.reduce((a, c) => a + c.pop, 0);
  }
  private buildIndustries(): void {
    const w = this.w;
    const g = new THREE.BoxGeometry(1, 1, 1); g.translate(0, 0.5, 0);
    const colors: Record<string, number> = { mine: 0x6e6257, steel: 0x7a7066, factory: 0x8a8a8a, forest: 0x4f7a3f, sawmill: 0x9a7b55, farm: 0xc9b864, oilwell: 0x3f3a36, refinery: 0x8c8c94 };
    this.industries = new THREE.InstancedMesh(g, new THREE.MeshStandardMaterial({ roughness: 0.8, flatShading: true }), Math.max(1, w.industries.length * 3));
    this.industries.castShadow = true;
    let i = 0;
    for (const ind of w.industries) {
      const h = heightAt(w, ind.x, ind.y);
      const base = new THREE.Vector3(ind.x * S, h * HX, ind.y * S);
      const parts: [number, number, number, number, number, number][] = ind.kind === 'forest' ? [[0, 0, 0.9, 0.6, 0.9, 0], [1, 0.4, 0.5, 0.5, 0.5, 0], [-0.9, 0.6, 0.6, 0.45, 0.6, 0]] : ind.kind === 'farm' ? [[0, 0, 1.2, 0.3, 0.8, 0], [1.1, 0.5, 0.4, 0.5, 0.4, 0]] : [[0, 0, 1.4, 0.6, 1.0, 0], [0.9, 0.5, 0.25, 1.4, 0.25, 0], [-0.8, 0.4, 0.6, 0.4, 0.6, 0]];
      for (const [dx, dz, sx, sy, sz] of parts) {
        this.tmpM.compose(new THREE.Vector3(base.x + dx, base.y, base.z + dz), new THREE.Quaternion(), new THREE.Vector3(sx, sy, sz));
        this.industries.setMatrixAt(i, this.tmpM); this.industries.setColorAt(i, this.tmpC.setHex(colors[ind.kind] ?? 0x888888)); i++;
      }
    }
    this.industries.count = i; this.industries.instanceMatrix.needsUpdate = true; if (this.industries.instanceColor) this.industries.instanceColor.needsUpdate = true;
    this.scene.add(this.industries);
  }
  private buildTrees(): void {
    const w = this.w; const rng = new Rng(w.seed ^ 0x7777);
    const g = new THREE.ConeGeometry(0.22, 0.7, 5); g.translate(0, 0.35, 0);
    const n = Math.min(5000, Math.round(w.size * w.size / 4));
    this.trees = new THREE.InstancedMesh(g, new THREE.MeshStandardMaterial({ roughness: 0.9, flatShading: true }), n);
    this.trees.castShadow = true;
    const ext = worldExtent(w); let i = 0;
    const forest = valueNoise2D(w.seed ^ 0x1234);
    for (let t = 0; t < n * 4 && i < n; t++) {
      const x = rng.range(0, ext), y = rng.range(0, ext);
      const h = heightAt(w, x, y); if (h < 10 || h > 420) continue;
      if (forest(x / ext * 9, y / ext * 9) < 0.55) continue;
      if (w.cities.some(c => Math.hypot(c.x - x, c.y - y) < 900 + Math.sqrt(c.pop) * 5)) continue;
      const scale = rng.range(0.7, 1.4);
      this.tmpM.compose(new THREE.Vector3(x * S, h * HX, y * S), new THREE.Quaternion(), new THREE.Vector3(scale, scale, scale));
      this.trees.setMatrixAt(i, this.tmpM);
      this.trees.setColorAt(i, this.tmpC.setHex(0x4e7a44).offsetHSL(0, 0, rng.range(-0.06, 0.06))); i++;
    }
    this.trees.count = i; this.trees.instanceMatrix.needsUpdate = true; if (this.trees.instanceColor) this.trees.instanceColor.needsUpdate = true;
    this.scene.add(this.trees);
  }
  /** Reconstruye edificios si la población cambió notablemente. */
  checkCities(): void {
    const pop = this.w.cities.reduce((a, c) => a + c.pop, 0);
    if (Math.abs(pop - this.lastPop) / Math.max(1, this.lastPop) > 0.08) { this.scene.remove(this.buildings); this.buildings.geometry.dispose(); this.buildCities(); }
  }

  terrainY(x: number, y: number): number { return Math.max(this.w.seaLevel + 0.5, heightAt(this.w, x, y)) * HX; }

  // ───────────── vías y líneas ─────────────
  private ribbon(path: Vec2[], width: number, lift: number, color: number, alpha: number, bridgeLift = 0): { pos: number[]; col: number[]; idx: number[]; alphaArr: number[] } {
    const pos: number[] = [], col: number[] = [], idx: number[] = [], alphaArr: number[] = [];
    const c = new THREE.Color(color);
    const n = path.length; if (n < 2) return { pos, col, idx, alphaArr };
    for (let i = 0; i < n; i++) {
      const p = path[i], q = path[Math.min(n - 1, i + 1)], o = path[Math.max(0, i - 1)];
      const dx = q.x - o.x, dy = q.y - o.y; const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len * width / 2 / S, ny = dx / len * width / 2 / S;
      const h = heightAt(this.w, p.x, p.y);
      const water = h <= this.w.seaLevel;
      const y = (water ? (this.w.seaLevel + 0.5) * HX + bridgeLift : h * HX) + lift;
      pos.push((p.x + nx) * S, y, (p.y + ny) * S, (p.x - nx) * S, y, (p.y - ny) * S);
      col.push(c.r, c.g, c.b, c.r, c.g, c.b); alphaArr.push(alpha, alpha);
      if (i < n - 1) { const b = i * 2; idx.push(b, b + 2, b + 1, b + 1, b + 2, b + 3); }
    }
    return { pos, col, idx, alphaArr };
  }
  private mergeRibbons(parts: ReturnType<Renderer3D['ribbon']>[]): THREE.BufferGeometry {
    const pos: number[] = [], col: number[] = [], idx: number[] = []; let off = 0;
    for (const r of parts) { for (const i of r.idx) idx.push(i + off); pos.push(...r.pos); col.push(...r.col); off += r.pos.length / 3; }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3)); g.setIndex(idx);
    g.computeVertexNormals(); return g;
  }
  rebuildTracks(): void {
    const s = this.game.s;
    if (this.tracksMesh) { this.scene.remove(this.tracksMesh); this.tracksMesh.geometry.dispose(); }
    const parts = s.tracks.map(t => {
      const building = t.doneDay > s.day;
      const col = building ? 0xbdbdbd : KIND_COLOR[t.kind];
      return this.ribbon(t.path, KIND_WIDTH[t.kind], t.kind === 'metro' ? 0.03 : 0.045, col, 1, 0.5);
    });
    const g = this.mergeRibbons(parts);
    this.tracksMesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -1 }));
    this.tracksMesh.receiveShadow = true; this.scene.add(this.tracksMesh);
    this.rebuildLines();
  }
  rebuildLines(): void {
    const s = this.game.s;
    if (this.linesMesh) { this.scene.remove(this.linesMesh); this.linesMesh.geometry.dispose(); this.linesMesh = null; }
    const parts = s.lines.filter(l => l.valid).map((l, i) => this.ribbon(l.route, 0.1, 0.07 + i * 0.012, l.color, 1, 0.5));
    if (!parts.length) return;
    this.linesMesh = new THREE.Mesh(this.mergeRibbons(parts), new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2 }));
    this.scene.add(this.linesMesh);
  }
  setPreview(path: Vec2[] | null, kind: TrackKind, ok = true): void {
    if (this.previewMesh) { this.scene.remove(this.previewMesh); this.previewMesh.geometry.dispose(); this.previewMesh = null; }
    if (!path) return;
    const r = this.ribbon(path, KIND_WIDTH[kind] + 0.1, 0.1, ok ? 0x111111 : 0xb03030, 1, 0.5);
    this.previewMesh = new THREE.Mesh(this.mergeRibbons([r]), new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
    this.scene.add(this.previewMesh);
  }
  rebuildStations(): void {
    const s = this.game.s; let si = 0, wi = 0;
    this.nodeIndex = [];
    for (const n of s.nodes) {
      const y = this.terrainY(n.x, n.y);
      const building = n.doneDay > s.day;
      if (n.station) {
        const size = n.kind === 'road' ? 0.45 : n.kind === 'hsr' ? 1.3 : 0.8; const h = n.kind === 'metro' ? 0.25 : 0.35 + n.level * 0.08;
        this.tmpM.compose(this.tmpV.set(n.x * S, y, n.y * S), this.tmpQ.identity(), this.tmpS.set(size * (1 + (n.level - 1) * 0.25), h, size * 0.7));
        this.stations.setMatrixAt(si, this.tmpM);
        this.stations.setColorAt(si, this.tmpC.setHex(building ? 0xcfcfcf : n.kind === 'metro' ? 0x6f4fa0 : n.kind === 'road' ? 0x6d6d6d : n.kind === 'hsr' ? 0x202020 : 0x3b3833));
        const r = catchment(s, n) * S;
        this.tmpM.compose(this.tmpV.set(n.x * S, y + 0.06, n.y * S), this.tmpQ.identity(), this.tmpS.set(r, 1, r));
        this.stationRings.setMatrixAt(si, this.tmpM);
        this.nodeIndex[si] = n; si++;
      } else {
        this.tmpM.compose(this.tmpV.set(n.x * S, y + 0.1, n.y * S), this.tmpQ.identity(), this.tmpS.set(1, 1, 1));
        this.waypoints.setMatrixAt(wi, this.tmpM); wi++;
      }
    }
    this.stations.count = si; this.stations.instanceMatrix.needsUpdate = true; if (this.stations.instanceColor) this.stations.instanceColor.needsUpdate = true;
    this.stationRings.count = si; this.stationRings.instanceMatrix.needsUpdate = true;
    this.waypoints.count = wi; this.waypoints.instanceMatrix.needsUpdate = true;
  }

  // ───────────── vehículos ─────────────
  updateVehicles(): void {
    const s = this.game.s;
    const counts = new Map<string, number>(); for (const k of this.vehicleMeshes.keys()) counts.set(k, 0);
    for (const v of s.vehicles) {
      if (v.lineId === null || v.state === 'delivery' || v.state === 'idle') continue;
      const line = this.game.lineMap.get(v.lineId); if (!line || !line.valid) continue;
      const m = model(v.model); if (!m) continue;
      const mesh = this.vehicleMeshes.get(m.mode); if (!mesh) continue;
      const i = counts.get(m.mode)!; if (i >= 2000) continue;
      const p = posAlong(line, v.dist);
      const h = heightAt(this.w, p.x, p.y); const y = (h <= this.w.seaLevel ? (this.w.seaLevel + 0.5) * HX + 0.5 : h * HX) + 0.05;
      this.tmpQ.setFromAxisAngle(this.tmpV.set(0, 1, 0), -p.ang);
      this.tmpM.compose(new THREE.Vector3(p.x * S, y, p.y * S), this.tmpQ, this.tmpS.set(1, 1, 1));
      mesh.setMatrixAt(i, this.tmpM);
      const col = v.state === 'broken' ? 0x9a9a9a : v.livery > 0 ? LIVERIES[v.livery % LIVERIES.length] : line.color;
      mesh.setColorAt(i, this.tmpC.setHex(col));
      counts.set(m.mode, i + 1);
    }
    for (const [k, mesh] of this.vehicleMeshes) { mesh.count = counts.get(k)!; mesh.instanceMatrix.needsUpdate = true; if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true; }
  }
  vehicleAt(v: Vehicle): THREE.Vector3 | null {
    if (v.lineId === null) return null; const line = this.game.lineMap.get(v.lineId); if (!line || !line.valid) return null;
    const p = posAlong(line, v.dist); return new THREE.Vector3(p.x * S, this.terrainY(p.x, p.y), p.y * S);
  }

  // ───────────── heatmap ─────────────
  updateHeat(): void {
    if (!this.heatOn) { this.heat.visible = false; return; }
    this.heat.visible = true;
    const s = this.game.s; const w = this.w; const n = 128; const ext = worldExtent(w);
    const data = this.heatTex.image.data as Uint8Array;
    const stations = s.nodes.filter(x => x.station && x.doneDay <= s.day).map(x => ({ x: x.x, y: x.y, r: catchment(s, x) }));
    const zones: { x: number; y: number; w: number }[] = [];
    for (const c of w.cities) for (const z of c.zones) zones.push({ x: z.x, y: z.y, w: z.w * c.pop });
    const maxW = Math.max(1, ...zones.map(z => z.w));
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = (i + 0.5) / n * ext, y = (j + 0.5) / n * ext;
      let demand = 0;
      for (const z of zones) { const d = Math.hypot(z.x - x, z.y - y); if (d < 900) demand += z.w / maxW * (1 - d / 900); }
      let cov = 0;
      for (const st of stations) { const d = Math.hypot(st.x - x, st.y - y); if (d < st.r) cov = Math.max(cov, 1 - d / st.r * 0.6); }
      const unmet = Math.min(1, demand) * (1 - cov);
      const k = ((n - 1 - j) * n + i) * 4;
      // ámbar sobrio para demanda sin cubrir; gris para cubierta
      data[k] = Math.round(200 * unmet + 60 * cov * Math.min(1, demand)); data[k + 1] = Math.round(130 * unmet + 60 * cov * Math.min(1, demand)); data[k + 2] = Math.round(40 * unmet + 60 * cov * Math.min(1, demand));
      data[k + 3] = Math.round(Math.min(1, demand) * (unmet > 0.02 ? 150 + 90 * unmet : 70));
    }
    this.heatTex.needsUpdate = true;
  }
  toggleHeat(): boolean { this.heatOn = !this.heatOn; this.updateHeat(); return this.heatOn; }

  // ───────────── picking ─────────────
  /** Intersección rayo-terreno por marcha. Devuelve coordenadas de mundo (m). */
  pickGround(nx: number, ny: number): Vec2 | null {
    const ray = new THREE.Raycaster(); ray.setFromCamera(new THREE.Vector2(nx, ny), this.camera);
    const o = ray.ray.origin, d = ray.ray.direction;
    if (d.y >= 0) return null;
    let t = 0; const maxT = 2500; let step = 0.5;
    let prevT = 0;
    while (t < maxT) {
      const x = o.x + d.x * t, y = o.y + d.y * t, z = o.z + d.z * t;
      const h = this.terrainY(x / S, z / S);
      if (y <= h) {
        // refina
        let lo = prevT, hi = t;
        for (let k = 0; k < 8; k++) { const m = (lo + hi) / 2; const yy = o.y + d.y * m; const hh = this.terrainY((o.x + d.x * m) / S, (o.z + d.z * m) / S); if (yy <= hh) hi = m; else lo = m; }
        const fx = (o.x + d.x * hi) / S, fy = (o.z + d.z * hi) / S;
        const ext = worldExtent(this.w); if (fx < 0 || fy < 0 || fx > ext || fy > ext) return null;
        return { x: fx, y: fy };
      }
      prevT = t; step = Math.max(0.5, (y - h) * 0.5); t += step;
    }
    return null;
  }
  pickVehicle(nx: number, ny: number): Vehicle | null {
    const ray = new THREE.Raycaster(); ray.setFromCamera(new THREE.Vector2(nx, ny), this.camera);
    let best: Vehicle | null = null, bd = Infinity;
    const s = this.game.s;
    for (const v of s.vehicles) {
      const p = this.vehicleAt(v); if (!p) continue;
      const d = ray.ray.distanceToPoint(p);
      const dist = p.distanceTo(this.camera.position);
      if (d < 0.6 + dist * 0.004 && d < bd) { bd = d; best = v; }
    }
    return best;
  }
  focusOn(x: number, y: number): void {
    const target = new THREE.Vector3(x * S, this.terrainY(x, y), y * S);
    const off = this.camera.position.clone().sub(this.controls.target);
    this.controls.target.copy(target); this.camera.position.copy(target).add(off); this.controls.update();
  }
  setSelection(p: Vec2 | null, radius = 1): void {
    if (!p) { this.selectRing.visible = false; return; }
    this.selectRing.visible = true; this.selectRing.position.set(p.x * S, this.terrainY(p.x, p.y) + 0.08, p.y * S); this.selectRing.scale.set(radius, radius, 1);
  }
  setHover(p: Vec2 | null): void {
    if (!p) { this.hoverMarker.visible = false; return; }
    this.hoverMarker.visible = true; this.hoverMarker.position.set(p.x * S, this.terrainY(p.x, p.y) + 0.08, p.y * S);
  }

  // ───────────── día/noche y clima ─────────────
  updateLighting(): void {
    const s = this.game.s;
    const hour = this.photo.active && this.photo.hour !== null ? this.photo.hour : this.clock;
    const elev = Math.sin((hour - 6) / 12 * Math.PI);          // -1..1
    const az = (hour / 24) * Math.PI * 2;
    const day = Math.max(0, elev);
    const t = this.controls.target;
    this.sun.position.set(t.x + Math.cos(az) * 300, 40 + day * 400, t.z + Math.sin(az) * 300); this.sun.target.position.copy(t);
    const warm = new THREE.Color(0xffd9a8), white = new THREE.Color(0xffffff);
    this.sun.color.copy(warm).lerp(white, Math.min(1, day * 2.5));
    const weather = s.weather;
    const cloud = weather === 'cloud' ? 0.6 : weather === 'rain' ? 0.4 : weather === 'fog' ? 0.5 : weather === 'snow' ? 0.5 : 1;
    this.sun.intensity = (0.35 + 2.3 * Math.pow(day, 0.6)) * cloud;
    this.hemi.intensity = 0.7 + 0.6 * day * cloud;
    this.hemi.color.setHex(day > 0.1 ? 0xcfd9e6 : 0x8ea0bf);
    const skyDay = new THREE.Color(0xdfe6ee), skyDusk = new THREE.Color(0xe8c7a6), skyNight = new THREE.Color(0x343c4c);
    let sky: THREE.Color;
    if (elev > 0.25) sky = skyDay.clone().lerp(skyDusk, 0);
    else if (elev > -0.1) sky = skyDusk.clone().lerp(skyDay, Math.max(0, elev) / 0.25).lerp(skyNight, Math.max(0, -elev) / 0.1 * 0.8);
    else sky = skyNight;
    if (weather === 'fog') sky.lerp(new THREE.Color(0xc4c8cc), 0.6);
    if (weather === 'rain') sky.lerp(new THREE.Color(0x8e959c), 0.5);
    if (this.photo.active) { if (this.photo.filter === 'warm') sky.lerp(new THREE.Color(0xf0c8a0), 0.2); if (this.photo.filter === 'cool') sky.lerp(new THREE.Color(0xa0c0e0), 0.25); }
    this.scene.background = sky; this.fog.color.copy(sky);
    const dist = this.camera.position.distanceTo(t);
    const fogNear = weather === 'fog' ? 20 + dist * 0.5 : weather === 'rain' || weather === 'snow' ? 60 + dist : 160 + dist * 1.5;
    this.fog.near = fogNear; this.fog.far = fogNear + (weather === 'fog' ? 180 : 700);
    this.renderer.toneMappingExposure = this.photo.active && this.photo.filter === 'film' ? 0.9 : 1.0;
    // partículas de clima
    const show = (weather === 'rain' || weather === 'snow') && !this.photo.active;
    this.weatherPts.visible = show;
    if (show) {
      this.weatherPts.position.set(t.x, 0, t.z);
      this.weatherMat.size = weather === 'snow' ? 0.22 : 0.1; this.weatherMat.color.setHex(weather === 'snow' ? 0xf4f4f4 : 0x8f9aa6);
      const p = this.weatherPts.geometry.attributes.position as THREE.BufferAttribute; const fall = weather === 'snow' ? 0.08 : 0.5;
      for (let i = 0; i < p.count; i++) { let y = p.getY(i) - fall; if (y < 0) y = 40; p.setY(i, y); }
      p.needsUpdate = true;
    }
    this.vignetteEl.style.opacity = this.photo.active ? String(this.photo.vignette) : '0';
    const f = this.photo.active ? this.photo.filter : 'none';
    this.renderer.domElement.style.filter = f === 'mono' ? 'grayscale(1) contrast(1.08)' : f === 'warm' ? 'sepia(0.18) saturate(1.05)' : f === 'cool' ? 'saturate(0.9) hue-rotate(-8deg)' : f === 'film' ? 'contrast(1.12) saturate(0.85)' : '';
  }

  // ───────────── frame ─────────────
  resize(): void {
    const w = window.innerWidth, h = window.innerHeight;
    this.renderer.setSize(w, h, false); this.composer.setSize(w, h);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  tickClock(dt: number): void { if (!this.game.s.paused) this.clock = (this.clock + dt * this.clockSpeed) % 24; }
  render(): void {
    this.controls.update();
    this.updateLighting();
    this.updateVehicles();
    if (this.photo.active) {
      const u = this.bokeh.uniforms as unknown as Record<string, { value: number }>;
      u.focus.value = this.photo.focus; u.aperture.value = this.photo.aperture;
      this.composer.render();
    } else this.renderer.render(this.scene, this.camera);
  }
  /** Captura PNG en alta resolución (×scale). */
  capture(scale = 2): string {
    const w = window.innerWidth, h = window.innerHeight;
    const pr = this.renderer.getPixelRatio();
    this.renderer.setPixelRatio(pr * scale); this.composer.setPixelRatio(pr * scale); this.resize();
    this.render();
    const url = this.renderer.domElement.toDataURL('image/png');
    this.renderer.setPixelRatio(pr); this.composer.setPixelRatio(pr); this.resize();
    return url;
  }
}
