/**
 * Game rules, AI prompting and scene description. A direct port of
 * macos/Sources/SquirrelFinder/Game.swift: same constants, same prompt text, same scoring.
 * The renderer (render.ts) only consumes `buildScene()`, `lights()` and `shake`.
 */
import { clamp, dist, length, normalize, rand, type V2, v2 } from './math';
import { LEVELS, type LevelSpec } from './levels';
import { GRID_N, loadRoomMesh, type RoomMesh, VOXEL } from './roomMesh';
import { PROP_RECTS, PROP_ATLAS_SIZE } from './propAtlas';
import { BROWSER_SPECS, backendFor, ollamaModelIds, specFor } from './models/backends';
import type { ModelSpec, OptionSpec } from './models/types';

/** An entry in the player picker. */
export interface ModelOption {
  id: string; // 'laya-web' | 'julia-1' | 'kev-0.6b-bu' | 'decision-lab' | 'ollama:<tag>'
  label: string;
  group: 'browser' | 'ollama';
  spec: ModelSpec;
}

/** Accepts old-style names too: '@laya-browser' and bare Ollama tags such as 'nimble:9b-nvfp4'. */
export function normalizePlayer(name: string): string {
  if (!name) return '';
  if (name === '@laya-browser') return 'laya-web';
  if (BROWSER_SPECS.some((m) => m.id === name) || name.startsWith('ollama:')) return name;
  return `ollama:${name}`;
}

export type GameState = 'playing' | 'won' | 'lost';
export type TexName = 'wizardWalk' | 'wizardIdle' | 'ghostWalk' | 'penguin' | 'props' | 'cracks';

/** Camera for the voxel rooms: orthographic, tilted, looking at the floor centre. */
export const STAGE = {
  pitch: (58 * Math.PI) / 180,
  scale: 140, // game units per room unit (voxel = 14)
  centerX: 640,
  centerY: 481, // plane y of the floor centre
  clipUnits: 2.2, // room units of wall kept above the floor
  sinP: Math.sin((58 * Math.PI) / 180),
  cosP: Math.cos((58 * Math.PI) / 180),
};
export const WORLD = { x: 1280, y: 800 }; // plane size (the flat graveyard plays on all of it)

/** One thing to draw this frame. See render.ts for how `floor`/`lift` become a 3D position. */
export interface SpriteDraw {
  tex: TexName | null; // null = procedural (shadow)
  kind: 0 | 1; // 0 textured sprite, 1 soft shadow
  uv: [number, number, number, number]; // sub-rect of the texture, normalised, y from the top
  smooth: boolean; // linear sampling instead of pixel-art nearest
  lit: boolean; // false = emissive
  floor: V2; // room: floor game coords; flat graveyard: plane coords of the feet
  lift: number; // plane units from the feet up to the sprite centre, along screen-up
  gpos: V2; // position used for lighting (same space as `floor`)
  size: V2; // plane units; negative x mirrors
  rot: number;
  tint: [number, number, number, number];
  flash: number;
  decal: boolean; // lies on the floor instead of facing the camera
}

export interface ParticleDraw {
  x: number; // plane coordinates
  y: number;
  size: number;
  color: [number, number, number];
  alpha: number;
}

/** One decision point captured from a (headless) run of the real game, with ground truth from the oracle. */
export interface BenchScenario {
  id: string;
  level: number;
  levelName: string;
  room: boolean;
  tick: number;
  category: 'goal' | 'threat' | 'evade' | 'calm';
  tight: boolean; // a wall / obstacle is close in some cardinal direction
  hasPenguin: boolean;
  nearestGhost: number | null; // game units
  ghostsNear: number;
  prompts: {
    compact: { state: string; instructions: string };
    full: { state: string; instructions: string };
  };
  oracle: {
    best: string[];
    acceptable: string[];
    fatal: string[];
    blocked: string[];
  };
}

export interface LightRec {
  x: number; // game floor coords (room) / plane coords (flat)
  z: number;
  radius: number;
  intensity: number;
  color: [number, number, number];
  height: number; // above the floor
}

interface Ghost {
  p: V2;
  v: V2;
  age: number;
  phase: number;
  near: boolean;
}

interface Particle {
  p: V2; // plane coordinates
  v: V2;
  life: number;
  maxLife: number;
  size: number;
  color: [number, number, number];
  gravity: number;
  grow: number;
}

interface Glow {
  color: [number, number, number];
  radius: number;
  intensity: number;
  lift: number;
}
interface Prop {
  name: string;
  p: V2;
  scale?: number;
  flat?: boolean;
  alpha?: number;
  glow?: Glow;
}

const TORCH: Glow = { color: [1.0, 0.68, 0.3], radius: 330, intensity: 0.85, lift: 0.8 };
const MUSHROOM: Glow = { color: [0.25, 1.0, 0.6], radius: 190, intensity: 0.65, lift: 0.5 };
const PUMPKIN: Glow = { color: [1.0, 0.5, 0.15], radius: 215, intensity: 0.7, lift: 0.5 };
const CAULDRON: Glow = { color: [0.4, 1.0, 0.3], radius: 290, intensity: 0.8, lift: 0.6 };

/** Graveyard scenery (art drawn at 3 units per pixel unless `scale` says otherwise). `p` is the sprite's base centre. */
const PROPS: Prop[] = [
  { name: 'web', p: v2(120, 150), flat: true, alpha: 0.75 },
  { name: 'web', p: v2(1215, 150), flat: true, alpha: 0.75 },
  { name: 'blood', p: v2(800, 300), flat: true },
  { name: 'blood', p: v2(360, 400), flat: true },
  { name: 'skull', p: v2(760, 610), scale: 3.2, flat: true },
  { name: 'bones', p: v2(540, 545), scale: 3.2, flat: true },
  { name: 'crossBig', p: v2(150, 255) },
  { name: 'crossBlood', p: v2(1130, 205) },
  { name: 'treeDead', p: v2(95, 700) },
  { name: 'treeGreen', p: v2(1195, 500) },
  { name: 'tombstone', p: v2(330, 630) },
  { name: 'tombstone', p: v2(405, 665) },
  { name: 'tombstone', p: v2(930, 700) },
  { name: 'tombstone', p: v2(870, 170) },
  { name: 'crossSmall', p: v2(500, 150) },
  { name: 'crossSmall', p: v2(1085, 650) },
  { name: 'crossBroken', p: v2(245, 580) },
  { name: 'bushBig', p: v2(720, 120) },
  { name: 'bushBig', p: v2(1235, 340) },
  { name: 'bushSmall', p: v2(60, 520) },
  { name: 'bushSmall', p: v2(770, 715) },
  { name: 'stump', p: v2(430, 110) },
  { name: 'cauldron', p: v2(985, 135), glow: CAULDRON },
  { name: 'torch', p: v2(120, 440), scale: 5, glow: TORCH },
  { name: 'torch', p: v2(1170, 395), scale: 5, glow: TORCH },
  { name: 'torch', p: v2(640, 765), scale: 5, glow: TORCH },
  { name: 'torch', p: v2(580, 135), scale: 5, glow: TORCH },
  { name: 'mushroomA', p: v2(195, 745), scale: 3.5, glow: MUSHROOM },
  { name: 'mushroomB', p: v2(1110, 575), scale: 3.5, glow: MUSHROOM },
  { name: 'mushroomC', p: v2(870, 770), scale: 3.5, glow: MUSHROOM },
  { name: 'pumpkin', p: v2(455, 725), scale: 3.5, glow: PUMPKIN },
  { name: 'pumpkin', p: v2(1055, 700), scale: 3.5, glow: PUMPKIN },
];

const DIRS: { key: string; label: string; vec: V2; desc: string }[] = [
  { key: 'up', label: '↑', vec: v2(0, -1), desc: 'Walk straight up' },
  { key: 'up_right', label: '↗', vec: v2(1, -1), desc: 'Walk diagonally up and to the right' },
  { key: 'right', label: '→', vec: v2(1, 0), desc: 'Walk straight to the right' },
  { key: 'down_right', label: '↘', vec: v2(1, 1), desc: 'Walk diagonally down and to the right' },
  { key: 'down', label: '↓', vec: v2(0, 1), desc: 'Walk straight down' },
  { key: 'down_left', label: '↙', vec: v2(-1, 1), desc: 'Walk diagonally down and to the left' },
  { key: 'left', label: '←', vec: v2(-1, 0), desc: 'Walk straight to the left' },
  { key: 'up_left', label: '↖', vec: v2(-1, -1), desc: 'Walk diagonally up and to the left' },
];

const roundAway = (x: number) => Math.sign(x) * Math.round(Math.abs(x));
const easeOutBack = (t: number) => {
  const x = clamp(t, 0, 1);
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
};
const pick = <T,>(a: T[]): T => a[Math.floor(Math.random() * a.length)];
const store = {
  getNum(key: string): number | null {
    try {
      const v = localStorage.getItem(key);
      return v === null ? null : Number(v);
    } catch { return null; }
  },
  set(key: string, v: number) {
    try { localStorage.setItem(key, String(v)); } catch { /* private mode */ }
  },
};

export class Game {
  // MARK: HUD-facing state (read every frame by hud.ts)
  state: GameState = 'playing';
  shownTime = 0;
  score = 0;
  bonus = 0;
  bestScore = store.getNum('sf_best_score') ?? 0;
  bestTime: number | null = store.getNum('sf_best');
  levelIndex = 0;
  models: ModelOption[] = [];
  player = ''; // '' = a human; otherwise an Ollama model name
  aiStatus = '';
  aiThinking = false;
  aiProbs: { id: string; label: string; p: number }[] = [];
  aiMoves = 0;
  aiError: string | null = null;
  loading = true;
  paused = false; // the intro screen freezes the world
  aiLoading = false; // a model is downloading / starting: the world is paused
  /** bumps whenever the stage (room / level) changes so the renderer can rebuild its scene */
  stageVersion = 0;

  // MARK: stage
  spec: LevelSpec = LEVELS[0];
  room: RoomMesh | null = null;
  private meshCache = new Map<string, RoomMesh>();
  private walk: Uint8Array = new Uint8Array(0);
  private spawnCells: V2[] = [];
  private levelGlows: LightRec[] = [];
  private levelStartScore = 0;
  private loops = 0;
  private startG = v2(0, 0);

  // MARK: simulation
  wizard = v2(WORLD.x / 2, WORLD.y / 2);
  private ghosts: Ghost[] = [];
  private penguin: V2 | null = null;
  private penguinBorn = 0;
  private particles: Particle[] = [];
  private elapsed = 0;
  private scoreF = 0;
  private clock = 0;
  private endClock = 0;
  private penguinAt = 2;
  private spawnTimer = 1.5;
  private faceRow = 0;
  private walkPhase = 0;
  private moving = false;
  private dustTimer = 0;
  private lastShown = -1;
  private trauma = 0;
  shake = v2(0, 0); // plane units
  private keys = new Set<string>();
  private forcedDir: V2 | null = null;

  // MARK: AI
  private aiDir = v2(0, 0);
  private moveLeft = 0;
  private pending = false;
  private epoch = 0;
  private readonly moveTime = 0.35;

  // constants
  private readonly marginX = 70;
  private readonly marginTop = 100;
  private readonly marginBottom = 70;
  private readonly ghostArm = 0.9;
  private readonly pointsPerSecond = 10;
  private readonly nearMissPoints = 25;
  private readonly goalPoints = 500;

  constructor(private assetBase: string) {
    window.addEventListener('keydown', (e) => this.onKey(e, true));
    window.addEventListener('keyup', (e) => this.onKey(e, false));
  }

  get levelCount() { return LEVELS.length; }
  get isLastLevel() { return this.levelIndex >= LEVELS.length - 1; }
  get isRoom() { return this.room !== null; }
  get levelName() { return this.spec.name; }
  get levelTagline() { return this.spec.tagline; }

  private get cellG() { return VOXEL * STAGE.scale; }
  private get spriteSize() { return this.isRoom ? 128 : 192; }
  private get rs() { return this.spriteSize / 192; }
  private get speedScale() { return this.isRoom ? 0.55 : 1; }
  private get wizardSpeed() { return 300 * this.speedScale; }
  private get hitGhost() { return 58 * this.rs; }
  private get hitGoal() { return 64 * this.rs; }
  private get nearMissRadius() { return 105 * this.rs; }

  // MARK: lifecycle
  async start(level: number, player: string) {
    this.player = normalizePlayer(player);
    this.levelStartScore = 0;
    this.loops = 0;
    this.models = BROWSER_SPECS.map((m) => ({ id: m.id, label: m.label, group: 'browser' as const, spec: m }));
    await this.loadLevel(Math.max(0, Math.min(LEVELS.length - 1, level)));
    void this.loadModels();
    if (this.player) void this.ensureModel(this.player);
  }

  /** Picker contents: the in-browser models, then whatever the local Ollama server has pulled. */
  async loadModels() {
    const browser: ModelOption[] = BROWSER_SPECS.map((m) => ({ id: m.id, label: m.label, group: 'browser', spec: m }));
    try {
      const ids = await ollamaModelIds();
      const local: ModelOption[] = ids.map((id) => {
        const spec = specFor(id)!;
        return { id, label: spec.label, group: 'ollama', spec };
      });
      this.models = [...browser, ...local];
    } catch (e) {
      // no Ollama running: the in-browser models still work
      this.models = browser;
      if (this.player.startsWith('ollama:')) this.aiStatus = (e as Error).message;
    }
  }

  /** Start a fresh run from level 1 with the given player ('' = human), even if it is the same player. */
  async newRun(name: string) {
    this.player = normalizePlayer(name);
    this.levelStartScore = 0;
    this.loops = 0;
    await this.loadLevel(0);
    if (this.player) void this.ensureModel(this.player);
  }

  /** Switch who plays (human / model) and start over from level 1. */
  async setPlayer(name: string) {
    const id = normalizePlayer(name);
    if (id === this.player) return;
    this.player = id;
    this.levelStartScore = 0;
    this.loops = 0;
    await this.loadLevel(0);
    if (id) void this.ensureModel(id);
  }

  /** Download + start the chosen model (first use only); the world is paused meanwhile. */
  private async ensureModel(id: string) {
    const backend = backendFor(id);
    if (backend.ready) {
      this.aiStatus = `${backend.spec.label} is ready · ${backend.runtime}`;
      return;
    }
    this.aiLoading = true;
    this.aiError = null;
    try {
      await backend.load((p) => {
        if (this.player === id) this.aiStatus = `${p.phase}… ${Math.round(p.fraction * 100)}%`;
      });
      if (this.player === id) this.aiStatus = `${backend.spec.label} is ready · ${backend.runtime}`;
    } catch (e) {
      this.aiError = `${backend.spec.label} failed to load: ${(e as Error).message}`;
      this.aiStatus = this.aiError;
    } finally {
      this.aiLoading = false;
    }
  }

  private async loadLevel(index: number) {
    this.loading = true;
    this.epoch++;
    this.levelIndex = index;
    this.spec = LEVELS[index];
    this.room = null;
    this.walk = new Uint8Array(0);
    this.spawnCells = [];
    this.levelGlows = [];
    const slug = this.spec.room;
    if (slug) {
      let mesh = this.meshCache.get(slug);
      if (!mesh) {
        try {
          mesh = await loadRoomMesh(`${this.assetBase}rooms/${slug}.obj`, `${this.assetBase}rooms/${slug}.png`);
          this.meshCache.set(slug, mesh);
        } catch (e) {
          console.error(`level ${index + 1} (${slug}): mesh failed to load`, e);
        }
      }
      if (mesh) {
        this.room = mesh;
        this.prepareGrid(mesh);
      }
    }
    this.stageVersion++;
    this.reset();
    this.loading = false;
  }

  private prepareGrid(m: RoomMesh) {
    const n = GRID_N;
    // a cell is walkable when it and its 8 neighbours are free (keeps the wizard off the walls)
    const free = new Uint8Array(n * n);
    for (let z = 1; z < n - 1; z++) {
      for (let x = 1; x < n - 1; x++) {
        let ok = true;
        for (let dz = -1; dz <= 1 && ok; dz++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (m.blocked[(z + dz) * n + (x + dx)]) { ok = false; break; }
          }
        }
        free[z * n + x] = ok ? 1 : 0;
      }
    }
    // keep only the largest connected region so nothing spawns in sealed pockets
    const label = new Int32Array(n * n).fill(-1);
    const sizes: number[] = [];
    for (let start = 0; start < n * n; start++) {
      if (!free[start] || label[start] >= 0) continue;
      const id = sizes.length;
      const stack = [start];
      label[start] = id;
      let count = 0;
      while (stack.length) {
        const c = stack.pop()!;
        count++;
        const cx = c % n;
        const cz = (c / n) | 0;
        for (const [nx, nz] of [[cx + 1, cz], [cx - 1, cz], [cx, cz + 1], [cx, cz - 1]]) {
          if (nx < 0 || nz < 0 || nx >= n || nz >= n) continue;
          const ni = nz * n + nx;
          if (free[ni] && label[ni] < 0) { label[ni] = id; stack.push(ni); }
        }
      }
      sizes.push(count);
    }
    let best = -1;
    let bestSize = -1;
    sizes.forEach((s, i) => { if (s > bestSize) { bestSize = s; best = i; } });
    this.walk = new Uint8Array(n * n);
    this.spawnCells = [];
    for (let i = 0; i < n * n; i++) {
      if (best >= 0 && label[i] === best) {
        this.walk[i] = 1;
        const cx = (i % n) + 0.5 - n / 2;
        const cz = Math.floor(i / n) + 0.5 - n / 2;
        this.spawnCells.push(v2(cx * this.cellG, cz * this.cellG));
      }
    }
    this.startG = this.spawnCells.reduce((a, b) => (length(b) < length(a) ? b : a), this.spawnCells[0] ?? v2(0, 0));

    const S = STAGE.scale;
    this.levelGlows = m.glows.map((g) => ({
      x: g.pos[0] * S, z: g.pos[2] * S, radius: 300, intensity: 0.95,
      color: g.color, height: Math.max(10, (g.pos[1] - m.floorY) * S),
    }));
  }

  /** Restart the current level (retry after a loss, or space). */
  reset() {
    this.epoch++;
    this.pending = false;
    this.moveLeft = 0;
    this.aiDir = v2(0, 0);
    this.aiError = null;
    this.aiThinking = false;
    this.aiProbs = [];
    this.aiMoves = 0;
    this.aiStatus = this.player ? `${this.player} is ready` : '';

    this.wizard = this.isRoom ? { ...this.startG } : v2(WORLD.x / 2, WORLD.y / 2);
    this.penguin = null;
    this.particles = [];
    this.elapsed = 0;
    this.trauma = 0;
    this.shake = v2(0, 0);
    this.scoreF = this.levelStartScore;
    this.score = Math.trunc(this.scoreF);
    this.bonus = 0;
    this.endClock = 0;
    this.penguinAt = rand(1, 3);
    this.spawnTimer = 2;
    this.moving = false;
    this.faceRow = 0;
    this.state = 'playing';
    this.shownTime = 0;
    this.lastShown = -1;
    this.ghosts = Array.from({ length: this.spec.ghosts }, () => this.newGhost(true));
  }

  /** After a win: go to the next level, carrying the score. */
  async nextLevel() {
    if (this.state !== 'won') return;
    this.levelStartScore = this.scoreF;
    if (this.isLastLevel) {
      this.loops++;
      await this.loadLevel(0);
    } else {
      await this.loadLevel(this.levelIndex + 1);
    }
  }

  // MARK: input
  /** On-screen d-pad: same key set the keyboard feeds. */
  setKey(code: string, down: boolean) {
    if (down) this.keys.add(code); else this.keys.delete(code);
  }

  private onKey(e: KeyboardEvent, down: boolean) {
    if (!down) this.keys.delete(e.code); // always release, even if the intro opened while a key was held
    const t = e.target as HTMLElement | null;
    // the landing page scrolls with these keys; focused controls keep Space for themselves
    if (document.body.classList.contains('intro-open')) return;
    const tag = t?.tagName;
    if (tag === 'SELECT') return;
    if (e.code === 'Space' && (tag === 'BUTTON' || tag === 'INPUT' || tag === 'A' || tag === 'SUMMARY')) return;
    const handled = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'KeyA', 'KeyD', 'KeyW', 'KeyS', 'Space'];
    if (!handled.includes(e.code)) return;
    e.preventDefault();
    if (down) {
      if (e.code === 'Space') {
        if (e.repeat) return;
        if (this.state === 'won') void this.nextLevel();
        else if (this.state === 'lost') this.reset();
      } else {
        this.keys.add(e.code);
      }
    } else {
      this.keys.delete(e.code);
    }
  }

  private humanDirection(): V2 {
    const d = v2(0, 0);
    const k = this.keys;
    if (k.has('ArrowLeft') || k.has('KeyA')) d.x -= 1;
    if (k.has('ArrowRight') || k.has('KeyD')) d.x += 1;
    if (k.has('ArrowUp') || k.has('KeyW')) d.y -= 1;
    if (k.has('ArrowDown') || k.has('KeyS')) d.y += 1;
    return d;
  }

  // MARK: stage helpers
  private cellIndex(g: V2): number | null {
    const n = GRID_N;
    const ix = Math.floor(g.x / this.cellG + n / 2);
    const iz = Math.floor(g.y / this.cellG + n / 2);
    if (ix < 0 || iz < 0 || ix >= n || iz >= n) return null;
    return iz * n + ix;
  }

  private isFree(g: V2): boolean {
    if (this.isRoom) {
      const i = this.cellIndex(g);
      return i !== null && this.walk[i] === 1;
    }
    return g.x >= this.marginX && g.x <= WORLD.x - this.marginX && g.y >= this.marginTop && g.y <= WORLD.y - this.marginBottom;
  }

  private slide(p: V2, d: V2): V2 {
    const t = v2(p.x + d.x, p.y + d.y);
    if (this.isFree(t)) return t;
    const tx = v2(p.x + d.x, p.y);
    const ty = v2(p.x, p.y + d.y);
    if (this.isFree(tx)) return tx;
    if (this.isFree(ty)) return ty;
    return p;
  }

  private randomPoint(awayFrom: V2, by: number): V2 {
    if (this.isRoom) {
      const far = this.spawnCells.filter((c) => dist(c, awayFrom) >= by);
      let c: V2;
      if (far.length) c = pick(far);
      else c = this.spawnCells.reduce((a, b) => (dist(b, awayFrom) > dist(a, awayFrom) ? b : a), this.spawnCells[0] ?? v2(0, 0));
      return v2(c.x + rand(-4, 4), c.y + rand(-4, 4));
    }
    for (let i = 0; i < 60; i++) {
      const c = v2(rand(this.marginX, WORLD.x - this.marginX), rand(this.marginTop, WORLD.y - this.marginBottom));
      if (dist(c, awayFrom) >= by) return c;
    }
    return v2(this.marginX, this.marginTop);
  }

  private newGhost(armed: boolean): Ghost {
    const p = this.randomPoint(this.wizard, 260 * this.rs);
    const a = rand(0, Math.PI * 2);
    const boost = 1 + 0.15 * this.loops;
    const s = rand(this.spec.speed[0], this.spec.speed[1]) * this.speedScale * boost;
    return { p, v: v2(Math.cos(a) * s, Math.sin(a) * s), age: armed ? this.ghostArm : 0, phase: rand(0, Math.PI * 2), near: false };
  }

  /** Row in the 8-direction sheets: down, down-left, left, up-left, up, up-right, right, down-right. */
  private dirRow(v: V2): number {
    if (length(v) <= 0.001) return 0;
    const a = Math.atan2(v.x, v.y); // 0 = down, positive toward the right
    const idx = roundAway(-a / (Math.PI / 4));
    return ((idx % 8) + 8) % 8;
  }

  /** Floor position (game units) -> plane position of that floor point. */
  private plane(g: V2): V2 {
    return this.isRoom ? v2(STAGE.centerX + g.x, STAGE.centerY + g.y * STAGE.sinP) : g;
  }

  /** Where a character at floor position `g` is drawn: sprite centre and feet in plane coordinates. */
  private place(g: V2, feetOff: number): { center: V2; feet: V2 } {
    if (this.isRoom) {
      const f = this.plane(g);
      return { center: v2(f.x, f.y - feetOff), feet: f };
    }
    return { center: g, feet: v2(g.x, g.y + feetOff) };
  }

  // MARK: stepping
  step(rawDt: number) {
    if (this.loading || this.aiLoading || this.paused) return;
    const dt = Math.min(rawDt, 1 / 30);
    this.clock += dt;
    this.updateParticles(dt);
    this.trauma = Math.max(0, this.trauma - 1.4 * dt);
    if (this.trauma > 0) {
      const a = this.trauma * this.trauma * 30;
      const c = this.clock;
      this.shake = v2(
        (Math.sin(c * 91) + Math.sin(c * 53 + 1.3)) * 0.5 * a,
        (Math.cos(c * 83) + Math.sin(c * 61 + 2.1)) * 0.5 * a,
      );
    } else {
      this.shake = v2(0, 0);
    }

    if (this.state !== 'playing') {
      this.endClock += dt;
      return;
    }

    let dir = v2(0, 0);
    if (this.forcedDir) {
      dir = this.forcedDir; // benchmark: the harness applies the decision directly
    } else if (!this.player) {
      dir = this.humanDirection();
    } else {
      if (this.aiError !== null) return;
      if (this.moveLeft <= 0) {
        // the wizard stands still while the model thinks; the world keeps moving
        this.moving = false;
        if (!this.pending) void this.requestMove();
      } else {
        dir = this.aiDir;
        this.moveLeft -= dt;
      }
    }
    this.elapsed += dt;
    this.scoreF += this.pointsPerSecond * dt;

    // wizard
    this.moving = length(dir) > 0.01;
    if (this.moving) {
      const n = normalize(dir);
      this.wizard = this.slide(this.wizard, v2(n.x * this.wizardSpeed * dt, n.y * this.wizardSpeed * dt));
      this.faceRow = this.dirRow(n);
      this.walkPhase += dt * 11;
      this.dustTimer -= dt;
      if (this.dustTimer <= 0) {
        this.dustTimer = 0.09;
        this.emitDust(this.place(this.wizard, 57 * this.rs).feet);
      }
    }

    // ghosts
    for (const g of this.ghosts) {
      g.age += dt;
      const d = v2(g.v.x * dt, g.v.y * dt);
      const nx = v2(g.p.x + d.x, g.p.y);
      if (this.isFree(nx)) g.p.x = nx.x; else g.v.x *= -1;
      const ny = v2(g.p.x, g.p.y + d.y);
      if (this.isFree(ny)) g.p.y = ny.y; else g.v.y *= -1;

      // near miss: a dangerous ghost brushes past without touching
      if (g.age >= this.ghostArm) {
        const dd = dist(g.p, this.wizard);
        if (dd < this.nearMissRadius && dd >= this.hitGhost) {
          if (!g.near) {
            g.near = true;
            this.scoreF += this.nearMissPoints;
            this.trauma = Math.max(this.trauma, 0.2);
            const c = this.place(this.wizard, 57 * this.rs).center;
            this.emitPuff(v2(c.x, c.y - 50 * this.rs), 90);
          }
        } else if (dd > this.nearMissRadius + 40 * this.rs) {
          g.near = false;
        }
      }
    }
    this.spawnTimer -= dt;
    if (this.spawnTimer <= 0 && this.ghosts.length < this.spec.maxGhosts) {
      this.ghosts.push(this.newGhost(false));
      this.spawnTimer = Math.max(1.0, this.spec.spawnEvery - this.elapsed * 0.04);
    }

    // penguin appears once
    if (!this.penguin && this.elapsed >= this.penguinAt) {
      const p = this.randomPoint(this.wizard, 340 * this.rs);
      this.penguin = p;
      this.penguinBorn = this.clock;
      const c = this.place(p, 76 * this.rs).center;
      for (let i = 0; i < 9; i++) this.emitPuff(c, 220);
    }

    // HUD (time in tenths, score in whole points)
    const tenths = Math.trunc(this.elapsed * 10);
    if (tenths !== this.lastShown) { this.lastShown = tenths; this.shownTime = this.elapsed; }
    if (Math.trunc(this.scoreF) !== this.score) this.score = Math.trunc(this.scoreF);

    // collisions
    for (const g of this.ghosts) {
      if (g.age >= this.ghostArm && dist(g.p, this.wizard) < this.hitGhost) {
        this.finish('lost', g.p);
        return;
      }
    }
    if (this.penguin && dist(this.penguin, this.wizard) < this.hitGoal) this.finish('won', this.penguin);
  }

  private finish(result: GameState, hit: V2) {
    this.state = result;
    this.endClock = 0;
    this.shownTime = this.elapsed;
    this.epoch++; // discard in-flight AI answers
    this.pending = false;
    this.aiThinking = false;
    if (result === 'won') {
      this.bonus = this.goalPoints + Math.trunc(Math.max(0, 60 - this.elapsed) * 10); // quicker catch, bigger bonus
      this.scoreF += this.bonus;
      if (this.bestTime === null || this.elapsed < this.bestTime) {
        this.bestTime = this.elapsed;
        store.set('sf_best', this.bestTime);
      }
      this.emitConfetti(this.place(hit, 76 * this.rs).center);
    } else {
      this.emitSplat(this.place(this.wizard, 57 * this.rs).center);
      this.trauma = 1; // camera shake on being hit
    }
    this.score = Math.trunc(this.scoreF);
    if (this.score > this.bestScore) {
      this.bestScore = this.score;
      store.set('sf_best_score', this.bestScore);
    }
  }

  // MARK: particles (plane coordinates)
  private emitDust(p: V2) {
    this.particles.push({
      p: v2(p.x + rand(-8, 8), p.y), v: v2(rand(-25, 25), rand(-30, -8)), life: 0.5, maxLife: 0.5,
      size: rand(12, 20) * this.rs, color: [0.45, 0.5, 0.62], gravity: 0, grow: 40,
    });
  }

  private emitPuff(p: V2, strength: number) {
    const a = rand(0, Math.PI * 2);
    const s = rand(0.3, 1) * strength;
    this.particles.push({
      p: { ...p }, v: v2(Math.cos(a) * s, Math.sin(a) * s), life: 0.7, maxLife: 0.7,
      size: rand(18, 34) * this.rs, color: [0.9, 0.95, 1], gravity: 0, grow: 30,
    });
  }

  private emitConfetti(p: V2) {
    const palette: [number, number, number][] = [
      [1.0, 0.35, 0.4], [1.0, 0.82, 0.25], [0.35, 0.85, 0.55], [0.35, 0.65, 1.0], [0.85, 0.5, 1.0],
    ];
    for (let i = 0; i < 90; i++) {
      const a = rand(0, Math.PI * 2);
      const s = rand(140, 520);
      const life = rand(0.9, 1.8);
      this.particles.push({
        p: { ...p }, v: v2(Math.cos(a) * s, (Math.sin(a) * 0.9 - 0.5) * s), life, maxLife: life,
        size: rand(9, 16), color: pick(palette), gravity: 520, grow: 0,
      });
    }
  }

  private emitSplat(p: V2) {
    for (let i = 0; i < 40; i++) {
      const a = rand(0, Math.PI * 2);
      const s = rand(80, 380);
      const life = rand(0.5, 1.0);
      this.particles.push({
        p: { ...p }, v: v2(Math.cos(a) * s, Math.sin(a) * s), life, maxLife: life,
        size: rand(10, 20), color: [0.75, 0.85, 1.0], gravity: 240, grow: -8,
      });
    }
  }

  private updateParticles(dt: number) {
    if (!this.particles.length) return;
    for (const p of this.particles) {
      p.life -= dt;
      p.v.y += p.gravity * dt;
      p.v.x *= 1 - 1.8 * dt;
      p.v.y *= 1 - 1.8 * dt;
      p.p.x += p.v.x * dt;
      p.p.y += p.v.y * dt;
      p.size = Math.max(0, p.size + p.grow * dt);
    }
    this.particles = this.particles.filter((p) => p.life > 0);
  }

  // MARK: AI
  private rel(target: V2): string {
    const dx = target.x - this.wizard.x;
    const dy = target.y - this.wizard.y;
    const h = Math.abs(dx) < 10 ? '' : `${Math.trunc(Math.abs(dx))} px ${dx > 0 ? 'right' : 'left'}`;
    const v = Math.abs(dy) < 10 ? '' : `${Math.trunc(Math.abs(dy))} px ${dy > 0 ? 'down' : 'up'}`;
    const parts = [h, v].filter((s) => s !== '');
    return parts.length ? parts.join(' and ') : 'right on top of the wizard';
  }

  /** Distance the wizard could walk in direction `v` before hitting a wall or obstacle. */
  private freeDistance(v: V2): number {
    const n = normalize(v);
    let d = 0;
    while (d < 400 && this.isFree(v2(this.wizard.x + n.x * (d + 8), this.wizard.y + n.y * (d + 8)))) d += 8;
    return Math.trunc(d);
  }

  private describeState(): string {
    const w = WORLD;
    const lines: string[] = [];
    if (this.isRoom) {
      const free = DIRS.map((d) => `${d.key.replace('_', ' ')} ${this.freeDistance(d.vec)} px`).join(', ');
      lines.push(
        `A young wizard walks around a dungeon room (${this.spec.name}). Open floor the wizard can walk ` +
        `before hitting a wall, pit or obstacle: ${free}.`,
      );
    } else {
      lines.push(
        `A young wizard walks around a ${w.x} by ${w.y} px meadow. It is ${Math.trunc(this.wizard.x)} px from the left wall, ` +
        `${Math.trunc(w.x - this.wizard.x)} px from the right wall, ${Math.trunc(this.wizard.y)} px from the top wall and ` +
        `${Math.trunc(w.y - this.wizard.y)} px from the bottom wall.`,
      );
    }
    if (this.penguin) {
      lines.push(`The goal penguin is ${this.rel(this.penguin)} of the wizard (distance ${Math.trunc(dist(this.penguin, this.wizard))} px).`);
    } else {
      lines.push('The goal penguin has not appeared yet.');
    }
    const near = [...this.ghosts].sort((a, b) => dist(a.p, this.wizard) - dist(b.p, this.wizard)).slice(0, 6);
    lines.push('Dangerous ghosts (touching one means losing), nearest first:');
    for (const g of near) {
      const toward = g.v.x * (this.wizard.x - g.p.x) + g.v.y * (this.wizard.y - g.p.y) > 0
        ? 'drifting toward the wizard' : 'drifting away from the wizard';
      const armed = g.age >= this.ghostArm ? '' : ', still harmless for a moment';
      lines.push(
        `- ${this.rel(g.p)} (distance ${Math.trunc(dist(g.p, this.wizard))} px), ${toward} at ` +
        `${Math.trunc(length(g.v))} px/s${armed}`,
      );
    }
    return lines.join('\n');
  }

  /**
   * Compact scene description for small classifier models (Laya). They key on the direction words they see, so
   * this states only what matters, dominant axis first: the penguin, the nearest ghosts, and a wall only when close.
   */
  private describeStateCompact(): string {
    const axes = (t: V2): string => {
      const dx = t.x - this.wizard.x;
      const dy = t.y - this.wizard.y;
      const h = Math.abs(dx) < 10 ? '' : `${Math.round(Math.abs(dx) / 10) * 10} px ${dx > 0 ? 'to the right' : 'to the left'}`;
      const v = Math.abs(dy) < 10 ? '' : `${Math.round(Math.abs(dy) / 10) * 10} px ${dy > 0 ? 'down' : 'up'}`;
      const parts = Math.abs(dx) >= Math.abs(dy) ? [h, v] : [v, h];
      const out = parts.filter((p) => p !== '').join(' and ');
      return out || 'right on top of the wizard';
    };
    const lines: string[] = [];
    lines.push(this.penguin ? `The penguin is ${axes(this.penguin)}.` : 'The penguin has not appeared yet.');
    const near = [...this.ghosts]
      .filter((g) => dist(g.p, this.wizard) < 280)
      .sort((a, b) => dist(a.p, this.wizard) - dist(b.p, this.wizard))
      .slice(0, 2);
    if (near.length === 0) lines.push('No ghost is near.');
    near.forEach((g, i) => {
      const coming = g.v.x * (this.wizard.x - g.p.x) + g.v.y * (this.wizard.y - g.p.y) > 0;
      lines.push(`${near.length > 1 ? `Ghost ${i + 1}` : 'A ghost'} is ${axes(g.p)}${coming ? ' and is coming at the wizard' : ''}.`);
    });
    const close = DIRS.filter((d) => d.key.indexOf('_') < 0 && this.freeDistance(d.vec) < 80).map((d) => d.key);
    if (close.length) lines.push(`A wall is close on the ${close.join(' and ')} side.`);
    return lines.join(' ');
  }

  /** The decision prompt a model receives; shared by the game and the benchmark. */
  buildPrompt(compact: boolean): { state: string; instructions: string; options: OptionSpec[] } {
    const safe = this.isRoom ? 100 : 150;
    const blockedNote =
      ' If a direction is blocked by a wall or obstacle, the wizard slides along it and may not ' +
      'move at all that tick, so prefer directions with open floor in the distances given.';
    // The small encoders have ~192 tokens for instructions + options, so they get a short question.
    const instructions = compact
      ? this.penguin
        ? 'Which way should the wizard walk to reach the penguin?'
        : 'Which way should the wizard walk to stay safe?'
      : this.penguin
        ? 'Choose the direction the wizard should walk for the next third of a second. It must get to the penguin ' +
          'while staying clear of every ghost, especially ones drifting toward it, and it cannot walk through ' +
          `walls or obstacles. If a ghost is within ${safe} px, prefer safety over speed.` + blockedNote
        : "Choose the direction the wizard should walk for the next third of a second. The penguin hasn't " +
          'appeared, so move to the safest spot: away from every ghost and away from walls and corners.' + blockedNote;
    return {
      state: compact ? this.describeStateCompact() : this.describeState(),
      instructions,
      options: DIRS.map((d) => ({ key: d.key, description: d.desc })),
    };
  }

  private async requestMove() {
    this.pending = true;
    this.aiThinking = true;
    const myEpoch = this.epoch;
    const model = this.player;
    const prompt = this.buildPrompt(specFor(model)?.compactPrompt ?? false);

    try {
      const d = await backendFor(model).decide(prompt.state, prompt.instructions, prompt.options);
      if (this.epoch !== myEpoch) return;
      this.aiDir = DIRS.find((x) => x.key === d.choice)?.vec ?? v2(0, 0);
      this.moveLeft = this.moveTime;
      this.pending = false;
      this.aiThinking = false;
      this.aiMoves++;
      this.aiProbs = DIRS.map((x) => ({ id: x.key, label: x.label, p: d.probabilities[x.key] ?? 0 }));
      this.aiStatus = `${d.choice.replace('_', ' ')} · ${(d.confidence * 100).toFixed(0)}% sure · ${d.latency.toFixed(1)}s`;
    } catch (e) {
      if (this.epoch !== myEpoch) return;
      this.pending = false;
      this.aiThinking = false;
      this.aiError = (e as Error).message;
      this.aiStatus = this.aiError;
    }
  }

  // MARK: lights + scene
  /** Light pools: wizard lantern, ghost glows, penguin beacon, spawn cracks, scenery. */
  lights(): LightRec[] {
    const out: LightRec[] = [];
    const lanternFade = this.state === 'lost' ? Math.max(0, 1 - this.endClock * 1.1) : 1;
    const flicker = 0.94 + 0.06 * Math.sin(this.clock * 13) * Math.sin(this.clock * 5.3);
    const room = this.isRoom;
    const sc = room ? 0.62 : 1; // rooms are smaller than the meadow
    const hL = room ? 85 : 40;
    out.push({ x: this.wizard.x, z: this.wizard.y, radius: 430 * sc, intensity: 1.05 * flicker * lanternFade, color: [1.0, 0.84, 0.55], height: hL });
    if (this.penguin) {
      const k = easeOutBack((this.clock - this.penguinBorn) / 0.4);
      out.push({
        x: this.penguin.x, z: this.penguin.y, radius: 270 * sc * k,
        intensity: 0.5 + 0.12 * Math.sin(this.clock * 4), color: [1.0, 0.86, 0.42], height: room ? 50 : 30,
      });
    }
    for (const g of this.ghosts) {
      const t = Math.min(1, g.age / this.ghostArm);
      out.push({ x: g.p.x, z: g.p.y, radius: 210 * sc, intensity: 0.55 * t, color: [0.4, 0.62, 1.0], height: room ? 80 : 40 });
      if (g.age < this.ghostArm + 0.4 && out.length < 24) {
        const fade = g.age < this.ghostArm ? 1 : Math.max(0, 1 - (g.age - this.ghostArm) / 0.4);
        out.push({
          x: g.p.x, z: g.p.y + (room ? 0 : 46), radius: 170 * sc,
          intensity: 0.8 * fade * (0.7 + 0.3 * Math.sin(this.clock * 16)), color: [1.0, 0.42, 0.12], height: room ? 8 : 20,
        });
      }
      if (out.length >= 24) break;
    }
    if (room) {
      this.levelGlows.forEach((gl, i) => {
        if (out.length >= 32) return;
        const flick = 0.85 + 0.15 * Math.sin(this.clock * 15 + i * 1.9) * Math.sin(this.clock * 6.1 + i);
        out.push({ ...gl, intensity: gl.intensity * flick });
      });
    } else {
      PROPS.forEach((pr, i) => {
        if (!pr.glow || out.length >= 32) return;
        const r = PROP_RECTS[pr.name];
        if (!r) return;
        const height = r[3] * (pr.scale ?? 3);
        const flick = 0.86 + 0.14 * Math.sin(this.clock * 17 + i * 1.7) * Math.sin(this.clock * 7.3 + i);
        out.push({
          x: pr.p.x, z: pr.p.y - height * pr.glow.lift, radius: pr.glow.radius,
          intensity: pr.glow.intensity * flick, color: pr.glow.color, height: 40,
        });
      });
    }
    return out;
  }

  private frame(col: number, row: number, cols: number): [number, number, number, number] {
    return [col / cols, row / 8, 1 / cols, 1 / 8];
  }

  /** Everything except particles/vignette, back to front. */
  buildScene(): { sprites: SpriteDraw[]; particles: ParticleDraw[] } {
    const sprites: SpriteDraw[] = [];
    const room = this.isRoom;
    const rs = this.rs;
    const WHITE: [number, number, number, number] = [1, 1, 1, 1];

    /** `feetPlane`/`centerPlane` are in plane coordinates (see place()); `g` is the floor game position. */
    const toDraw = (
      tex: TexName, uv: [number, number, number, number], g: V2, feet: V2, center: V2, size: V2,
      opts: { rot?: number; tint?: [number, number, number, number]; lit?: boolean; smooth?: boolean; flash?: number } = {},
    ): SpriteDraw => ({
      tex, kind: 0, uv, smooth: opts.smooth ?? false, lit: opts.lit ?? true,
      floor: room ? g : feet, lift: feet.y - center.y, gpos: room ? g : center,
      size, rot: opts.rot ?? 0, tint: opts.tint ?? WHITE, flash: opts.flash ?? 0, decal: false,
    });

    // spawn cracks (ground layer)
    for (const g of this.ghosts) {
      if (g.age >= this.ghostArm + 0.4) continue;
      const t = Math.min(1, g.age / this.ghostArm);
      const fade = g.age < this.ghostArm ? 1 : Math.max(0, 1 - (g.age - this.ghostArm) / 0.4);
      const pulse = 0.65 + 0.35 * Math.sin(this.clock * 16);
      const s = 190 * rs * (1.15 - 0.15 * t);
      const feet = room ? g.p : v2(g.p.x, g.p.y + 46);
      sprites.push({
        tex: 'cracks', kind: 0, uv: [0, 0, 1, 1], smooth: false, lit: false,
        floor: feet, lift: 0, gpos: g.p, size: v2(s, s * (room ? STAGE.sinP : 1)), rot: 0,
        tint: [1, 1, 1, 0.95 * fade * pulse], flash: 0, decal: true,
      });
    }

    // shadows (on the floor)
    const shadow = (g: V2, width: number, feetOff: number, a = 0.5) => {
      const f = this.place(g, feetOff).feet;
      sprites.push({
        tex: null, kind: 1, uv: [0, 0, 1, 1], smooth: false, lit: false,
        floor: room ? g : f, lift: 0, gpos: g, size: v2(width, width * 0.36), rot: 0,
        tint: [0.02, 0.07, 0.02, a], flash: 0, decal: true,
      });
    };
    for (const g of this.ghosts) shadow(g.p, 110 * rs * easeOutBack(g.age / 0.5), 58 * rs, 0.28);
    if (this.penguin) shadow(this.penguin, 130 * rs * easeOutBack((this.clock - this.penguinBorn) / 0.4), 76 * rs);
    const winHop = this.state === 'won' ? Math.max(0.5, 1 - 0.12 * Math.abs(Math.sin(this.endClock * 7))) : 1;
    shadow(this.wizard, 100 * rs * winHop, 56 * rs);

    // characters + scenery, back to front
    const ents: { y: number; d: SpriteDraw }[] = [];

    if (!room) {
      for (const pr of PROPS) {
        const r = PROP_RECTS[pr.name];
        if (!r) continue;
        const sc = pr.scale ?? 3;
        const size = v2(r[2] * sc, r[3] * sc);
        const uv: [number, number, number, number] = [
          r[0] / PROP_ATLAS_SIZE.w, r[1] / PROP_ATLAS_SIZE.h, r[2] / PROP_ATLAS_SIZE.w, r[3] / PROP_ATLAS_SIZE.h,
        ];
        const center = v2(pr.p.x, pr.p.y - size.y / 2);
        ents.push({
          y: pr.flat ? -1000 + pr.p.y : pr.p.y,
          d: toDraw('props', uv, center, pr.p, center, size, { tint: [1, 1, 1, pr.alpha ?? 1] }),
        });
      }
    }

    for (const g of this.ghosts) {
      const k = easeOutBack(g.age / 0.5);
      const armed = g.age >= this.ghostArm;
      const alpha = armed ? 1 : 0.25 + 0.6 * (g.age / this.ghostArm);
      const bob = 5 * rs * Math.sin(this.clock * 3 + g.phase);
      const col = Math.floor(this.clock * 8 + g.phase * 2) % 6;
      const row = this.dirRow(g.v);
      const size = this.spriteSize * k;
      const pl = this.place(g.p, 58 * rs);
      const flick = 0.9 + 0.1 * Math.sin(this.clock * 9 + g.phase * 3);
      ents.push({
        y: g.p.y,
        d: toDraw('ghostWalk', this.frame(col, row, 6), g.p, pl.feet, v2(pl.center.x, pl.center.y + bob - 4 * rs), v2(size, size),
          { tint: [0.78, 0.9, 1.0, alpha * flick], lit: false }),
      });
    }

    if (this.penguin) {
      const s = this.penguin;
      const born = this.clock - this.penguinBorn;
      const k = easeOutBack(born / 0.4);
      const phase = this.clock % 1.3;
      const hop = (phase < 0.45 ? Math.sin((phase / 0.45) * Math.PI) * 22 : 0) * rs;
      const squash = phase < 0.45 ? 1 + 0.06 * Math.sin((phase / 0.45) * Math.PI) : 1 - 0.04 * Math.exp(-(phase - 0.45) * 12);
      // The penguin watches the wizard. Its sheet has no up-left row, so mirror up-right.
      let row = this.dirRow(v2(this.wizard.x - s.x, this.wizard.y - s.y));
      let mirror = 1;
      if (row === 3) { row = 5; mirror = -1; }
      const col = Math.floor(this.clock * 3) % 4;
      const size = 180 * rs * k;
      const pl = this.place(s, 76 * rs);
      ents.push({
        y: s.y,
        d: toDraw('penguin', this.frame(col, row, 4), s, pl.feet, v2(pl.center.x, pl.center.y - 7 * rs - hop),
          v2((size * mirror) / squash, size * squash), { smooth: true }),
      });
    }

    {
      const pl = this.place(this.wizard, 57 * rs);
      let center = { ...pl.center };
      let size = v2(this.spriteSize, this.spriteSize);
      let rot = 0;
      let tint = WHITE;
      let tex: TexName = 'wizardIdle';
      let uv = this.frame(Math.floor(this.clock * 4) % 4, this.faceRow, 4);
      switch (this.state) {
        case 'playing':
          if (this.moving) { tex = 'wizardWalk'; uv = this.frame(Math.floor(this.walkPhase) % 6, this.faceRow, 6); }
          break;
        case 'won':
          tex = 'wizardIdle';
          uv = this.frame(Math.floor(this.clock * 6) % 4, 0, 4);
          center.y -= Math.abs(Math.sin(this.endClock * 7)) * 46 * rs;
          break;
        case 'lost':
          tex = 'wizardWalk';
          uv = this.frame(0, this.faceRow, 6);
          rot = Math.min(this.endClock * 4, 1) * 1.45;
          size = v2(size.x * Math.max(0.75, 1 - this.endClock * 0.2), size.y * Math.max(0.75, 1 - this.endClock * 0.2));
          tint = [1, 0.6, 0.6, Math.max(0.2, 1 - this.endClock * 0.5)];
          break;
      }
      ents.push({ y: this.wizard.y, d: toDraw(tex, uv, this.wizard, pl.feet, center, size, { rot, tint }) });
    }
    ents.sort((a, b) => a.y - b.y);
    for (const e of ents) sprites.push(e.d);

    const particles: ParticleDraw[] = this.particles.map((p) => ({
      x: p.p.x, y: p.p.y, size: p.size * 2, color: p.color,
      alpha: Math.min(1, (p.life / p.maxLife) * 1.6) * 0.7,
    }));
    return { sprites, particles };
  }

  // MARK: benchmark hooks (used by src/bench; not part of normal play)

  /** Start a level for a headless run: no model, no keyboard; the harness drives it with benchAdvance(). */
  async benchStart(level: number) {
    this.player = '@bench';
    this.levelStartScore = 0;
    this.loops = 0;
    await this.loadLevel(level);
  }

  get benchState(): GameState { return this.state; }
  get benchScore(): number { return Math.trunc(this.scoreF); }
  get benchElapsed(): number { return this.elapsed; }

  /** Walk in direction `key` for one decision interval (0.35 s of game time) at a fixed 60 Hz step. */
  benchAdvance(key: string): GameState {
    const d = DIRS.find((x) => x.key === key);
    this.forcedDir = d ? d.vec : v2(0, 0);
    const dt = 1 / 60;
    for (let t = 0; t < this.moveTime && this.state === 'playing'; t += dt) this.step(dt);
    this.forcedDir = null;
    return this.state;
  }

  /** Let time pass with the wizard standing still (used to reach the interesting moments). */
  benchWait(seconds: number): GameState {
    this.forcedDir = v2(0, 0);
    const dt = 1 / 60;
    for (let t = 0; t < seconds && this.state === 'playing'; t += dt) this.step(dt);
    this.forcedDir = null;
    return this.state;
  }

  /**
   * Ground truth for the current position: simulate each of the 8 directions for two decision intervals
   * (0.7 s) with the real collision rules, ghosts moving on their current headings.
   */
  private oracleNow() {
    const horizon = 2 * this.moveTime;
    const dt = 1 / 30;
    const steps = Math.round(horizon / dt);
    const startPen = this.penguin ? dist(this.wizard, this.penguin) : 0;
    const out = DIRS.map((d) => {
      let w = { ...this.wizard };
      const start = { ...w };
      const gh = this.ghosts.map((g) => ({ p: { ...g.p }, v: { ...g.v }, age: g.age }));
      const n = normalize(d.vec);
      let fatal = false;
      let minDist = Infinity;
      for (let i = 0; i < steps; i++) {
        w = this.slide(w, v2(n.x * this.wizardSpeed * dt, n.y * this.wizardSpeed * dt));
        for (const g of gh) {
          g.age += dt;
          const nx = v2(g.p.x + g.v.x * dt, g.p.y);
          if (this.isFree(nx)) g.p.x = nx.x; else g.v.x *= -1;
          const ny = v2(g.p.x, g.p.y + g.v.y * dt);
          if (this.isFree(ny)) g.p.y = ny.y; else g.v.y *= -1;
          if (g.age >= this.ghostArm) {
            const dd = dist(g.p, w);
            minDist = Math.min(minDist, dd);
            if (dd < this.hitGhost) fatal = true;
          }
        }
      }
      const wanted = this.wizardSpeed * horizon;
      return {
        key: d.key,
        fatal,
        minDist: Number.isFinite(minDist) ? minDist : 999,
        progress: this.penguin ? startPen - dist(w, this.penguin) : 0,
        blocked: dist(start, w) < 0.4 * wanted,
      };
    });
    const ok = out.filter((o) => !o.fatal && !o.blocked);
    const pool = ok.length ? ok : out.filter((o) => !o.fatal).length ? out.filter((o) => !o.fatal) : out;
    const score = (o: (typeof out)[number]) =>
      this.penguin ? o.progress + 0.15 * Math.min(o.minDist, 200) : Math.min(o.minDist, 300);
    const top = Math.max(...pool.map(score));
    const best = pool.filter((o) => score(o) >= top - 2).map((o) => o.key);
    let acceptable: string[];
    if (this.penguin) {
      const bestProgress = Math.max(...pool.map((o) => o.progress));
      acceptable = pool.filter((o) => bestProgress <= 0 || o.progress >= 0.5 * bestProgress).map((o) => o.key);
    } else {
      const bestMin = Math.max(...pool.map((o) => o.minDist));
      acceptable = pool.filter((o) => o.minDist >= 0.7 * bestMin).map((o) => o.key);
    }
    return {
      best,
      acceptable: acceptable.length ? acceptable : best,
      fatal: out.filter((o) => o.fatal).map((o) => o.key),
      blocked: out.filter((o) => o.blocked).map((o) => o.key),
    };
  }

  /** The decision point the game is at right now, with both prompt styles and the oracle's verdict. */
  benchSnapshot(id: string, tick: number): BenchScenario {
    const rs = this.rs;
    const armed = this.ghosts.filter((g) => g.age >= this.ghostArm);
    const nearest = armed.reduce<number | null>((m, g) => {
      const d = dist(g.p, this.wizard);
      return m === null || d < m ? d : m;
    }, null);
    const coming = armed.some(
      (g) =>
        dist(g.p, this.wizard) < 170 * rs &&
        g.v.x * (this.wizard.x - g.p.x) + g.v.y * (this.wizard.y - g.p.y) > 0,
    );
    const evadeNear = nearest !== null && nearest < 220 * rs;
    const category: BenchScenario['category'] = this.penguin
      ? coming ? 'threat' : 'goal'
      : evadeNear ? 'evade' : 'calm';
    const tight = DIRS.filter((d) => d.key.indexOf('_') < 0).some((d) => this.freeDistance(d.vec) < 50 * rs);
    const compact = this.buildPrompt(true);
    const full = this.buildPrompt(false);
    return {
      id,
      level: this.levelIndex + 1,
      levelName: this.spec.name,
      room: this.isRoom,
      tick,
      category,
      tight,
      hasPenguin: !!this.penguin,
      nearestGhost: nearest === null ? null : Math.round(nearest),
      ghostsNear: armed.filter((g) => dist(g.p, this.wizard) < 260 * rs).length,
      prompts: {
        compact: { state: compact.state, instructions: compact.instructions },
        full: { state: full.state, instructions: full.instructions },
      },
      oracle: this.oracleNow(),
    };
  }
}
