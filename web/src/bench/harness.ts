/**
 * Benchmark harness: captures decision points from headless runs of the real game, then asks each model to
 * decide on them and scores the answer against the oracle (see Game.oracleNow).
 */
import { Game, type BenchScenario } from '../game';
import { LEVELS } from '../levels';
import type { ModelBackend } from '../models/types';

const KEYS = ['up', 'up_right', 'right', 'down_right', 'down', 'down_left', 'left', 'up_left'];

/** Small deterministic PRNG so every run sees the same scenarios. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface GenerateOptions {
  episodesPerLevel: number;
  maxTicks: number;
  explore: number; // probability of a random move instead of the oracle's, to wander into trouble
  total: number; // scenarios kept after stratification
  seed: number;
}

/** Play seeded headless episodes with a noisy oracle policy and keep a stratified sample of the decision points. */
export async function generateScenarios(game: Game, o: GenerateOptions, log: (m: string) => void): Promise<BenchScenario[]> {
  const realRandom = Math.random;
  const all: BenchScenario[] = [];
  try {
    for (let level = 0; level < LEVELS.length; level++) {
      for (let ep = 0; ep < o.episodesPerLevel; ep++) {
        const rng = mulberry32(o.seed + level * 1000 + ep * 17);
        Math.random = rng;
        await game.benchStart(level);
        for (let tick = 0; tick < o.maxTicks; tick++) {
          const snap = game.benchSnapshot(`L${level + 1}-e${ep}-t${tick}`, tick);
          all.push(snap);
          const pool = snap.oracle.best;
          const key = rng() < o.explore ? KEYS[Math.floor(rng() * KEYS.length)] : pool[Math.floor(rng() * pool.length)];
          if (game.benchAdvance(key) !== 'playing') break;
        }
      }
      log(`generated level ${level + 1}/${LEVELS.length}: ${all.length} decision points so far`);
    }
  } finally {
    Math.random = realRandom;
  }

  // Stratified, deterministic sample: quotas per category, spread round-robin across levels.
  const rng = mulberry32(o.seed ^ 0x9e3779b9);
  const shuffle = <T,>(a: T[]) => {
    const b = [...a];
    for (let i = b.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [b[i], b[j]] = [b[j], b[i]];
    }
    return b;
  };
  const quota: Record<BenchScenario['category'], number> = {
    goal: Math.round(o.total * 0.34),
    threat: Math.round(o.total * 0.34),
    evade: Math.round(o.total * 0.22),
    calm: Math.round(o.total * 0.1),
  };
  const picked: BenchScenario[] = [];
  for (const cat of Object.keys(quota) as BenchScenario['category'][]) {
    // prefer decision points where the choice matters: some direction is fatal or blocked
    const pool = shuffle(all.filter((s) => s.category === cat));
    const matters = pool.filter((s) => s.oracle.fatal.length > 0 || s.oracle.blocked.length > 0);
    const rest = pool.filter((s) => !matters.includes(s));
    const ordered = [...matters.slice(0, Math.ceil(quota[cat] * 0.6)), ...rest, ...matters.slice(Math.ceil(quota[cat] * 0.6))];
    const byLevel = new Map<number, BenchScenario[]>();
    for (const s of ordered) byLevel.set(s.level, [...(byLevel.get(s.level) ?? []), s]);
    const out: BenchScenario[] = [];
    while (out.length < quota[cat] && [...byLevel.values()].some((l) => l.length)) {
      for (const l of byLevel.values()) {
        const s = l.shift();
        if (s && out.length < quota[cat]) out.push(s);
      }
    }
    picked.push(...out);
  }
  log(`kept ${picked.length} of ${all.length} decision points`);
  return picked;
}

export type Variant = 'compact' | 'full' | 'natural';

const NATURAL: Record<string, string> = {
  up: 'walk up', up_right: 'walk up and to the right', right: 'walk right', down_right: 'walk down and to the right',
  down: 'walk down', down_left: 'walk down and to the left', left: 'walk left', up_left: 'walk up and to the left',
};

export interface DecisionRecord {
  s: string; // scenario id
  v: Variant;
  choice: string;
  best: boolean; // matches an oracle-best direction
  ok: boolean; // acceptable: safe and makes progress
  fatal: boolean; // walks into a ghost within 0.7 s
  blocked: boolean; // walks into a wall and barely moves
  ms: number;
  conf: number;
  top: [string, number][];
  err?: string;
}

export async function runModel(
  backend: ModelBackend,
  scenarios: BenchScenario[],
  variant: Variant,
  onProgress: (done: number, total: number) => void,
): Promise<DecisionRecord[]> {
  const rows: DecisionRecord[] = [];
  const options = KEYS.map((k) => ({
    key: k,
    label: variant === 'natural' ? NATURAL[k] : undefined,
    description: {
      up: 'Walk straight up', up_right: 'Walk diagonally up and to the right', right: 'Walk straight to the right',
      down_right: 'Walk diagonally down and to the right', down: 'Walk straight down',
      down_left: 'Walk diagonally down and to the left', left: 'Walk straight to the left',
      up_left: 'Walk diagonally up and to the left',
    }[k]!,
  }));
  for (let i = 0; i < scenarios.length; i++) {
    const sc = scenarios[i];
    const p = sc.prompts[variant === 'natural' ? 'compact' : variant];
    const t0 = performance.now();
    try {
      const d = await backend.decide(p.state, p.instructions, options);
      const ms = performance.now() - t0;
      const top = Object.entries(d.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 3)
        .map(([k, v]) => [k, Math.round(v * 1000) / 1000] as [string, number]);
      rows.push({
        s: sc.id, v: variant, choice: d.choice,
        best: sc.oracle.best.includes(d.choice),
        ok: sc.oracle.acceptable.includes(d.choice),
        fatal: sc.oracle.fatal.includes(d.choice),
        blocked: sc.oracle.blocked.includes(d.choice),
        ms: Math.round(ms), conf: Math.round(d.confidence * 1000) / 1000, top,
      });
    } catch (e) {
      rows.push({ s: sc.id, v: variant, choice: '', best: false, ok: false, fatal: false, blocked: false, ms: Math.round(performance.now() - t0), conf: 0, top: [], err: (e as Error).message });
    }
    onProgress(i + 1, scenarios.length);
    await new Promise((r) => setTimeout(r, 0)); // let the page breathe so progress can be polled
  }
  return rows;
}

export function percentile(values: number[], p: number): number {
  if (!values.length) return 0;
  const a = [...values].sort((x, y) => x - y);
  return a[Math.min(a.length - 1, Math.floor((p / 100) * a.length))];
}

// ----------------------------------------------------------------------------------------------- episodes

export interface EpisodeResult {
  level: number;
  seed: number;
  outcome: 'won' | 'lost' | 'timeout';
  decisions: number;
  seconds: number; // game time
  score: number;
}

export interface EpisodeOptions {
  levels: number[]; // 1-based
  per: number; // episodes per level
  maxDecisions: number; // each decision is 0.35 s of game time
  seed: number;
}

/**
 * Play complete headless runs. The world is frozen while a decision is computed (like the macOS app), so this
 * measures decision quality, not speed; latency is reported separately. `policy` returns a direction key.
 */
export async function playEpisodes(
  game: Game,
  policy: (g: Game) => Promise<string>,
  o: EpisodeOptions,
  onProgress: (done: number, total: number) => void,
): Promise<EpisodeResult[]> {
  const out: EpisodeResult[] = [];
  const total = o.levels.length * o.per;
  const realRandom = Math.random;
  try {
    for (const level of o.levels) {
      for (let ep = 0; ep < o.per; ep++) {
        const seed = o.seed + level * 101 + ep * 7;
        Math.random = mulberry32(seed);
        await game.benchStart(level - 1);
        let decisions = 0;
        while (game.benchState === 'playing' && decisions < o.maxDecisions) {
          const key = await policy(game);
          game.benchAdvance(key);
          decisions++;
          if (decisions % 4 === 0) await new Promise((r) => setTimeout(r, 0));
        }
        const st = game.benchState;
        out.push({
          level, seed, decisions,
          outcome: st === 'won' ? 'won' : st === 'lost' ? 'lost' : 'timeout',
          seconds: Math.round(game.benchElapsed * 100) / 100,
          score: game.benchScore,
        });
        onProgress(out.length, total);
      }
    }
  } finally {
    Math.random = realRandom;
  }
  return out;
}

export function summarizeEpisodes(rs: EpisodeResult[]) {
  const n = rs.length;
  const won = rs.filter((r) => r.outcome === 'won');
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
  return {
    n,
    winRate: n ? won.length / n : 0,
    lossRate: n ? rs.filter((r) => r.outcome === 'lost').length / n : 0,
    timeoutRate: n ? rs.filter((r) => r.outcome === 'timeout').length / n : 0,
    meanSecondsToWin: mean(won.map((r) => r.seconds)),
    meanSecondsSurvived: mean(rs.map((r) => r.seconds)),
    meanScore: mean(rs.map((r) => r.score)),
    byLevel: Object.fromEntries(
      [...new Set(rs.map((r) => r.level))].map((l) => {
        const xs = rs.filter((r) => r.level === l);
        return [l, { n: xs.length, winRate: xs.filter((r) => r.outcome === 'won').length / xs.length }];
      }),
    ),
  };
}

export const DIRECTION_KEYS = KEYS;
