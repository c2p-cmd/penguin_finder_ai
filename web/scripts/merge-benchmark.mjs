// Merges bench-results/scenarios.json + bench-results/models/*.json into public/data/benchmark.json,
// the file the intro page renders. Keeps per-model summaries plus a few showcase scenarios with every model's
// answer; the raw per-decision rows stay in bench-results (not shipped).
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dir = resolve(here, '../bench-results');
const out = resolve(here, '../public/data/benchmark.json');

if (!existsSync(join(dir, 'scenarios.json'))) {
  console.error('merge-benchmark: no bench-results/scenarios.json (run the benchmark first)');
  process.exit(1);
}
const scen = JSON.parse(readFileSync(join(dir, 'scenarios.json'), 'utf8'));
const scenarios = scen.scenarios;
const order = ['laya-web', 'julia-1', 'kev-0.6b-bu', 'decision-lab'];
const all = existsSync(join(dir, 'models')) ? readdirSync(join(dir, 'models')).filter((f) => f.endsWith('.json')) : [];
const files = all.filter((f) => !f.startsWith('episodes_'));
const results = files.map((f) => JSON.parse(readFileSync(join(dir, 'models', f), 'utf8')));
// whole-episode runs: one file per model id, plus the @random / @oracle baselines
const episodes = Object.fromEntries(
  all.filter((f) => f.startsWith('episodes_')).map((f) => {
    const r = JSON.parse(readFileSync(join(dir, 'models', f), 'utf8'));
    return [r.id, r];
  }),
);
const episodeSummary = (r) =>
  r && {
    variant: r.variant, n: r.summary.n, winRate: r.summary.winRate, lossRate: r.summary.lossRate,
    timeoutRate: r.summary.timeoutRate, meanSecondsToWin: r.summary.meanSecondsToWin,
    meanSecondsSurvived: r.summary.meanSecondsSurvived, meanScore: r.summary.meanScore, byLevel: r.summary.byLevel,
  };
results.sort((a, b) => {
  const ia = order.indexOf(a.spec.id), ib = order.indexOf(b.spec.id);
  return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.spec.id.localeCompare(b.spec.id);
});

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const cats = ['goal', 'threat', 'evade', 'calm'];
const random = {
  best: mean(scenarios.map((s) => s.oracle.best.length / 8)),
  ok: mean(scenarios.map((s) => s.oracle.acceptable.length / 8)),
  fatal: mean(scenarios.map((s) => s.oracle.fatal.length / 8)),
  byCategory: Object.fromEntries(
    cats.map((c) => {
      const xs = scenarios.filter((s) => s.category === c);
      return [c, { n: xs.length, best: mean(xs.map((s) => s.oracle.best.length / 8)), ok: mean(xs.map((s) => s.oracle.acceptable.length / 8)), fatal: mean(xs.map((s) => s.oracle.fatal.length / 8)) }];
    }),
  ),
};

// Showcase: a couple of scenarios per category where the answer matters, with each model's compact-prompt choice.
const rowsByModel = Object.fromEntries(
  results.map((r) => [r.spec.id, new Map(((r.variants.compact ?? {}).rows ?? []).map((row) => [row.s, row]))]),
);
const showcase = [];
for (const c of cats) {
  const pool = scenarios.filter((s) => s.category === c && (s.oracle.fatal.length > 0 || s.oracle.blocked.length > 0));
  const pick = (pool.length >= 2 ? pool : scenarios.filter((s) => s.category === c)).filter((_, i, a) => i % Math.max(1, Math.floor(a.length / 2)) === 0).slice(0, 2);
  for (const s of pick) {
    showcase.push({
      id: s.id, level: s.level, levelName: s.levelName, category: s.category,
      state: s.prompts.compact.state, question: s.prompts.compact.instructions, oracle: s.oracle,
      answers: Object.fromEntries(
        results.map((r) => {
          const row = rowsByModel[r.spec.id].get(s.id);
          return [r.spec.id, row ? { choice: row.choice, ok: row.ok, fatal: row.fatal, best: row.best, top: row.top } : null];
        }),
      ),
    });
  }
}

const levelNames = [...new Set(scenarios.map((s) => `${s.level}|${s.levelName}`))].sort().map((x) => x.split('|')[1]);
const data = {
  generatedAt: new Date().toISOString(),
  device: results.find((r) => r.device)?.device ?? null,
  scenarios: {
    count: scenarios.length,
    categories: Object.fromEntries(cats.map((c) => [c, scenarios.filter((s) => s.category === c).length])),
    tight: scenarios.filter((s) => s.tight).length,
    levels: levelNames,
    random,
  },
  models: results.map((r) => ({
    ...r.spec,
    ranOn: r.runtime,
    loadMs: r.loadMs,
    loadError: r.loadError,
    scenarios: r.scenarios,
    variants: Object.fromEntries(Object.entries(r.variants).map(([k, v]) => [k, v.summary])),
    episodes: episodeSummary(episodes[r.spec.id]) ?? null,
    finishedAt: r.finishedAt,
  })),
  baselines: {
    random: episodeSummary(episodes['@random']) ?? null,
    oracle: episodeSummary(episodes['@oracle']) ?? null,
    episodeSetup: (episodes['@random'] ?? episodes['@oracle'])?.options ?? null,
  },
  showcase,
};
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(data));
console.log(`merge-benchmark: ${data.models.length} models, ${data.scenarios.count} scenarios, ${showcase.length} showcase -> public/data/benchmark.json (${(JSON.stringify(data).length / 1024).toFixed(0)} KB)`);
