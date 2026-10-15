/** Dev-only benchmark page: see bench.html. Results are POSTed to the Vite dev server's bench sink. */
import { Game, type BenchScenario } from '../game';
import { backendFor, specFor } from '../models/backends';
import { DIRECTION_KEYS, generateScenarios, percentile, playEpisodes, runModel, summarizeEpisodes, type DecisionRecord, type Variant } from './harness';

const logEl = document.getElementById('log')!;
const status: { state: string; message: string; detail?: unknown } = { state: 'idle', message: '' };
(window as any).__benchStatus = status;

function log(m: string) {
  status.message = m;
  logEl.textContent = `${new Date().toLocaleTimeString()}  ${m}\n${logEl.textContent}`.slice(0, 6000);
  console.log('[bench]', m);
}

async function post(path: string, body: unknown) {
  const r = await fetch(path, { method: 'POST', body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`save failed: ${r.status}`);
}

async function deviceInfo() {
  const info: Record<string, unknown> = {
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated: self.crossOriginIsolated,
    deviceMemoryGB: (navigator as any).deviceMemory ?? null,
  };
  try {
    const a = await (navigator as any).gpu?.requestAdapter();
    if (a) {
      info.gpu = { ...(a.info ? { vendor: a.info.vendor, architecture: a.info.architecture, device: a.info.device, description: a.info.description } : {}) };
      info.webgpuFeatures = [...a.features].filter((f: string) => ['shader-f16', 'subgroups', 'timestamp-query'].includes(f));
    }
  } catch { /* no WebGPU */ }
  return info;
}

function summarize(rows: DecisionRecord[], scenarios: Map<string, BenchScenario>) {
  const ok = rows.filter((r) => !r.err);
  const pct = (xs: DecisionRecord[], f: (r: DecisionRecord) => boolean) => (xs.length ? xs.filter(f).length / xs.length : 0);
  const lat = ok.map((r) => r.ms);
  const cats = ['goal', 'threat', 'evade', 'calm'] as const;
  const byCategory = Object.fromEntries(
    cats.map((c) => {
      const xs = ok.filter((r) => scenarios.get(r.s)?.category === c);
      return [c, { n: xs.length, best: pct(xs, (r) => r.best), ok: pct(xs, (r) => r.ok), fatal: pct(xs, (r) => r.fatal) }];
    }),
  );
  const tight = ok.filter((r) => scenarios.get(r.s)?.tight);
  return {
    n: rows.length,
    errors: rows.length - ok.length,
    best: pct(ok, (r) => r.best),
    ok: pct(ok, (r) => r.ok),
    fatal: pct(ok, (r) => r.fatal),
    blocked: pct(ok, (r) => r.blocked),
    meanConfidence: ok.length ? ok.reduce((a, r) => a + r.conf, 0) / ok.length : 0,
    latencyMs: { mean: lat.length ? lat.reduce((a, b) => a + b, 0) / lat.length : 0, p50: percentile(lat, 50), p95: percentile(lat, 95) },
    byCategory,
    tight: { n: tight.length, ok: pct(tight, (r) => r.ok), blocked: pct(tight, (r) => r.blocked) },
  };
}

async function taskScenarios() {
  status.state = 'running';
  const game = new Game(`${import.meta.env.BASE_URL}art/`);
  const scenarios = await generateScenarios(
    game,
    { episodesPerLevel: 8, maxTicks: 30, explore: 0.3, total: 120, seed: 20261015 },
    log,
  );
  const counts: Record<string, number> = {};
  for (const s of scenarios) counts[s.category] = (counts[s.category] ?? 0) + 1;
  await post('/__bench/scenarios', { generatedAt: new Date().toISOString(), counts, scenarios });
  status.detail = counts;
  log(`saved ${scenarios.length} scenarios ${JSON.stringify(counts)}`);
}

async function taskModel(id: string, variants: Variant[], max: number) {
  status.state = 'running';
  const spec = specFor(id);
  if (!spec) throw new Error(`unknown model ${id}`);
  const file = await (await fetch('/__bench/scenarios')).json();
  let scenarios: BenchScenario[] = file.scenarios;
  if (max < scenarios.length) {
    const stride = scenarios.length / max;
    scenarios = Array.from({ length: max }, (_, i) => scenarios[Math.floor(i * stride)]);
  }
  const index = new Map(scenarios.map((s) => [s.id, s]));
  const backend = backendFor(id);
  const device = await deviceInfo();

  log(`loading ${spec.label}…`);
  const t0 = performance.now();
  let loadError: string | undefined;
  try {
    await backend.load((p) => {
      status.message = `${spec.label}: ${p.phase} ${Math.round(p.fraction * 100)}%`;
    });
  } catch (e) {
    loadError = (e as Error).message;
    log(`LOAD FAILED: ${loadError}`);
  }
  const loadMs = Math.round(performance.now() - t0);

  // keep variants and the cold-load measurement from earlier runs of this model
  let ex: any = null;
  try {
    const r = await fetch(`/__bench/model/${id}`);
    if (r.ok) ex = await r.json();
  } catch { /* first run */ }
  const result: Record<string, unknown> = {
    spec, device: ex?.device ?? device, loadMs: ex?.loadMs ?? loadMs, loadError: loadError ?? null,
    runtime: ex?.runtime ?? backend.runtime, scenarios: scenarios.length, finishedAt: null,
    variants: { ...(ex?.variants ?? {}) },
  };
  if (!loadError) {
    log(`${spec.label} loaded in ${(loadMs / 1000).toFixed(1)}s on ${backend.runtime}`);
    for (const v of variants) {
      log(`${spec.label} · ${v} prompt: 0/${scenarios.length}`);
      const rows = await runModel(backend, scenarios, v, (d, n) => {
        if (d % 10 === 0 || d === n) status.message = `${spec.label} · ${v}: ${d}/${n}`;
      });
      (result.variants as any)[v] = { summary: summarize(rows, index), rows };
      log(`${spec.label} · ${v}: ok ${(100 * (result.variants as any)[v].summary.ok).toFixed(0)}%`);
    }
  }
  result.finishedAt = new Date().toISOString();
  await post(`/__bench/model/${id}`, result);
  status.detail = Object.fromEntries(Object.entries(result.variants as any).map(([k, v]: any) => [k, v.summary]));
  log(`saved ${id}`);
}

async function taskEpisodes(id: string, variant: Variant, levels: number[], per: number, maxDecisions: number) {
  status.state = 'running';
  const game = new Game(`${import.meta.env.BASE_URL}art/`);
  const opts = { levels, per, maxDecisions, seed: 424242 };
  let policy: (g: Game) => Promise<string>;
  let label = id;
  let backend = null as ReturnType<typeof backendFor> | null;
  if (id === '@random') {
    policy = async () => DIRECTION_KEYS[Math.floor(Math.random() * DIRECTION_KEYS.length)];
  } else if (id === '@oracle') {
    policy = async (g) => g.benchSnapshot('x', 0).oracle.best[0];
  } else {
    backend = backendFor(id);
    label = backend.spec.label;
    log(`loading ${label}…`);
    const t0 = performance.now();
    await backend.load((p) => { status.message = `${label}: ${p.phase} ${Math.round(p.fraction * 100)}%`; });
    log(`${label} loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    const compact = variant !== 'full';
    const keys = DIRECTION_KEYS;
    const desc: Record<string, string> = {
      up: 'Walk straight up', up_right: 'Walk diagonally up and to the right', right: 'Walk straight to the right',
      down_right: 'Walk diagonally down and to the right', down: 'Walk straight down',
      down_left: 'Walk diagonally down and to the left', left: 'Walk straight to the left', up_left: 'Walk diagonally up and to the left',
    };
    const natural: Record<string, string> = {
      up: 'walk up', up_right: 'walk up and to the right', right: 'walk right', down_right: 'walk down and to the right',
      down: 'walk down', down_left: 'walk down and to the left', left: 'walk left', up_left: 'walk up and to the left',
    };
    const options = keys.map((k) => ({ key: k, description: desc[k], label: variant === 'natural' ? natural[k] : undefined }));
    policy = async (g) => {
      const p = g.buildPrompt(compact);
      return (await backend!.decide(p.state, p.instructions, options)).choice;
    };
  }
  log(`${label}: playing ${levels.length * per} episodes (levels ${levels.join(',')})…`);
  const t1 = performance.now();
  const results = await playEpisodes(game, policy, opts, (d, n) => { status.message = `${label}: episode ${d}/${n}`; });
  const summary = summarizeEpisodes(results);
  const record = { id, label, variant, options: opts, wallMs: Math.round(performance.now() - t1), summary, results };
  await post(`/__bench/model/episodes_${id.replace(/[^a-zA-Z0-9]+/g, '_')}`, record);
  status.detail = summary;
  log(`${label}: won ${Math.round(summary.winRate * 100)}% lost ${Math.round(summary.lossRate * 100)}% timeout ${Math.round(summary.timeoutRate * 100)}%`);
}

async function main() {
  const q = new URLSearchParams(location.search);
  try {
    const task = q.get('task');
    if (task === 'scenarios') await taskScenarios();
    else if (task === 'episodes') {
      await taskEpisodes(
        q.get('id') ?? '@random',
        (q.get('variant') ?? 'compact') as Variant,
        (q.get('levels') ?? '1,3,5').split(',').map(Number),
        Number(q.get('per') ?? 5),
        Number(q.get('max') ?? 40),
      );
    } else if (task === 'model') {
      await taskModel(
        q.get('id') ?? 'laya-web',
        (q.get('variants') ?? 'compact').split(',') as Variant[],
        Number(q.get('n') ?? 120),
      );
    } else {
      log('idle: pass ?task=scenarios or ?task=model&id=…');
      return;
    }
    status.state = 'done';
  } catch (e) {
    status.state = 'error';
    log(`ERROR: ${(e as Error).message}`);
    console.error(e);
  }
}

void main();
