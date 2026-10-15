/**
 * The landing page: what this is, why it exists, how the models compare (rendered from data/benchmark.json),
 * and the play buttons (human or agentic).
 */
import { BROWSER_SPECS, ollamaModelIds, specFor } from './models/backends';
import { ollamaBase, ollamaDefaultBase, setOllamaBase } from './ollama';
import type { ModelSpec } from './models/types';

type Cat = 'goal' | 'threat' | 'evade' | 'calm';
interface Summary {
  n: number;
  errors: number;
  best: number;
  ok: number;
  fatal: number;
  blocked: number;
  meanConfidence: number;
  latencyMs: { mean: number; p50: number; p95: number };
  byCategory: Record<Cat, { n: number; best: number; ok: number; fatal: number }>;
  tight: { n: number; ok: number; blocked: number };
}
interface EpisodeSummary {
  variant: string;
  n: number;
  winRate: number;
  lossRate: number;
  timeoutRate: number;
  meanSecondsToWin: number;
  meanSecondsSurvived: number;
  meanScore: number;
}
type Variant = 'compact' | 'full' | 'natural';
interface BenchModel extends ModelSpec {
  ranOn: string;
  loadMs: number;
  loadError: string | null;
  scenarios: number;
  variants: Partial<Record<Variant, Summary>>;
  episodes: EpisodeSummary | null;
}
interface Showcase {
  id: string;
  level: number;
  levelName: string;
  category: Cat;
  state: string;
  question: string;
  oracle: { best: string[]; acceptable: string[]; fatal: string[]; blocked: string[] };
  answers: Record<string, { choice: string; ok: boolean; fatal: boolean; best: boolean } | null>;
}
export interface Benchmark {
  generatedAt: string;
  device: { userAgent?: string; hardwareConcurrency?: number; gpu?: { vendor?: string; architecture?: string; description?: string } } | null;
  scenarios: {
    count: number;
    categories: Record<Cat, number>;
    tight: number;
    levels: string[];
    random: { best: number; ok: number; fatal: number; byCategory: Record<Cat, { n: number; best: number; ok: number; fatal: number }> };
  };
  models: BenchModel[];
  baselines?: { random: EpisodeSummary | null; oracle: EpisodeSummary | null; episodeSetup: { levels: number[]; per: number; maxDecisions: number } | null };
  showcase: Showcase[];
}

export interface PlayChoice {
  mode: 'human' | 'agentic';
  model: string; // '' for human
}

const ARROW: Record<string, string> = {
  up: '↑', up_right: '↗', right: '→', down_right: '↘', down: '↓', down_left: '↙', left: '←', up_left: '↖',
};
const CAT_LABEL: Record<Cat, string> = {
  goal: 'Reach the penguin', threat: 'Penguin + ghost close', evade: 'No penguin yet, ghosts near', calm: 'Nothing close',
};

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K, props: { class?: string; text?: string; html?: string; attrs?: Record<string, string> } = {}, ...kids: (Node | string | null)[]
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (props.class) e.className = props.class;
  if (props.text !== undefined) e.textContent = props.text;
  if (props.html !== undefined) e.innerHTML = props.html; // only ever called with static strings from this file
  for (const [k, v] of Object.entries(props.attrs ?? {})) e.setAttribute(k, v);
  for (const k of kids) if (k !== null) e.append(k);
  return e;
}

const pct = (x: number) => `${Math.round(x * 100)}%`;
const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

// ------------------------------------------------------------------ charts (Plotly, vendored, loaded on demand)
interface ChartSpec {
  title: string;
  note: string;
  summary: string; // the same numbers as text, for screen readers and the load-failure fallback
  rows: number;
  traces: number;
  data: Record<string, unknown>[];
  xTitle?: string;
  pct?: boolean;
  log?: boolean;
  heat?: boolean;
  legend?: boolean;
}

let plotlyPromise: Promise<any> | null = null;
function loadPlotly(): Promise<any> {
  if ((window as any).Plotly) return Promise.resolve((window as any).Plotly);
  plotlyPromise ??= new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = `${import.meta.env.BASE_URL}vendor/plotly/plotly-cartesian.min.js`;
    s.async = true;
    s.onload = () => ((window as any).Plotly ? resolve((window as any).Plotly) : reject(new Error('Plotly missing')));
    s.onerror = () => { plotlyPromise = null; reject(new Error('Plotly failed to load')); };
    document.head.append(s);
  });
  return plotlyPromise;
}

const PLOT_CONFIG = { displayModeBar: false, responsive: true, staticPlot: false };
const chartHeight = (s: ChartSpec) => Math.round(70 + s.rows * (s.traces * 17 + 12) + (s.heat ? 40 : 0) + (s.legend === false ? 0 : 30));

function plotLayout(s: ChartSpec): Record<string, unknown> {
  const grid = 'rgba(255,255,255,0.09)';
  return {
    height: chartHeight(s),
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#eef1fb', family: 'ui-rounded, "SF Pro Rounded", system-ui, -apple-system, "Segoe UI", sans-serif', size: 13 },
    margin: { l: 8, r: 16, t: s.legend === false || s.heat ? 8 : 36, b: s.heat ? 8 : 48 },
    barmode: 'group',
    bargap: 0.28,
    showlegend: s.legend !== false && !s.heat,
    legend: { orientation: 'h', x: 0, y: 1.0, yanchor: 'bottom' },
    hoverlabel: { bgcolor: '#141826', bordercolor: 'rgba(255,255,255,0.3)', font: { color: '#eef1fb' } },
    xaxis: s.heat
      ? { side: 'top', fixedrange: true, automargin: true, tickangle: 0, tickfont: { size: 12 } }
      : {
          title: { text: s.xTitle ?? '' },
          fixedrange: true,
          gridcolor: grid,
          zeroline: false,
          ...(s.pct ? { range: [0, 100], ticksuffix: '%' } : {}),
          ...(s.log ? { type: 'log' } : {}),
        },
    yaxis: { autorange: 'reversed', fixedrange: true, automargin: true, gridcolor: grid, ticklabelstandoff: 6 },
  };
}

export async function loadBenchmark(): Promise<Benchmark | null> {
  try {
    const r = await fetch(`${import.meta.env.BASE_URL}data/benchmark.json`, { cache: 'no-cache' });
    if (!r.ok) return null;
    return (await r.json()) as Benchmark;
  } catch {
    return null;
  }
}

async function weightsReachable(spec: ModelSpec): Promise<boolean> {
  if (spec.hosted) return true;
  try {
    const r = await fetch(`${import.meta.env.BASE_URL}models/model-v3-q4/manifest.json`, { method: 'HEAD' });
    return r.ok && !(r.headers.get('content-type') ?? '').includes('html');
  } catch {
    return false;
  }
}

export class Intro {
  private root: HTMLElement;
  private variant: Variant = 'compact';
  private selected = '';
  private mode: 'human' | 'agentic' = 'human';
  private resultsHost!: HTMLElement;
  private playBtn!: HTMLButtonElement;
  private modelHost!: HTMLElement;
  private plotToken = 0;
  private engine: 'loading' | 'ready' | 'error' = 'loading';
  private engineMsg = '';
  private engineNote!: HTMLElement;
  private plotWatch: IntersectionObserver | null = null;
  private plotObservers: ResizeObserver[] = [];
  private resumeBtn!: HTMLButtonElement;
  private heroPlay!: HTMLAnchorElement;
  private newRunNote!: HTMLElement;
  private resumable = false;
  onPlay: (c: PlayChoice) => void = () => {};
  onResume: () => void = () => {};
  /** The Ollama server URL changed: refresh anything that lists its models. */
  onOllamaChange: () => void = () => {};
  private serverHost!: HTMLElement;

  get canResume() { return this.resumable; }

  /** The game engine loads behind the page; Play stays disabled until it is ready. */
  setEngine(state: 'loading' | 'ready' | 'error', msg = '') {
    this.engine = state;
    this.engineMsg = msg;
    if (this.engineNote) {
      this.engineNote.hidden = state === 'ready';
      this.engineNote.textContent = state === 'error' ? msg : 'Loading the game engine…';
    }
    this.updatePlay();
  }
  /** A run is in progress: offer to go back to it instead of only starting over. */
  setResumable(on: boolean) {
    this.resumable = on;
    if (this.resumeBtn) this.resumeBtn.hidden = !on;
    if (this.newRunNote) this.newRunNote.hidden = !on;
    if (this.heroPlay) {
      this.heroPlay.textContent = on ? 'Start a new run' : 'Play now';
      this.heroPlay.classList.toggle('primary', !on);
    }
  }

  constructor(root: HTMLElement, private data: Benchmark | null) {
    this.root = root;
  }

  show() { this.root.hidden = false; this.root.scrollTop = 0; document.body.classList.add('intro-open'); }
  hide() { this.root.hidden = true; document.body.classList.remove('intro-open'); }

  async render() {
    const r = this.root;
    r.innerHTML = '';
    this.resumeBtn = h('button', { class: 'btn primary', text: 'Resume game', attrs: { type: 'button' } });
    this.resumeBtn.hidden = !this.resumable;
    this.resumeBtn.addEventListener('click', () => this.onResume());
    this.heroPlay = h('a', { class: 'btn', attrs: { href: '#play' } });
    this.setResumable(this.resumable);
    r.append(
      h('header', { class: 'hero' },
        h('p', { class: 'eyebrow', text: 'A small research experiment' }),
        h('h1', { text: 'Squirrel Finder' }),
        h('p', { class: 'lede', text: 'Can a tiny decision model that runs entirely inside your browser play a video game?' }),
        h('div', { class: 'cta-row' },
          this.resumeBtn,
          this.heroPlay,
          h('a', { class: 'btn ghost', text: 'See the model comparison', attrs: { href: '#compare' } }),
        ),
      ),

      h('section', { class: 'block', attrs: { id: 'what' } },
        h('h2', { text: 'What is this?' }),
        h('div', { class: 'cols' },
          h('div', {}, h('p', {
            html: 'A night-time arcade game. A wizard walks a haunted graveyard and then six dungeon rooms, dodging drifting ghosts ' +
              'and trying to reach a penguin before they catch him. Seven levels, a score with near-miss bonuses, and a lantern that lights the way.',
          }), h('p', {
            html: 'You can play it with the keyboard, or hand the controls to a <strong>System One</strong> model: a small network that is shown a short ' +
              'description of the scene plus eight possible directions, scores each option in <em>one forward pass</em>, and the game takes the winner. ' +
              'Nothing is generated token by token; the model can only choose from the options we give it.',
          })),
          h('div', { class: 'aside panel' },
            h('h3', { text: 'One typed decision' }),
            h('dl', { class: 'kv' },
              h('dt', { text: 'state' }), h('dd', { text: '"The penguin is 280 px down and 40 px right. Ghost 1 is 90 px to the left and is coming at the wizard."' }),
              h('dt', { text: 'question' }), h('dd', { text: '"Which way should the wizard walk to reach the penguin?"' }),
              h('dt', { text: 'options' }), h('dd', { text: 'up · up_right · right · down_right · down · down_left · left · up_left' }),
              h('dt', { text: 'answer' }), h('dd', { text: 'one forward pass, a probability for each option; the game takes the highest' }),
            ),
          ),
        ),
      ),

      h('section', { class: 'block', attrs: { id: 'why' } },
        h('h2', { text: 'Why create this?' }),
        h('p', {
          html: 'Typed-decision ("System One") models are small, fast classifiers that sit between a rules engine and a chat model. ' +
            'Several open ones now ship with ONNX weights and a browser runtime, which raises a practical question: ' +
            '<strong>can they actually be loaded from a static web page, with no server, and are they good enough to drive a real-time game?</strong>',
        }),
        h('p', {
          html: 'This project is a small, honest test of that. We took the open models we could find, ran each one on WebGPU or WebAssembly in the browser, ' +
            'recorded real decision points from the game, and scored every answer against a lookahead oracle. The results below are measured, not claimed.',
        }),
      ),

      this.modelsBlock(),
      this.compareBlock(),
      await this.playBlock(),

      h('footer', { class: 'foot' },
        h('p', { html: 'Built with Three.js (WebGPU), ONNX Runtime Web, Transformers.js and Plotly.' }),
        h('h3', { text: 'Credits' }),
        h('ul', { class: 'credits' },
          h('li', { html: '<a href="https://www.flaticon.com/free-icons/penguin" title="penguin icons" target="_blank" rel="noopener">Penguin icons created by G-CAT - Flaticon</a> (site icon).' }),
          h('li', { html: 'Browser inference: <a href="https://github.com/vishalmysore/layaForWeb" target="_blank" rel="noopener">layaForWeb</a> and <a href="https://huggingface.co/VishalMysore/layaForWebTrained" target="_blank" rel="noopener">layaForWebTrained</a> (Apache-2.0); <a href="https://huggingface.co/SupersonicLabs/Julia-1-ONNX" target="_blank" rel="noopener">Julia 1</a> by Supersonic Labs (Apache-2.0).' }),
          h('li', { html: 'Libraries: <a href="https://threejs.org" target="_blank" rel="noopener">three.js</a>, <a href="https://onnxruntime.ai" target="_blank" rel="noopener">ONNX Runtime Web</a>, <a href="https://github.com/huggingface/transformers.js" target="_blank" rel="noopener">Transformers.js</a> and <a href="https://plotly.com/javascript/" target="_blank" rel="noopener">Plotly.js</a> (MIT / Apache-2.0).' }),
          h('li', { html: 'Characters (the wizard and the ghosts): pixel art from <a href="https://freepixel.art/" target="_blank" rel="noopener">FreePixel</a>, free for personal and commercial use under the <a href="https://freepixel.art/license/" target="_blank" rel="noopener">FreePixel asset licence</a>.' }),
          h('li', { html: 'Dungeon rooms: <a href="https://maxparata.itch.io/tinyvoxeldungeon" target="_blank" rel="noopener">Tiny Voxel Dungeon</a> by <strong>Max Parata</strong>, free to use in any project, commercial or not.' }),
          h('li', { html: 'Graveyard tiles and props: <a href="https://nonokaatm.itch.io/dark-horror-asset-pack-16x16" target="_blank" rel="noopener">Dark Horror Asset Pack 16x16</a> by nonokaatm, free to use with attribution. The penguin sprite was generated with AI by the author.' }),
          h('li', { html: 'Library licence texts: <a href="./licenses/plotly-LICENSE.txt">Plotly</a>, <a href="./licenses/three-LICENSE.txt">three.js</a>, <a href="./licenses/onnxruntime-web-LICENSE.txt">ONNX Runtime Web</a>, <a href="./licenses/transformers-js-LICENSE.txt">Transformers.js</a>, <a href="./licenses/layaForWeb-LICENSE.txt">layaForWeb</a>.' }),
        ),
      ),
    );
  }

  // ------------------------------------------------------------------ models
  private modelsBlock(): HTMLElement {
    const sec = h('section', { class: 'block', attrs: { id: 'models' } }, h('h2', { text: 'The models' }),
      h('p', { class: 'muted', text: 'Everything below runs in the page: no inference server. Download sizes are one-off; the browser caches them.' }));
    const grid = h('div', { class: 'model-grid' });
    for (const m of BROWSER_SPECS) {
      grid.append(
        h('article', { class: 'panel model' },
          h('div', { class: 'model-head' }, h('h3', { text: m.label }), h('span', { class: 'pill', text: m.params })),
          h('p', { class: 'muted small', text: m.baseModel }),
          h('dl', {},
            h('dt', { text: 'Download' }), h('dd', { text: `${m.sizeMB} MB` }),
            h('dt', { text: 'Runs on' }), h('dd', { text: m.runtime }),
            h('dt', { text: 'Licence' }), h('dd', { text: m.license }),
          ),
          h('p', { class: 'small', text: m.note }),
          h('a', { class: 'small link', text: 'Model page ↗', attrs: { href: m.source, target: '_blank', rel: 'noopener' } }),
        ),
      );
    }
    sec.append(grid);
    return sec;
  }

  // ------------------------------------------------------------------ comparison
  private compareBlock(): HTMLElement {
    const sec = h('section', { class: 'block', attrs: { id: 'compare' } }, h('h2', { text: 'How they compare' }));
    const d = this.data;
    if (!d) {
      sec.append(h('p', { class: 'muted', text: 'No benchmark data found (data/benchmark.json). Run the benchmark and rebuild.' }));
      return sec;
    }
    sec.append(
      h('p', {
        html: `We captured <strong>${d.scenarios.count} decision points</strong> by playing seeded headless runs of the real game across all ${d.scenarios.levels.length} levels ` +
          `(${d.scenarios.categories.goal} reach-the-penguin, ${d.scenarios.categories.threat} with a ghost closing in, ${d.scenarios.categories.evade} with no penguin yet, ${d.scenarios.categories.calm} calm). ` +
          'For each one an <strong>oracle</strong> simulates all eight directions for 0.7 s with the game’s own collision rules. A move counts as <em>best</em> if it matches the oracle’s top direction, ' +
          '<em>acceptable</em> if it is safe and makes at least half the best progress, and <em>fatal</em> if it walks into a ghost.',
      }),
    );

    const toggle = h('div', { class: 'seg' });
    const variants: [Variant, string][] = [
      ['compact', 'Short prompt'],
      ['natural', 'Short prompt, plain English'],
      ['full', 'Full prompt'],
    ];
    for (const [v, label] of variants) {
      const b = h('button', { text: label, attrs: { type: 'button', 'aria-pressed': String(this.variant === v) } });
      b.classList.toggle('on', this.variant === v);
      b.addEventListener('click', () => {
        this.variant = v;
        [...toggle.children].forEach((c, i) => {
          c.classList.toggle('on', variants[i][0] === v);
          c.setAttribute('aria-pressed', String(variants[i][0] === v));
        });
        this.renderResults();
      });
      toggle.append(b);
    }
    sec.append(h('div', { class: 'toolbar' }, h('span', { class: 'muted small', text: 'Prompt style:' }), toggle));
    this.resultsHost = h('div', { class: 'results' });
    sec.append(this.resultsHost);
    this.renderResults();

    sec.append(
      h('details', { class: 'method' },
        h('summary', { text: 'How we measured' }),
        h('ul', {},
          h('li', { text: 'Scenarios: 8 seeded episodes per level; a noisy oracle policy plays on, and a stratified sample of its decision points is kept (more where the choice matters).' }),
          h('li', { text: 'Oracle: for each of the 8 directions, simulate 0.7 s (two decision intervals) with ghosts on their current headings and the real walls; pick the safest direction that closes most distance to the penguin.' }),
          h('li', { text: 'Short prompt: a one-line question and a two-sentence scene (dominant axis first). Full prompt: the long description with walls, free distances and up to six ghosts.' }),
          h('li', { text: 'Plain-English options: the same short scene, but the options read "walk up and to the right" instead of up_right.' }),
          h('li', { text: 'Every model sees identical scenarios, one decision at a time, on this machine. Latency is wall-clock per decision after warm-up, in the page.' }),
          h('li', { text: 'Wins in play: 18 complete headless runs (levels 1, 3 and 5, six seeded starts each, at most 40 decisions of 0.35 s) using each model\'s best prompt style from the scenario test. The world is frozen while a model thinks, so this measures decision quality, not speed. With 18 runs one win is 5.6 points, so differences under ~15 points are noise.' }),
          h('li', { text: 'Random baseline: picking uniformly among the eight directions.' }),
          h('li', { text: d.device?.gpu ? `Measured on: ${d.device.gpu.architecture || d.device.gpu.vendor || 'GPU'} (${d.device.hardwareConcurrency ?? '?'} threads), ${new Date(d.generatedAt).toLocaleDateString()}.` : `Generated ${new Date(d.generatedAt).toLocaleDateString()}.` }),
        ),
      ),
    );

    if (d.showcase.length) sec.append(this.showcaseBlock());
    return sec;
  }

  private renderResults() {
    const d = this.data!;
    const v = this.variant;
    const host = this.resultsHost;
    host.innerHTML = '';
    this.plotObservers.forEach((o) => o.disconnect());
    this.plotObservers = [];
    const token = ++this.plotToken;
    const rows = d.models.filter((m) => m.variants[v]).sort((a, b) => b.variants[v]!.ok - a.variants[v]!.ok);
    const rnd = d.scenarios.random;
    const sv = (m: BenchModel) => m.variants[v]!;
    const p1 = (x: number) => Math.round(x * 1000) / 10; // fraction -> percent, one decimal

    // the oracle in one line, right above the charts that score against it
    host.append(h('p', {
      class: 'muted small oracle-note',
      html: '<strong>The oracle</strong> is the referee, not a player: at each decision point it tries all eight directions for 0.7 s in the real game and keeps the safest one that closes most distance to the penguin. Every model is scored against it.',
    }));

    const baselineNames = ['Random baseline', ...(d.baselines?.oracle ? ['Oracle'] : [])];
    const specs: ChartSpec[] = [];

    // 1. single-move quality: best / acceptable / walks into a ghost
    {
      const names = [...rows.map((m) => m.label), ...baselineNames];
      const best = [...rows.map((m) => p1(sv(m).best)), p1(rnd.best), ...(d.baselines?.oracle ? [100] : [])];
      const ok = [...rows.map((m) => p1(sv(m).ok)), p1(rnd.ok), ...(d.baselines?.oracle ? [100] : [])];
      const fatal = [...rows.map((m) => p1(sv(m).fatal)), p1(rnd.fatal), ...(d.baselines?.oracle ? [0] : [])];
      specs.push({
        title: 'Single decisions',
        note: 'Share of decision points where the model picked the oracle’s best move, an acceptable move, or walked into a ghost.',
        summary: names.map((n, i) => `${n}: best ${best[i]}%, acceptable ${ok[i]}%, walks into a ghost ${fatal[i]}%`).join('. '),
        rows: names.length,
        traces: 3,
        data: [
          { type: 'bar', orientation: 'h', name: 'Best move', y: names, x: best, marker: { color: '#46c878' }, hovertemplate: '%{y}<br>Best move: %{x}%<extra></extra>' },
          { type: 'bar', orientation: 'h', name: 'Acceptable', y: names, x: ok, marker: { color: '#3cbebe' }, hovertemplate: '%{y}<br>Acceptable: %{x}%<extra></extra>' },
          { type: 'bar', orientation: 'h', name: 'Walks into a ghost', y: names, x: fatal, marker: { color: '#f05a5a' }, hovertemplate: '%{y}<br>Walks into a ghost: %{x}%<extra></extra>' },
        ],
        xTitle: 'Share of decisions (%)',
        pct: true,
      });
    }

    // 2. whole runs
    {
      const played = d.models.filter((m) => m.episodes).sort((a, b) => b.episodes!.winRate - a.episodes!.winRate);
      const names = [...played.map((m) => m.label), ...baselineNames];
      const eps = [...played.map((m) => m.episodes!), d.baselines?.random, ...(d.baselines?.oracle ? [d.baselines.oracle] : [])];
      const win = eps.map((e) => (e ? p1(e.winRate) : 0));
      const custom = eps.map((e) => (e ? `${e.n} runs · caught ${pct(e.lossRate)} · timed out ${pct(e.timeoutRate)}` : 'no data'));
      if (played.length) {
        specs.push({
          title: 'Wins when playing whole runs',
          note: `Complete headless runs. With ${eps[0]?.n ?? 18} runs, one win is ${p1(1 / (eps[0]?.n ?? 18))} points, so gaps under ~15 points are noise.`,
          summary: names.map((n, i) => `${n}: won ${win[i]}% of runs`).join('. '),
          rows: names.length,
          traces: 1,
          data: [{
            type: 'bar', orientation: 'h', name: 'Wins', y: names, x: win, customdata: custom,
            marker: { color: names.map((n) => (baselineNames.includes(n) ? '#8b91a8' : '#9db7ff')) },
            hovertemplate: '%{y}<br>Won %{x}% of runs<br>%{customdata}<extra></extra>',
          }],
          xTitle: 'Runs won (%)',
          pct: true,
          legend: false,
        });
      }
    }

    // 3. situations (heatmap)
    {
      const cats: Cat[] = ['goal', 'threat', 'evade', 'calm'];
      const SHORT: Record<Cat, string> = { goal: 'Reach<br>the penguin', threat: 'Penguin +<br>ghost close', evade: 'No penguin,<br>ghosts near', calm: 'Nothing<br>close' };
      const xs = cats.map((c) => `${SHORT[c]}<br>n=${d.scenarios.categories[c]}`);
      const names = [...rows.map((m) => m.label), 'Random baseline'];
      const z = [...rows.map((m) => cats.map((c) => p1(sv(m).byCategory[c]?.ok ?? 0))), cats.map((c) => p1(rnd.byCategory[c]?.ok ?? 0))];
      specs.push({
        title: 'Acceptable moves by situation',
        note: 'Where models do well or badly: reaching the penguin, a ghost closing in, no penguin yet, or nothing close.',
        summary: names.map((n, i) => `${n}: ${cats.map((c, j) => `${CAT_LABEL[c]} ${z[i][j]}%`).join(', ')}`).join('. '),
        rows: names.length,
        traces: 1.6,
        heat: true,
        data: [{
          type: 'heatmap', x: xs, y: names, z, zmin: 0, zmax: 100,
          colorscale: [[0, '#7a2a1e'], [0.5, '#8a8a22'], [1, '#2a9d55']],
          xgap: 2, ygap: 2,
          texttemplate: '%{z}%', textfont: { color: '#ffffff', size: 13 },
          hovertemplate: '%{y}<br>%{x}<br>Acceptable: %{z}%<extra></extra>',
          colorbar: { ticksuffix: '%', thickness: 10, len: 0.8, outlinewidth: 0, tickfont: { color: '#eef1fb' } },
        }],
      });
    }

    // 4. speed
    {
      const names = rows.map((m) => m.label);
      const p50 = rows.map((m) => Math.round(sv(m).latencyMs.p50));
      const p95 = rows.map((m) => Math.round(sv(m).latencyMs.p95));
      specs.push({
        title: 'Decision latency',
        note: 'Wall-clock time per decision after warm-up. The game asks every 350 ms of play. Log scale.',
        summary: names.map((n, i) => `${n}: median ${fmtMs(p50[i])}, 95th percentile ${fmtMs(p95[i])}`).join('. '),
        rows: names.length,
        traces: 2,
        data: [
          { type: 'bar', orientation: 'h', name: 'Median', y: names, x: p50, marker: { color: '#9db7ff' }, hovertemplate: '%{y}<br>Median: %{x} ms<extra></extra>' },
          { type: 'bar', orientation: 'h', name: '95th percentile', y: names, x: p95, marker: { color: '#5d6aa8' }, hovertemplate: '%{y}<br>95th percentile: %{x} ms<extra></extra>' },
        ],
        xTitle: 'Milliseconds per decision',
        log: true,
      });
    }

    // 5. load time (in-browser models only)
    {
      const local = rows.filter((m) => !m.id.startsWith('ollama:'));
      if (local.length) {
        const names = local.map((m) => m.label);
        const sec = local.map((m) => Math.round(m.loadMs / 100) / 10);
        specs.push({
          title: 'Load time',
          note: 'Time to get an in-browser model ready on this machine. Download sizes are in The models above.',
          summary: names.map((n, i) => `${n}: ${sec[i]} seconds`).join('. '),
          rows: names.length,
          traces: 1,
          data: [{ type: 'bar', orientation: 'h', name: 'Load time', y: names, x: sec, marker: { color: '#9db7ff' }, hovertemplate: '%{y}<br>%{x} s<extra></extra>' }],
          xTitle: 'Seconds',
          legend: false,
        });
      }
    }

    const grid = h('div', { class: 'charts' });
    const plots: { el: HTMLElement; spec: ChartSpec }[] = [];
    for (const spec of specs) {
      const el = h('div', { class: 'plot', attrs: { role: 'img', 'aria-label': `${spec.title}. ${spec.summary}` } });
      el.style.minHeight = `${chartHeight(spec)}px`;
      grid.append(h('figure', { class: 'chart panel' }, h('h3', { text: spec.title }), h('p', { class: 'muted small', text: spec.note }), el));
      plots.push({ el, spec });
    }
    host.append(grid);

    // plain-language takeaways computed from the data
    const best = rows[0];
    const fastest = [...rows].sort((a, b) => sv(a).latencyMs.p50 - sv(b).latencyMs.p50)[0];
    const safest = [...rows].sort((a, b) => sv(a).fatal - sv(b).fatal)[0];
    const beatsRandom = rows.filter((m) => sv(m).ok > rnd.ok + 0.05).length;
    const played = d.models.filter((m) => m.episodes);
    const winners = played.filter((m) => m.episodes!.winRate > (d.baselines?.random?.winRate ?? 0) + 0.1);
    const bestPlay = [...played].sort((a, b) => b.episodes!.winRate - a.episodes!.winRate)[0];
    const items = [
      h('li', { html: `<strong>${best.label}</strong> makes an acceptable move on single decisions ${pct(sv(best).ok)} of the time (random: ${pct(rnd.ok)}).` }),
      h('li', { text: `${beatsRandom} of ${rows.length} models beat random by more than 5 points on acceptable moves with this prompt style.` }),
      h('li', { html: `Fastest: <strong>${fastest.label}</strong> at ${fmtMs(sv(fastest).latencyMs.p50)} per decision (the game asks every 0.35 s of play).` }),
      h('li', { html: `Fewest single-decision walks into a ghost: <strong>${safest.label}</strong> (${pct(sv(safest).fatal)}; random: ${pct(rnd.fatal)}).` }),
    ];
    if (bestPlay) {
      items.push(
        h('li', { html: `<strong>Playing whole runs is much harder than scoring single moves:</strong> the best model, <strong>${bestPlay.label}</strong>, won ${pct(bestPlay.episodes!.winRate)} of 18 runs; the random baseline won ${pct(d.baselines?.random?.winRate ?? 0)} and the lookahead oracle ${pct(d.baselines?.oracle?.winRate ?? 0)}. ` +
          `${winners.length} of ${played.length} models won clearly more often than random.` }),
      );
    }
    host.append(h('ul', { class: 'takeaways' }, ...items));

    const draw = () => void loadPlotly().then((Plotly) => {
      if (token !== this.plotToken) return;
      for (const { el, spec } of plots) {
        Plotly.react(el, spec.data, plotLayout(spec), PLOT_CONFIG);
        // the intro can be hidden or resized when a chart is drawn; keep it fitted
        const ro = new ResizeObserver(() => { if (el.clientWidth > 0) Plotly.Plots.resize(el); });
        ro.observe(el);
        this.plotObservers.push(ro);
      }
    }).catch(() => {
      if (token !== this.plotToken) return;
      for (const { el, spec } of plots) {
        el.replaceChildren(h('p', { class: 'muted small', text: `Chart unavailable. ${spec.summary}` }));
      }
    });
    // the 1.4 MB chart library loads only when the charts are about to scroll into view
    this.plotWatch?.disconnect();
    if (!('IntersectionObserver' in window)) { draw(); return; }
    this.plotWatch = new IntersectionObserver((entries, obs) => {
      if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); draw(); }
    }, { root: this.root, rootMargin: '600px 0px' });
    this.plotWatch.observe(grid);
  }

  private showcaseBlock(): HTMLElement {
    const d = this.data!;
    const wrap = h('div', { class: 'showcase' }, h('h3', { text: 'Look at some real decisions' }),
      h('p', { class: 'muted small', text: '✓ best = the oracle’s top move · ○ ok = acceptable · ✕ fatal = walks into a ghost · – safe = safe but not useful. Colors repeat the same four states.' }));
    const models = d.models.filter((m) => !m.id.startsWith('ollama:nimble') && !m.id.startsWith('ollama:kev'));
    for (const s of d.showcase.slice(0, 6)) {
      const answers = h('div', { class: 'answers' });
      for (const m of models) {
        const a = s.answers[m.id];
        if (!a) continue;
        const cls = a.fatal ? 'bad' : a.best ? 'best' : a.ok ? 'ok' : 'meh';
        const verdict = a.fatal ? '✕ fatal' : a.best ? '✓ best' : a.ok ? '○ ok' : '– safe';
        answers.append(h('span', { class: `ans ${cls}` }, h('span', { class: 'who', text: m.label.replace('Ollama · ', 'Ollama ') }), h('span', { class: 'arrow', text: ARROW[a.choice] ?? '?' }), h('span', { class: 'verdict', text: verdict })));
      }
      wrap.append(
        h('article', { class: 'panel scn' },
          h('div', { class: 'scn-head' }, h('span', { class: 'pill', text: CAT_LABEL[s.category] }), h('span', { class: 'muted small', text: s.levelName })),
          h('p', { class: 'state', text: s.state }),
          h('p', { class: 'muted small', text: `Oracle: best ${s.oracle.best.map((k) => ARROW[k]).join(' ')} · acceptable ${s.oracle.acceptable.map((k) => ARROW[k]).join(' ')}${s.oracle.fatal.length ? ' · fatal ' + s.oracle.fatal.map((k) => ARROW[k]).join(' ') : ''}` }),
          answers,
        ),
      );
    }
    return wrap;
  }

  // ------------------------------------------------------------------ play
  private async playBlock(): Promise<HTMLElement> {
    const sec = h('section', { class: 'block play', attrs: { id: 'play' } }, h('h2', { text: 'Play' }));
    const modes = h('div', { class: 'modes' });
    const human = h('button', { class: 'mode on', attrs: { type: 'button', 'aria-pressed': 'true' } },
      h('span', { class: 'big', text: '🧙' }), h('strong', { text: 'Human mode' }), h('span', { class: 'muted small', text: 'Arrow keys or WASD. Beat the models.' }));
    const agent = h('button', { class: 'mode', attrs: { type: 'button', 'aria-pressed': 'false' } },
      h('span', { class: 'big', text: '🤖' }), h('strong', { text: 'Agentic mode' }), h('span', { class: 'muted small', text: 'Pick a model and watch it play.' }));
    modes.append(human, agent);
    this.modelHost = h('div', { class: 'model-pick', hidden: true } as never);
    this.modelHost.hidden = true;
    this.serverHost = this.serverBlock();
    this.serverHost.hidden = true;
    this.playBtn = h('button', { class: 'btn primary big', text: 'Play as human', attrs: { type: 'button' } });

    const setMode = (m: 'human' | 'agentic') => {
      this.mode = m;
      human.classList.toggle('on', m === 'human');
      agent.classList.toggle('on', m === 'agentic');
      human.setAttribute('aria-pressed', String(m === 'human'));
      agent.setAttribute('aria-pressed', String(m === 'agentic'));
      this.modelHost.hidden = m === 'human';
      this.serverHost.hidden = m === 'human';
      this.updatePlay();
    };
    human.addEventListener('click', () => setMode('human'));
    agent.addEventListener('click', () => setMode('agentic'));
    this.playBtn.addEventListener('click', () => {
      this.onPlay({ mode: this.mode, model: this.mode === 'agentic' ? this.selected : '' });
    });

    this.newRunNote = h('p', { class: 'muted small note', text: 'Starting a new run replaces the one in progress.' });
    this.newRunNote.hidden = !this.resumable;
    this.engineNote = h('p', { class: 'muted small note', attrs: { role: 'status' } });
    this.engineNote.hidden = this.engine === 'ready';
    this.engineNote.textContent = this.engine === 'error' ? this.engineMsg : 'Loading the game engine…';
    sec.append(modes, this.serverHost, this.modelHost, h('div', { class: 'cta-row' }, this.playBtn), this.engineNote, this.newRunNote);
    // checking which weights are reachable takes a few requests: don't hold the page for it
    void this.fillModelPicker();
    return sec;
  }

  /** Ollama server field: lets visitors point the page at their own server without editing the URL. */
  private serverBlock(): HTMLElement {
    const input = h('input', { class: 'server-input', attrs: { type: 'url', id: 'ollama-url', placeholder: ollamaDefaultBase, autocomplete: 'off', spellcheck: 'false', inputmode: 'url' } });
    input.value = ollamaBase();
    const status = h('span', { class: 'muted small server-status', attrs: { role: 'status' } });
    const save = h('button', { class: 'btn', text: 'Connect', attrs: { type: 'button' } });
    const reset = h('button', { class: 'btn', text: 'Reset', attrs: { type: 'button' } });

    const apply = async (url: string | null) => {
      let value = url;
      if (value !== null) {
        value = value.trim();
        if (!/^https?:\/\/[^\s/]+/i.test(value)) {
          status.textContent = 'Enter a full address, e.g. http://127.0.0.1:11434';
          return;
        }
        try { new URL(value); } catch { status.textContent = 'That doesn’t look like a valid address.'; return; }
      }
      const now = setOllamaBase(value);
      input.value = now;
      status.textContent = 'Connecting…';
      save.disabled = reset.disabled = true;
      try {
        const ids = await ollamaModelIds();
        status.textContent = ids.length ? `Connected · ${ids.length} model${ids.length === 1 ? '' : 's'} found` : 'Connected, but no models are pulled yet.';
      } catch (e) {
        const insecure = location.protocol === 'https:' && /^http:\/\//i.test(now) && !/^http:\/\/(127\.0\.0\.1|localhost|\[::1\])/i.test(now);
        status.textContent = insecure
          ? 'Can’t reach it: this page is HTTPS, so the browser blocks a plain http:// server unless it is localhost. Use https:// or run the page locally.'
          : (e as Error).message;
      } finally {
        save.disabled = reset.disabled = false;
      }
      this.onOllamaChange();
      await this.fillModelPicker();
    };
    save.addEventListener('click', () => void apply(input.value));
    reset.addEventListener('click', () => void apply(null));
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') void apply(input.value); });

    return h('div', { class: 'server panel' },
      h('label', { class: 'server-label', attrs: { for: 'ollama-url' } }, h('strong', { text: 'Ollama server' }),
        h('span', { class: 'muted small', text: ' · optional, for models you run yourself' })),
      h('div', { class: 'server-row' }, input, save, reset),
      status,
      h('p', { class: 'muted small', text: 'Ollama must allow this site: start it with OLLAMA_ORIGINS set to this page’s address. localhost and 127.0.0.1 are allowed by default.' }),
    );
  }

  private async fillModelPicker() {
    const host = this.modelHost;
    host.innerHTML = '';
    const hasGpu = !!(navigator as any).gpu;
    const specs: ModelSpec[] = [...BROWSER_SPECS];
    try {
      for (const id of await ollamaModelIds()) {
        const s = specFor(id);
        if (s) specs.push(s);
      }
    } catch { /* no Ollama running: fine */ }

    const best = this.data?.models
      .filter((m) => m.variants.compact)
      .sort((a, b) => (b.episodes?.winRate ?? 0) - (a.episodes?.winRate ?? 0) || (b.variants.compact?.ok ?? 0) - (a.variants.compact?.ok ?? 0))
      .filter((m) => BROWSER_SPECS.some((x) => x.id === m.id))[0]?.id;
    for (const s of specs) {
      const reachable = await weightsReachable(s);
      const disabledWhy = s.needsWebGPU && !hasGpu ? 'Browser lacks WebGPU' : !reachable ? 'Model files aren’t on this site' : '';
      const id = `m-${s.id.replace(/[^a-z0-9]/gi, '_')}`;
      const input = h('input', { attrs: { type: 'radio', name: 'model', id, value: s.id } });
      if (disabledWhy) input.disabled = true;
      input.addEventListener('change', () => { this.selected = s.id; this.updatePlay(); });
      const label = h('label', { class: `pick${disabledWhy ? ' off' : ''}`, attrs: { for: id } }, input,
        h('span', { class: 'pick-main' },
          h('strong', { text: s.label }),
          h('span', { class: 'muted small', text: s.id.startsWith('ollama:') ? ' · local Ollama server' : ` · ${s.sizeMB} MB download · ${s.runtime}` }),
        ),
        disabledWhy ? h('span', { class: 'pill warn', text: disabledWhy }) : (s.id === best ? h('span', { class: 'pill good', text: 'Most wins in our runs' }) : null),
      );
      host.append(label);
      if (!this.selected && !disabledWhy) {
        this.selected = s.id;
        input.checked = true;
      }
    }
    host.append(h('p', { class: 'muted small', text: 'The first time you pick an in-browser model it downloads its files (progress shows in the game’s top bar); after that it is cached. The game pauses until it is ready.' }));
    this.updatePlay();
  }

  private updatePlay() {
    if (!this.playBtn) return;
    const ready = this.engine === 'ready';
    if (this.mode === 'human') {
      this.playBtn.textContent = 'Play as human';
      this.playBtn.disabled = !ready;
    } else {
      const s = specFor(this.selected);
      this.playBtn.textContent = s ? `Watch ${s.label} play` : 'Pick a model';
      this.playBtn.disabled = !s || !ready;
    }
  }
}
