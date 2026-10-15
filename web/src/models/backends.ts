/**
 * One adapter per model family. Every adapter maps the game's decision (state text + question + 8 direction
 * options) onto that model's own API and returns the same Decision shape.
 */
import { listModels, decide as ollamaDecide } from '../ollama';
import { layaBrowser } from '../laya/browser';
import { fetchBytes, hasWebGPU, loadOrt } from '../ort';
import { JuliaWebGPU } from './julia-core.js';
import { confidenceOf, softmax } from './types';
import type { Decision, LoadProgress, ModelBackend, ModelSpec, OptionSpec } from './types';

const BASE = import.meta.env.BASE_URL;

function paramsOf(name: string, fallback: string): string {
  try {
    return new URLSearchParams(location.search).get(name) || localStorage.getItem(`sf_${name}`) || fallback;
  } catch {
    return fallback;
  }
}

const labelOf = (o: OptionSpec) => o.label ?? o.key;

/** Map an answer keyed by option labels back to option keys. */
function toKeys(d: Decision, options: OptionSpec[]): Decision {
  const byLabel = new Map(options.map((o) => [labelOf(o), o.key]));
  return {
    ...d,
    choice: byLabel.get(d.choice) ?? d.choice,
    probabilities: Object.fromEntries(Object.entries(d.probabilities).map(([k, v]) => [byLabel.get(k) ?? k, v])),
  };
}

function finish(keys: string[], probs: number[], t0: number): Decision {
  const top = probs.indexOf(Math.max(...probs));
  return {
    choice: keys[top],
    probabilities: Object.fromEntries(keys.map((k, i) => [k, probs[i]])),
    confidence: confidenceOf(probs),
    latency: (performance.now() - t0) / 1000,
  };
}

// ---------------------------------------------------------------- Laya (layaForWeb, ONNX Runtime Web)
export const LAYA_SPEC: ModelSpec = {
  id: 'laya-web',
  label: 'Laya (typed decisions)',
  family: 'Laya',
  baseModel: 'ModernBERT-large + 2-layer head',
  params: '421M',
  sizeMB: 440,
  runtime: 'ONNX Runtime Web · WASM (int8)',
  needsWebGPU: false,
  hosted: true,
  source: 'https://huggingface.co/VishalMysore/layaForWebTrained',
  license: 'Apache-2.0',
  note: 'Unofficial browser port of convaiinnovations/laya-typed-decisions by layaForWeb. English only, 512 tokens.',
  compactPrompt: true,
};

class LayaBackend implements ModelBackend {
  readonly spec = LAYA_SPEC;
  get ready() { return layaBrowser.ready; }
  get runtime() { return layaBrowser.backend || this.spec.runtime; }
  async load(p: (x: LoadProgress) => void) {
    await layaBrowser.load(p);
  }
  async decide(state: string, instructions: string, options: OptionSpec[]) {
    const d = await layaBrowser.decide(state, instructions, Object.fromEntries(options.map((o) => [labelOf(o), ''])));
    return toKeys(d, options);
  }
  async dispose() { /* kept alive: reloading costs a minute */ }
}

// ---------------------------------------------------------------- Julia-1 (SupersonicLabs, WebGPU)
export const JULIA_SPEC: ModelSpec = {
  id: 'julia-1',
  label: 'Julia-1',
  family: 'Julia',
  baseModel: 'mmBERT-small finetune',
  params: '~140M',
  sizeMB: 617,
  runtime: 'ONNX Runtime Web · WebGPU (fp32)',
  needsWebGPU: true,
  hosted: true,
  source: 'https://huggingface.co/SupersonicLabs/Julia-1-ONNX',
  license: 'Apache-2.0',
  note: 'Multilingual encoder, 1024-token context, own Rust-WASM tokenizer. WebGPU only.',
  compactPrompt: true,
};

class JuliaBackend implements ModelBackend {
  readonly spec = JULIA_SPEC;
  private julia: JuliaWebGPU | null = null;
  private session: any = null;
  get ready() { return this.julia !== null; }
  get runtime() { return 'WebGPU'; }

  async load(progress: (x: LoadProgress) => void) {
    if (this.julia) return;
    if (!(await hasWebGPU())) throw new Error('Julia-1 needs a browser with WebGPU');
    const hf = paramsOf('juliaBase', 'https://huggingface.co/SupersonicLabs/Julia-1-ONNX/resolve/main/').replace(/\/?$/, '/');
    progress({ phase: 'Preparing Julia-1', fraction: 0 });
    const ort = await loadOrt();
    const vendor = new URL(`${BASE}vendor/julia/`, location.href).href;
    const glue = await import(/* @vite-ignore */ `${vendor}julia_webgpu_encode.js`);
    await glue.default({ module_or_path: new URL(`${vendor}julia_webgpu_encode_bg.wasm`) });
    progress({ phase: 'Downloading the tokenizer', fraction: 0.02 });
    const tokenizerJson = await (await fetch(`${hf}tokenizer.json`)).text();
    const encoder = new glue.WasmEncoder(tokenizerJson);
    const graph = await fetchBytes(`${hf}model.onnx`, { cacheName: 'julia-1-graph' });
    const weights = await fetchBytes(`${hf}model.onnx.data`, {
      cacheName: 'julia-1-weights',
      onProgress: (f) => progress({ phase: 'Downloading the model', fraction: 0.05 + 0.85 * f }),
    });
    progress({ phase: 'Starting the model (WebGPU)', fraction: 0.92 });
    this.session = await ort.InferenceSession.create(graph, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
      externalData: [{ path: 'model.onnx.data', data: weights }],
    });
    this.julia = new JuliaWebGPU(ort, this.session, null, encoder, 1024, 256, true);
    progress({ phase: 'Warming up', fraction: 0.97 });
    await this.julia.warm();
    progress({ phase: 'Ready', fraction: 1 });
  }

  async decide(state: string, instructions: string, options: OptionSpec[]): Promise<Decision> {
    if (!this.julia) throw new Error('Julia-1 is not loaded');
    const keys = options.map((o) => o.key);
    const t0 = performance.now();
    const [logits] = await this.julia.logits([{ state, question: instructions, options: options.map(labelOf), type: 'choice' }]);
    return finish(keys, softmax(logits), t0);
  }

  async dispose() {
    try { await this.session?.release?.(); } catch { /* ignore */ }
    this.julia = null;
    this.session = null;
  }
}

// ---------------------------------------------------------------- Kev-0.6B browser-use (open-jev + transformers.js)
export const KEV_SPEC: ModelSpec = {
  id: 'kev-0.6b-bu',
  label: 'Kev-0.6B browser-use',
  family: 'Kev',
  baseModel: 'Qwen3-0.6B-Base (fine-tuned)',
  params: '0.6B',
  sizeMB: 361,
  runtime: 'Transformers.js · WebGPU (q4f16)',
  needsWebGPU: true,
  hosted: true,
  source: 'https://huggingface.co/arbazsiddiqui/kev-0.6b-browser-use-ONNX',
  license: 'Apache-2.0',
  note: 'Fine-tuned on Mind2Web / WebChain to choose page elements; used here through open-jev. 8192-token context.',
  compactPrompt: true,
};

class KevBackend implements ModelBackend {
  readonly spec = KEV_SPEC;
  private jev: any = null;
  private choiceFn: any = null;
  get ready() { return this.jev !== null; }
  get runtime() { return 'WebGPU'; }

  async load(progress: (x: LoadProgress) => void) {
    if (this.jev) return;
    progress({ phase: 'Preparing Kev-0.6B', fraction: 0 });
    const mod: any = await import('open-jev');
    this.choiceFn = mod.choice;
    const model = paramsOf('kevModel', 'arbazsiddiqui/kev-0.6b-browser-use-ONNX');
    this.jev = await mod.OpenJev.load({
      model,
      dtype: 'q4f16',
      onProgress: (e: { progress?: number }) => {
        const f = typeof e?.progress === 'number' ? e.progress : 0;
        progress({ phase: 'Downloading the model', fraction: 0.02 + 0.93 * f });
      },
    });
    progress({ phase: 'Warming up', fraction: 0.97 });
    await this.jev.decide('warm up', { w: this.choiceFn('pick one', ['a', 'b']) });
    progress({ phase: 'Ready', fraction: 1 });
  }

  async decide(state: string, instructions: string, options: OptionSpec[]): Promise<Decision> {
    if (!this.jev) throw new Error('Kev is not loaded');
    const keys = options.map((o) => o.key);
    const labels = options.map(labelOf);
    const t0 = performance.now();
    const res = await this.jev.decide(state, { move: this.choiceFn(instructions, labels) });
    const a = res.move;
    const probs = labels.map((k) => Number(a.probabilities?.[k] ?? 0));
    const sum = probs.reduce((x, y) => x + y, 0) || 1;
    return finish(keys, probs.map((v) => v / sum), t0);
  }

  async dispose() {
    try { await this.jev?.dispose?.(); } catch { /* ignore */ }
    this.jev = null;
  }
}

// ---------------------------------------------------------------- Decision Lab V3 INT4 (350M, WebGPU)
export const DECISION_LAB_SPEC: ModelSpec = {
  id: 'decision-lab',
  label: 'Decision Lab V3',
  family: 'Decision Lab',
  baseModel: 'LFM2-350M (fine-tuned)',
  params: '350M',
  sizeMB: 296,
  runtime: 'ONNX Runtime Web · WebGPU (int4)',
  needsWebGPU: true,
  hosted: false,
  source: 'https://github.com/khalilelghoul01/decision-lab',
  license: 'MIT code · LFM Open License v1.0 weights',
  note: 'Typed-decision research model with a shared-context cache. Needs shader-f16. Weights are one 296 MB file, so they are served locally (see README).',
  compactPrompt: true,
};

class DecisionLabBackend implements ModelBackend {
  readonly spec = DECISION_LAB_SPEC;
  private model: any = null;
  private choiceFn: any = null;
  get ready() { return this.model !== null; }
  get runtime() { return 'WebGPU'; }

  async load(progress: (x: LoadProgress) => void) {
    if (this.model) return;
    if (!(await hasWebGPU())) throw new Error('Decision Lab needs a browser with WebGPU (shader-f16)');
    progress({ phase: 'Preparing Decision Lab', fraction: 0 });
    // Dev/preview only: the SDK (and its pinned ONNX Runtime) stays out of the production bundle because
    // the 296 MB weights are a single file that can't be hosted on GitHub Pages. See the README.
    if (!import.meta.env.DEV) {
      throw new Error('Decision Lab is only wired up in the dev build (its 296 MB weights are served from web/local-models)');
    }
    const sdk: any = await import('decision-lab-sdk');
    this.choiceFn = sdk.choice;
    const dir = new URL(`${BASE}models/ort-dl/`, location.href).href;
    const modelUrl = new URL(paramsOf('dlBase', `${BASE}models/model-v3-q4`), location.href).href;
    this.model = await sdk.loadModel({
      modelUrl,
      wasmPaths: { mjs: `${dir}ort-wasm-simd-threaded.asyncify.mjs`, wasm: `${dir}ort-wasm-simd-threaded.asyncify.wasm` },
      onProgress: (e: { progress?: number; file?: string }) => {
        const f = typeof e?.progress === 'number' ? e.progress : 0;
        progress({ phase: `Loading ${e?.file ?? 'model'}`, fraction: 0.05 + 0.85 * f });
      },
    });
    progress({ phase: 'Ready', fraction: 1 });
  }

  async decide(state: string, instructions: string, options: OptionSpec[]): Promise<Decision> {
    if (!this.model) throw new Error('Decision Lab is not loaded');
    const keys = options.map((o) => o.key);
    const labels = options.map(labelOf);
    const t0 = performance.now();
    const res = await this.model.systemOne(
      { state, questions: { move: this.choiceFn(instructions, Object.fromEntries(labels.map((k) => [k, null]))) } },
      { mode: 'fast' },
    );
    const a = res.answers.move;
    const probs = labels.map((k) => Number(a.probabilities?.[k] ?? 0));
    const sum = probs.reduce((x, y) => x + y, 0) || 1;
    return finish(keys, probs.map((v) => v / sum), t0);
  }

  async dispose() {
    try { await this.model?.dispose?.(); } catch { /* ignore */ }
    this.model = null;
  }
}

// ---------------------------------------------------------------- Ollama (local server, any pulled model)
export function ollamaSpec(tag: string): ModelSpec {
  const isLaya = tag.startsWith('laya');
  return {
    id: `ollama:${tag}`,
    label: `Ollama · ${tag}`,
    family: isLaya ? 'Laya (Ollama)' : 'Ollama',
    baseModel: tag,
    params: tag.match(/(\d+(?:\.\d+)?[mb])/i)?.[1]?.toUpperCase() ?? '?',
    sizeMB: 0,
    runtime: 'Ollama server (MLX)',
    needsWebGPU: false,
    hosted: false,
    source: 'https://ollama.com',
    license: 'model-specific',
    note: 'Runs in the local Ollama server, outside the browser.',
    compactPrompt: isLaya,
  };
}

class OllamaBackend implements ModelBackend {
  readonly spec: ModelSpec;
  private readonly tag: string;
  constructor(tag: string) {
    this.tag = tag;
    this.spec = ollamaSpec(tag);
  }
  get ready() { return true; }
  get runtime() { return 'Ollama'; }
  async load(p: (x: LoadProgress) => void) { p({ phase: 'Ready', fraction: 1 }); }
  async decide(state: string, instructions: string, options: OptionSpec[]) {
    return ollamaDecide(this.tag, state, instructions, Object.fromEntries(options.map((o) => [o.key, o.description])));
  }
  async dispose() { /* nothing to free */ }
}

// ---------------------------------------------------------------- registry
export const BROWSER_SPECS: ModelSpec[] = [LAYA_SPEC, JULIA_SPEC, KEV_SPEC, DECISION_LAB_SPEC];

const cache = new Map<string, ModelBackend>();

export function backendFor(id: string): ModelBackend {
  let b = cache.get(id);
  if (b) return b;
  if (id === LAYA_SPEC.id) b = new LayaBackend();
  else if (id === JULIA_SPEC.id) b = new JuliaBackend();
  else if (id === KEV_SPEC.id) b = new KevBackend();
  else if (id === DECISION_LAB_SPEC.id) b = new DecisionLabBackend();
  else if (id.startsWith('ollama:')) b = new OllamaBackend(id.slice('ollama:'.length));
  else throw new Error(`unknown model "${id}"`);
  cache.set(id, b);
  return b;
}

export async function ollamaModelIds(): Promise<string[]> {
  return (await listModels()).map((t) => `ollama:${t}`);
}

export function specFor(id: string): ModelSpec | null {
  const s = BROWSER_SPECS.find((m) => m.id === id);
  if (s) return s;
  if (id.startsWith('ollama:')) return ollamaSpec(id.slice('ollama:'.length));
  return null;
}
