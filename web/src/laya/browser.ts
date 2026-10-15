/**
 * Laya running inside the page: ONNX Runtime Web (WASM, or WebGPU for the int4 build) + the typed-decisions
 * checkpoint converted by layaForWeb (https://github.com/vishalmysore/layaForWeb, Apache-2.0).
 *
 * Weights default to https://huggingface.co/VishalMysore/layaForWebTrained and are downloaded once, verified
 * part by part and cached in Cache Storage. Override with ?layaBase=<url ending in /> or localStorage 'sf_laya_base'.
 * Variant: ?laya=q8e8 (default, ~440 MB, WASM) or ?laya=q4e8 (~290 MB, WebGPU when available).
 */
import { Laya } from './laya-core.js';
import type { Decision } from '../ollama';
import { hasWebGPU, loadOrt } from '../ort';

const DEFAULT_BASE = 'https://huggingface.co/VishalMysore/layaForWebTrained/resolve/main/';

interface Manifest {
  variants: Record<string, { label: string; onnx: string; data: { name: string; size: number; sha256: string; parts: string[] } }>;
}

export interface LoadProgress {
  phase: string;
  fraction: number; // 0..1 overall
}

function params() {
  const q = new URLSearchParams(location.search);
  let base = q.get('layaBase');
  try {
    if (base) localStorage.setItem('sf_laya_base', base);
    else base = localStorage.getItem('sf_laya_base');
  } catch { /* private mode */ }
  const b = (base || DEFAULT_BASE).replace(/\/?$/, '/');
  const variant = q.get('laya') === 'q4e8' ? 'q4e8' : 'q8e8';
  return { base: b, variant };
}

async function fetchOk(url: string): Promise<Response> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  return res;
}

/** Download one weight part, via Cache Storage keyed by the manifest hash so reloads are instant. */
async function fetchPart(cache: Cache | null, url: string): Promise<Uint8Array> {
  if (cache) {
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  }
  const res = await fetchOk(url);
  const buf = new Uint8Array(await res.arrayBuffer());
  if (cache) {
    try { await cache.put(url, new Response(buf)); } catch { /* quota: keep going without caching */ }
  }
  return buf;
}

export class LayaBrowser {
  private laya: Laya | null = null;
  private loading: Promise<void> | null = null;
  backend = '';
  variant = '';

  get ready() { return this.laya !== null; }

  load(onProgress: (p: LoadProgress) => void): Promise<void> {
    if (this.laya) return Promise.resolve();
    if (!this.loading) {
      this.loading = this.doLoad(onProgress).catch((e) => {
        this.loading = null; // allow a retry
        throw e;
      });
    }
    return this.loading;
  }

  private async doLoad(report: (p: LoadProgress) => void) {
    const { base, variant } = params();
    this.variant = variant;
    report({ phase: 'Preparing Laya', fraction: 0 });

    const manifest = (await (await fetchOk(base + 'manifest.json')).json()) as Manifest;
    const v = manifest.variants[variant];
    if (!v) throw new Error(`variant ${variant} is not in ${base}manifest.json`);

    const [ort, tokMod, tokJson, tokCfg, cfg] = await Promise.all([
      loadOrt(),
      import('@huggingface/tokenizers'),
      fetchOk(base + 'tokenizer.json').then((r) => r.json()),
      fetchOk(base + 'tokenizer_config.json').then((r) => r.json()),
      fetchOk(base + 'rl_agent_config.json').then((r) => r.json()),
    ]);
    const tokenizer = new (tokMod as any).Tokenizer(tokJson, tokCfg);

    // weights: graph (small) + parts, reassembled into one buffer as external data
    const cache = 'caches' in self ? await caches.open(`laya-${v.data.sha256.slice(0, 16)}`) : null;
    report({ phase: 'Downloading the model', fraction: 0.02 });
    const graph = await fetchPart(cache, base + v.onnx);
    const data = new Uint8Array(v.data.size);
    let off = 0;
    for (let i = 0; i < v.data.parts.length; i++) {
      const part = await fetchPart(cache, base + v.data.parts[i]);
      data.set(part, off);
      off += part.length;
      report({ phase: `Downloading the model (${i + 1}/${v.data.parts.length})`, fraction: 0.02 + 0.9 * ((i + 1) / v.data.parts.length) });
    }
    if (off !== v.data.size) throw new Error(`weights incomplete: ${off} of ${v.data.size} bytes`);

    report({ phase: 'Starting the model', fraction: 0.94 });
    // WebGPU only supports the 2/4-bit MatMulNBits kernels, i.e. the int4 build.
    const wantGpu = variant === 'q4e8' && (await hasWebGPU());
    const session = await ort.InferenceSession.create(graph, {
      executionProviders: wantGpu ? ['webgpu', 'wasm'] : ['wasm'],
      graphOptimizationLevel: 'all',
      externalData: [{ path: v.data.name, data }],
    });
    this.backend = wantGpu ? 'WebGPU' : `WASM ×${ort.env.wasm.numThreads}`;
    const laya = new Laya(ort, session, tokenizer, cfg);
    report({ phase: 'Warming up', fraction: 0.98 });
    await laya.systemOne('warm up', { w: { type: 'choice', instructions: 'pick one', criteria: ['a', 'b'] } });
    this.laya = laya;
    report({ phase: 'Ready', fraction: 1 });
  }

  async decide(state: string, instructions: string, options: Record<string, string>): Promise<Decision> {
    if (!this.laya) throw new Error('Laya is not loaded yet');
    const t0 = performance.now();
    const res = await this.laya.systemOne(state, { move: { type: 'choice', instructions, criteria: options } });
    const a = res.answers.move;
    return {
      choice: a.choice,
      probabilities: a.probabilities,
      confidence: a.confidence,
      latency: (performance.now() - t0) / 1000,
    };
  }
}

export const layaBrowser = new LayaBrowser();
