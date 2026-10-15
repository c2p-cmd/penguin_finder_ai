import type { Decision } from '../ollama';

export type { Decision };

export interface LoadProgress {
  phase: string;
  fraction: number; // 0..1 overall
}

export interface OptionSpec {
  key: string; // e.g. "up_right"
  description: string; // e.g. "Walk diagonally up and to the right"
  /** text the model reads for this option; defaults to `key` (the benchmark's "natural" variant sets it) */
  label?: string;
}

/** Everything the intro page and the benchmark need to know about a model, without loading it. */
export interface ModelSpec {
  id: string;
  label: string;
  family: string;
  baseModel: string;
  params: string;
  sizeMB: number;
  runtime: string;
  needsWebGPU: boolean;
  /** true = loadable straight from the public site; false = needs local weights or a local server */
  hosted: boolean;
  source: string;
  license: string;
  note: string;
  /** small encoders get the short, dominant-axis-first scene description; LLM-style models get the full one */
  compactPrompt: boolean;
}

export interface ModelBackend {
  readonly spec: ModelSpec;
  readonly ready: boolean;
  /** what it actually ran on after loading, e.g. "WebGPU" or "WASM ×4" */
  readonly runtime: string;
  load(onProgress: (p: LoadProgress) => void): Promise<void>;
  decide(state: string, instructions: string, options: OptionSpec[]): Promise<Decision>;
  dispose(): Promise<void>;
}

export function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** 1 - normalised entropy, the same "confidence" Laya and Decision Lab report. */
export function confidenceOf(p: number[]): number {
  const k = p.length;
  if (k < 2) return 1;
  let ent = 0;
  for (const v of p) ent -= v * Math.log(Math.min(Math.max(v, 1e-12), 1));
  return Math.min(Math.max(1 - ent / Math.log(k), 0), 1);
}
