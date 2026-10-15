// Types for the vendored layaForWeb core (laya-core.js, Apache-2.0).
export class Laya {
  constructor(ort: unknown, session: unknown, tokenizer: unknown, cfg: unknown);
  systemOne(
    state: unknown,
    questions: Record<string, { type: 'choice' | 'score' | 'noul'; instructions: unknown; criteria?: unknown }>,
  ): Promise<{
    model: string;
    answers: Record<string, any>;
    usage: { input_tokens: number; output_tokens: number };
    latency_ms: number;
  }>;
}
