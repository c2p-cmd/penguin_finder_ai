// Types for the vendored Julia-1 runtime (julia-core.js, Apache-2.0).
export class JuliaWebGPU {
  constructor(
    ort: unknown,
    session: unknown,
    tokenizer: unknown,
    encoder: { encode(requestJson: string, maxLength: number, headLength: number, strict: boolean): string } | null,
    maxLength: number,
    headLength: number,
    strictEncoding: boolean,
  );
  warm(): Promise<this>;
  logits(rows: { state: unknown; question: string; options: string[]; type?: string }[]): Promise<number[][]>;
  predict(rows: { state: unknown; question: string; options: string[]; type?: string }[]): Promise<{ index: number; probabilities: number[] }[]>;
}
