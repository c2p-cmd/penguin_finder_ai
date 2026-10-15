/**
 * Ollama client for decision models (port of macos/.../OllamaClient.swift).
 * Decision-only models reject chat/generate, so they are driven through /v1/systemone.
 *
 * The browser talks to Ollama directly, so Ollama must allow this page's origin:
 *   OLLAMA_ORIGINS="https://<you>.github.io" ollama serve        (localhost / 127.0.0.1 are allowed by default)
 * Override the server with ?ollama=http://host:11434 (remembered in localStorage).
 */
const DEFAULT_BASE = 'http://127.0.0.1:11434';

// ?ollama=http://host:11434 is read once at startup and remembered; the intro page can change it afterwards
try {
  const q = new URLSearchParams(location.search).get('ollama');
  if (q) localStorage.setItem('sf_ollama', q.replace(/\/$/, ''));
} catch { /* ignore */ }

function base(): string {
  try {
    return (localStorage.getItem('sf_ollama') ?? DEFAULT_BASE).replace(/\/$/, '');
  } catch {
    return DEFAULT_BASE;
  }
}

export const ollamaDefaultBase = DEFAULT_BASE;
export const ollamaBase = base;

/** Persist a new server URL (null resets to the default). Returns the normalised URL in effect. */
export function setOllamaBase(url: string | null): string {
  try {
    if (url === null) localStorage.removeItem('sf_ollama');
    else localStorage.setItem('sf_ollama', url.replace(/\/$/, ''));
  } catch { /* storage blocked: the change lasts only until reload via ?ollama= */ }
  return url === null ? DEFAULT_BASE : url.replace(/\/$/, '');
}

export interface Decision {
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
  latency: number; // seconds
}

async function send<T>(url: string, init?: RequestInit, timeoutMs = 90_000): Promise<T> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(url, { ...init, signal: ctl.signal });
  } catch {
    throw new Error(`Ollama isn't reachable at ${base()}. Is it running, and does OLLAMA_ORIGINS include ${location.origin}?`);
  } finally {
    clearTimeout(timer);
  }
  let body: any = null;
  try { body = await res.json(); } catch { /* handled below */ }
  if (!res.ok) throw new Error(body?.error ?? `HTTP ${res.status}`);
  if (body === null) throw new Error('Unexpected reply from Ollama');
  return body as T;
}

export async function listModels(): Promise<string[]> {
  const body = await send<{ data?: { id: string }[] }>(`${base()}/v1/models`, undefined, 5_000);
  return (body.data ?? []).map((m) => m.id);
}

/** Ask `model` to pick one of `options` (key -> description) given `state`. */
export async function decide(
  model: string,
  state: string,
  instructions: string,
  options: Record<string, string>,
): Promise<Decision> {
  const t0 = performance.now();
  const body = await send<any>(`${base()}/v1/systemone`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      state,
      questions: { move: { type: 'choice', instructions, criteria: options } },
    }),
  });
  const move = body?.answers?.move;
  if (!move || typeof move.choice !== 'string') throw new Error('Unexpected reply from Ollama');
  return {
    choice: move.choice,
    probabilities: move.probabilities ?? {},
    confidence: move.confidence ?? 0,
    latency: (performance.now() - t0) / 1000,
  };
}
