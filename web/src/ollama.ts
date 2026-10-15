/**
 * Ollama client for decision models (port of macos/.../OllamaClient.swift).
 * Decision-only models reject chat/generate, so they are driven through /v1/systemone.
 *
 * The browser talks to Ollama directly, so Ollama must allow this page's origin:
 *   OLLAMA_ORIGINS="https://<you>.github.io" ollama serve        (localhost / 127.0.0.1 are allowed by default)
 * Override the server with ?ollama=http://host:11434 (remembered in localStorage).
 */
const DEFAULT_BASE = 'http://127.0.0.1:11434';

function base(): string {
  try {
    const q = new URLSearchParams(location.search).get('ollama');
    if (q) {
      localStorage.setItem('sf_ollama', q);
      return q.replace(/\/$/, '');
    }
    return (localStorage.getItem('sf_ollama') ?? DEFAULT_BASE).replace(/\/$/, '');
  } catch {
    return DEFAULT_BASE;
  }
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
    throw new Error(`Ollama isn't reachable at ${base()} (is it running, and does OLLAMA_ORIGINS allow this page?)`);
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
