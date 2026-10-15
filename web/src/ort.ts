/**
 * ONNX Runtime Web (1.30, JSEP build: WASM + WebGPU execution providers), loaded from vendor/ort at runtime so the
 * 28 MB wasm is only fetched when an in-browser model is used. Shared by Laya and Julia-1.
 */
let ortPromise: Promise<any> | null = null;

export function loadOrt(): Promise<any> {
  ortPromise ??= (async () => {
    const dir = new URL(`${import.meta.env.BASE_URL}vendor/ort/`, location.href).href;
    const ort = await import(/* @vite-ignore */ `${dir}ort.min.mjs`);
    ort.env.wasm.wasmPaths = dir;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    return ort;
  })();
  return ortPromise;
}

export async function hasWebGPU(): Promise<boolean> {
  try {
    return !!(navigator as any).gpu && !!(await (navigator as any).gpu.requestAdapter());
  } catch {
    return false;
  }
}

/** Streaming download with progress and Cache Storage persistence (big single files such as Julia's 577 MB weights). */
export async function fetchBytes(
  url: string,
  opts: { cacheName: string; onProgress?: (fraction: number) => void },
): Promise<Uint8Array> {
  const cache = 'caches' in self ? await caches.open(opts.cacheName) : null;
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      opts.onProgress?.(1);
      return buf;
    }
  }
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  const total = Number(res.headers.get('content-length') ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) opts.onProgress?.(got / total);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  if (cache) {
    try { await cache.put(url, new Response(out)); } catch { /* quota exceeded: continue uncached */ }
  }
  opts.onProgress?.(1);
  return out;
}
