import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';

const here = dirname(fileURLToPath(import.meta.url));
const modelsDir = resolve(here, 'local-models');
const benchDir = resolve(here, 'bench-results');

/** Serves ./local-models at /models/* in dev and preview only (never copied into docs/). */
function localModels(): Plugin {
  const types: Record<string, string> = {
    '.json': 'application/json', '.onnx': 'application/octet-stream', '.weights': 'application/octet-stream',
    '.jinja': 'text/plain', '.md': 'text/plain',
  };
  const handler = (req: any, res: any, next: () => void) => {
    const url: string = req.url ?? '';
    if (!url.startsWith('/models/')) return next();
    const rel = decodeURIComponent(url.slice('/models/'.length).split('?')[0]);
    const file = resolve(modelsDir, rel);
    if (!file.startsWith(modelsDir + sep) || !existsSync(file) || !statSync(file).isFile()) return next();
    res.setHeader('Content-Length', statSync(file).size);
    res.setHeader('Content-Type', types[extname(file)] ?? 'application/octet-stream');
    res.setHeader('Cache-Control', 'no-store');
    createReadStream(file).pipe(res);
  };
  return {
    name: 'sf-local-models',
    configureServer: (server) => void server.middlewares.use(handler),
    configurePreviewServer: (server) => void server.middlewares.use(handler),
  };
}

/** Dev-only endpoints the benchmark page uses to persist its results under ./bench-results. */
function benchSink(): Plugin {
  const safe = (n: string) => n.replace(/[^a-zA-Z0-9._-]+/g, '_');
  return {
    name: 'sf-bench-sink',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? '').split('?')[0];
        if (!url.startsWith('/__bench/')) return next();
        const name = url === '/__bench/scenarios' ? 'scenarios.json' : url.startsWith('/__bench/model/') ? `models/${safe(url.slice('/__bench/model/'.length))}.json` : null;
        if (!name) { res.statusCode = 404; return res.end('unknown'); }
        const file = resolve(benchDir, name);
        if (req.method === 'GET') {
          if (!existsSync(file)) { res.statusCode = 404; return res.end('not found'); }
          res.setHeader('Content-Type', 'application/json');
          return res.end(readFileSync(file));
        }
        if (req.method === 'POST') {
          const chunks: Buffer[] = [];
          req.on('data', (c) => chunks.push(c));
          req.on('end', () => {
            mkdirSync(dirname(file), { recursive: true });
            writeFileSync(file, Buffer.concat(chunks));
            res.end('saved');
          });
          return;
        }
        next();
      });
    },
  };
}

const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

// The production build is written to ../docs (served as GitHub Pages / any static host).
// `base: './'` keeps every URL relative so it works from a sub-path like /squirrel_finder_ai/.
export default defineConfig({
  base: './',
  plugins: [localModels(), benchSink()],
  build: {
    outDir: '../docs',
    emptyOutDir: true,
    target: 'es2022',
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 1024,
  },
  // Cross-origin isolation lets ONNX Runtime Web use multithreaded WebAssembly (Laya in the browser).
  // Static hosts get the same headers from public/coi-sw.js, a service worker.
  server: { host: '127.0.0.1', port: 5173, headers: isolation },
  preview: { host: '127.0.0.1', port: 4173, headers: isolation },
});
