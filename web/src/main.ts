import './style.css';
import './intro.css';
import { Intro, loadBenchmark } from './intro';
import { Game } from './game';
import { Hud } from './hud';
import type { Renderer } from './render';
import { backendFor } from './models/backends';

/**
 * Static hosts (GitHub Pages) can't send COOP/COEP headers, which multithreaded WASM needs. public/coi-sw.js adds
 * them; the page reloads once so the service worker controls it. Guarded so it can never loop.
 */
async function ensureIsolation(): Promise<boolean> {
  if (self.crossOriginIsolated || !window.isSecureContext || !('serviceWorker' in navigator)) return false;
  if (sessionStorage.getItem('sf_coi_reload')) return false;
  try {
    await navigator.serviceWorker.register(`${import.meta.env.BASE_URL}coi-sw.js`);
    await navigator.serviceWorker.ready; // resolves once the worker is active
    sessionStorage.setItem('sf_coi_reload', '1'); // at most one reload per tab session
    location.reload();
    return true;
  } catch {
    return false; // single-threaded WASM still works
  }
}

// A redeploy replaces hashed chunk files, so a tab (or cached index.html) from the previous build can 404 on a
// lazy import. Reload once to pick up the new build instead of leaving a dead feature.
window.addEventListener('vite:preloadError', (e) => {
  try {
    if (sessionStorage.getItem('sf_chunk_reload')) return;
    sessionStorage.setItem('sf_chunk_reload', '1');
  } catch { /* storage blocked: fall through to the in-page message */ return; }
  e.preventDefault();
  location.reload();
});
setTimeout(() => { try { sessionStorage.removeItem('sf_chunk_reload'); } catch { /* ignore */ } }, 15_000);

async function main() {
  if (await ensureIsolation()) return;
  const params = new URLSearchParams(location.search);
  const assetBase = `${import.meta.env.BASE_URL}art/`;
  const canvas = document.getElementById('view') as HTMLCanvasElement;

  // ?level=3 / ?player=<model> / ?play=1 skip the intro and go straight into the game
  const direct = params.has('level') || params.has('player') || params.has('play');
  const level = Math.max(1, Number(params.get('level') ?? 1) || 1) - 1;
  const player = params.get('player') ?? '';

  const game = new Game(assetBase);
  const hud = new Hud(game);

  // The landing page is static content, so it is interactive immediately. The engine (three.js, WebGPU, the level)
  // loads only when someone presses Play (or opens a ?play=/?level=/?player= link), which keeps first load light.
  const intro = new Intro(document.getElementById('intro')!, await loadBenchmark());
  let renderer: Renderer | null = null;
  let engine: Promise<boolean> | null = null;
  const startEngine = (lvl: number, pl: string) => (engine ??= (async () => {
    try {
      const { Renderer: R } = await import('./render');
      renderer = new R(canvas, game, assetBase);
      await renderer.init();
      await game.start(lvl, pl);
    } catch (e) {
      console.error(e);
      const msg = `Couldn't start the renderer: ${(e as Error).message}`;
      intro.setEngine('error', msg);
      const box = document.getElementById('fatal')!;
      box.hidden = false;
      box.textContent = msg;
      return false;
    }
    if (import.meta.env.DEV) (window as any).__backend = backendFor; // console experiments with prompt formats
    // While the intro is open the world is frozen, so redraw only occasionally and keep the GPU free.
    let lastIdle = 0;
    const loop = (t: number) => {
      if (!game.paused || t - lastIdle > 400) {
        renderer!.frame();
        lastIdle = t;
      }
      hud.update(renderer!.framesPerSecond, renderer!.backend);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
    return true;
  })());

  intro.onIntent = () => void import('./render'); // warm the engine chunk when someone hovers Play
  intro.onPlay = async ({ mode, model }) => {
    intro.setEngine('loading');
    const ok = await startEngine(0, '');
    if (!ok) return;
    intro.setEngine('ready');
    intro.setResumable(false);
    intro.hide();
    game.paused = false;
    void game.newRun(mode === 'agentic' ? model : '');
  };
  intro.onOllamaChange = () => void game.loadModels(true);
  intro.onResume = () => {
    intro.hide();
    game.paused = false;
    (document.getElementById('menu-btn') as HTMLElement).focus({ preventScroll: true });
  };
  document.getElementById('menu-btn')!.addEventListener('click', () => {
    game.paused = true;
    intro.setResumable(true);
    intro.show();
  });
  // Esc returns to the run in progress
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Escape' && intro.canResume && document.body.classList.contains('intro-open')) intro.onResume();
  });

  game.paused = !direct;
  if (direct) intro.hide(); else intro.show();
  await intro.render();
  if (direct) await startEngine(level, player);
}

void main();
