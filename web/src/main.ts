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

  // The landing page is static content: show it first, while the engine (three.js, WebGPU) loads behind it.
  const intro = new Intro(document.getElementById('intro')!, await loadBenchmark());
  intro.setEngine('loading');
  intro.onPlay = ({ mode, model }) => {
    intro.setResumable(false);
    intro.hide();
    game.paused = false;
    void game.newRun(mode === 'agentic' ? model : '');
  };
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
  game.paused = !direct; // the level-1 scene stays visible, frozen, behind the intro
  if (direct) intro.hide(); else intro.show();
  await intro.render();

  let renderer: Renderer;
  try {
    const { Renderer: R } = await import('./render');
    renderer = new R(canvas, game, assetBase);
    await renderer.init();
  } catch (e) {
    console.error(e);
    const msg = `Couldn't start the renderer: ${(e as Error).message}`;
    intro.setEngine('error', msg);
    const box = document.getElementById('fatal')!;
    box.hidden = false;
    box.textContent = msg;
    return;
  }
  await game.start(level, direct ? player : '');
  intro.setEngine('ready');

  if (import.meta.env.DEV) (window as any).__backend = backendFor; // console experiments with prompt formats

  // While the intro is open the world is frozen, so redraw only occasionally and keep the GPU free.
  let lastIdle = 0;
  const loop = (t: number) => {
    if (!game.paused || t - lastIdle > 400) {
      renderer.frame();
      lastIdle = t;
    }
    hud.update(renderer.framesPerSecond, renderer.backend);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

void main();
