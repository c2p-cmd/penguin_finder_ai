// Copies the art the web build needs from the macOS project (the single source of truth for assets)
// into public/art, so `vite build` ships it to ../docs.
import { cpSync, existsSync, mkdirSync, readdirSync, realpathSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, '../../macos/Assets');
const dst = resolve(here, '../public/art');

const files = [
  'ground.png', 'wizard_walk.png', 'wizard_idle.png', 'ghost_walk.png',
  'penguin.png', 'props.png', 'cracks.png',
];

if (!existsSync(src)) {
  console.error(`sync-assets: missing ${src}`);
  process.exit(1);
}
rmSync(dst, { recursive: true, force: true });
mkdirSync(join(dst, 'rooms'), { recursive: true });

for (const f of files) {
  if (!existsSync(join(src, f))) {
    console.error(`sync-assets: missing ${f}`);
    process.exit(1);
  }
  cpSync(join(src, f), join(dst, f));
}
let rooms = 0;
for (const f of readdirSync(join(src, 'rooms'))) {
  if (f.endsWith('.obj') || f.endsWith('.png')) {
    cpSync(join(src, 'rooms', f), join(dst, 'rooms', f));
    rooms++;
  }
}
console.log(`sync-assets: ${files.length} sprites + ${rooms} room files -> public/art`);

// ONNX Runtime Web (same files layaForWeb vendors): loader + JSEP wasm, which provides both the WASM and WebGPU
// execution providers. Loaded at runtime from vendor/ort so the wasm is fetched only when Laya is used.
const ort = resolve(here, '../node_modules/onnxruntime-web/dist');
const vendor = resolve(here, '../public/vendor/ort');
rmSync(vendor, { recursive: true, force: true });
mkdirSync(vendor, { recursive: true });
for (const f of ['ort.min.mjs', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm']) {
  if (!existsSync(join(ort, f))) {
    console.error(`sync-assets: missing onnxruntime-web/dist/${f} (run pnpm install)`);
    process.exit(1);
  }
  cpSync(join(ort, f), join(vendor, f));
}
console.log('sync-assets: onnxruntime-web runtime -> public/vendor/ort');

// decision-lab-sdk pins onnxruntime-web 1.24.3 and wants its asyncify build files served next to the page.
// With pnpm, that version sits next to the SDK in the virtual store.
{
  const sdkReal = realpathSync(resolve(here, '../node_modules/decision-lab-sdk'));
  const ortDist = join(dirname(sdkReal), 'onnxruntime-web', 'dist');
  const out = resolve(here, '../local-models/ort-dl'); // served at /models/ort-dl (dev/preview only)
  rmSync(out, { recursive: true, force: true });
  mkdirSync(out, { recursive: true });
  for (const f of ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm']) {
    if (!existsSync(join(ortDist, f))) {
      console.error(`sync-assets: missing ${f} in ${ortDist}`);
      process.exit(1);
    }
    cpSync(join(ortDist, f), join(out, f));
  }
  console.log('sync-assets: decision-lab ONNX Runtime files -> local-models/ort-dl');
}
