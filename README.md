# Squirrel Finder

Dodge the ghosts, find the penguin. You can play it yourself, or let a local
[Ollama](https://ollama.com) decision model play the wizard. Seven levels: a moonlit graveyard and six
voxel dungeon rooms, all lit at night.

```
macos/   native Swift + Metal app (SwiftPM)
web/     WebGPU edition: TypeScript + Three.js (WebGPURenderer) + Vite, developed with pnpm
docs/    the web build output (served as GitHub Pages / any static host)
```

## web/

```bash
cd web
pnpm install
pnpm dev          # http://127.0.0.1:5173  (add ?level=3 or ?player=<ollama model> to jump in)
pnpm build        # type-checks, then writes the production site to ../docs
pnpm preview      # serves ../docs on http://127.0.0.1:4173
```

- `pnpm dev` / `pnpm build` first run `scripts/sync-assets.mjs`, which copies the art from
  `macos/Assets` into `web/public/art` (the macOS project is the single source of truth for art).
- Rendering uses Three.js `WebGPURenderer` with TSL node materials (it falls back to WebGL 2 where WebGPU
  is missing). Voxel rooms are real 3D meshes under a tilted orthographic camera; every sprite, shadow
  and crack is drawn from one instanced mesh.
- Controls: arrow keys / WASD, Space for next level or retry.

### Letting an Ollama model play from the browser

The page calls Ollama directly (`/v1/models`, `/v1/systemone`). `localhost` / `127.0.0.1` origins are
allowed by default, so `pnpm dev` and `pnpm preview` just work. For a deployed site, allow its origin
first (and note browsers may block an `https` page from reaching a plain-http local server):

```bash
OLLAMA_ORIGINS="https://c2p-cmd.github.io" ollama serve
```

Point the page at another server from the **Ollama server** field in Agentic mode (Connect / Reset), or with `?ollama=http://host:11434`. The choice is remembered in `localStorage`.

### Laya running inside the browser (no Ollama)

The player menu has **🌐 Laya (in your browser)**. It runs the typed-decisions Laya checkpoint locally with
ONNX Runtime Web, via [layaForWeb](https://github.com/vishalmysore/layaForWeb) (Apache-2.0; its `laya-core.js`
is vendored in `web/src/laya/`).

- Weights are fetched from [VishalMysore/layaForWebTrained](https://huggingface.co/VishalMysore/layaForWebTrained)
  on first use (~440 MB, split into 18 parts), then cached in the browser's Cache Storage.
- `?laya=q4e8` selects the smaller int4 build (~290 MB), which can use WebGPU. The default `q8e8` runs on WASM.
- `?layaBase=https://…/` points at another copy of the weights (needs CORS and the same file layout).
- Multithreaded WASM needs cross-origin isolation. `pnpm dev` / `pnpm preview` send the headers; on static hosts
  `public/coi-sw.js` (a service worker) adds them and the page reloads once. Without it Laya still runs, single-threaded.
- Laya is a 421M-parameter English classifier, not a planner: it follows a clearly stated goal direction well but
  is weak at "avoid that" reasoning, so it plays worse than the 9B Ollama models. Its prompt is therefore a short,
  dominant-axis-first description (see `describeStateCompact` in `web/src/game.ts`).

## Can small "System One" models play it? (the experiment)

The web page opens on an intro that explains the project, shows this comparison (rendered from
`web/public/data/benchmark.json`) and offers human mode or agentic mode with any model below.

| Model | Where it runs | Source |
|---|---|---|
| Laya (typed decisions) | ONNX Runtime Web, WASM | [VishalMysore/layaForWebTrained](https://huggingface.co/VishalMysore/layaForWebTrained) via [layaForWeb](https://github.com/vishalmysore/layaForWeb) |
| Julia-1 | ONNX Runtime Web, WebGPU | [SupersonicLabs/Julia-1-ONNX](https://huggingface.co/SupersonicLabs/Julia-1-ONNX) |
| Kev-0.6B browser-use | Transformers.js + [open-jev](https://www.npmjs.com/package/open-jev), WebGPU | [arbazsiddiqui/kev-0.6b-browser-use-ONNX](https://huggingface.co/arbazsiddiqui/kev-0.6b-browser-use-ONNX) |
| Decision Lab V3 INT4 | decision-lab SDK, WebGPU (dev build only) | [khalilelghoul01/decision-lab](https://github.com/khalilelghoul01/decision-lab) |
| laya / nimble / kev (Ollama) | local Ollama server (reference) | your `ollama list` |

### What was measured

113 decision points were captured by playing seeded headless episodes of the real game (all 7 levels), then each
model decided on every one, with three prompt styles (short, short with plain-English option labels, full). An
oracle that simulates each of the 8 directions for 0.7 s with the game's own collision rules provides ground truth:
*best* (the oracle's top move), *acceptable* (safe and makes at least half the best progress), *fatal* (walks into a
ghost). Each model's best prompt style then played 18 complete runs (levels 1, 3, 5; six seeded starts each).

| Model | Params | Download | Short prompt: best / acceptable / fatal | Latency p50 | Wins in 18 runs |
|---|---|---|---|---|---|
| Ollama · kev:9b-nvfp4 | 9B | local server | 30% / 50% / 20% | 499 ms | 22% (full) |
| Julia-1 | ~140M | 617 MB | 14% / 31% / 16% | 16 ms | 17% (compact) |
| Ollama · laya:421m-typed-decisions-mlx-fp16 | 421M | local server | 25% / 42% / 16% | 41 ms | 11% (compact) |
| Ollama · nimble:9b-nvfp4 | 9B | local server | 26% / 42% / 13% | 906 ms | 11% (full) |
| Laya (typed decisions) | 421M | 440 MB | 30% / 49% / 19% | 500 ms | 0% (compact) |
| Kev-0.6B browser-use | 0.6B | 361 MB | 22% / 40% / 27% | 61 ms | 0% (natural) |
| Decision Lab V3 | 350M | 296 MB | 12% / 27% / 12% | 39 ms | 0% (natural) |
| *Random baseline* | | | 15% / 29% / 17% | | 6% |
| *0.7 s lookahead oracle* | | | 100% / 100% / 0% | | 94% |

The honest summary: **single-decision accuracy over-states playing strength.** Most models beat random on isolated
moves, but only the 9B models and Julia-1 won clearly more often than random over whole runs, and with 18 runs one
win is 5.6 points, so differences under ~15 points are noise. Laya (and Kev) read direction words in the state
rather than reason about them: asked to avoid a ghost "to the left" they pick left. The oracle wins 94%, so the game
is solvable; these models just aren't planners. Raw per-decision rows are in `web/bench-results/` (not published).

### Re-running the benchmark (dev server only)

```bash
cd web && pnpm dev                       # http://127.0.0.1:5173
# in the browser, one page per task (results are saved by the dev server into web/bench-results):
#   /bench.html?task=scenarios
#   /bench.html?task=model&id=laya-web&variants=compact,natural,full&n=113
#   /bench.html?task=episodes&id=laya-web&variant=compact&levels=1,3,5&per=6     (@random and @oracle are baselines)
pnpm bench:merge                         # writes public/data/benchmark.json for the intro page
pnpm build                               # republishes docs/
```

Model ids: `laya-web`, `julia-1`, `kev-0.6b-bu`, `decision-lab`, `ollama:<tag>`.

### Decision Lab needs local weights

Its INT4 weights are one 296 MB file (over GitHub's 100 MB file limit), so they are not part of `docs/`. To try it:
download `decision-lab-v3-int4.zip` from the project's v0.1.0 release (sha256
`63275c9263beb3105782a687f8277019a25218dc18cb458ea6fd5e2cdd46c55f`), extract `model-v3-q4/` into `web/local-models/`,
then `pnpm dev`. The weights are under the LFM Open License v1.0; the code is MIT.

### Publishing

GitHub Pages: set the source to the `/docs` folder of the default branch. `docs/` is regenerated by
`pnpm build`; do not edit it by hand.

## macos/

```bash
cd macos
./run.sh                       # builds build/SquirrelFinder.app and launches it
swift build -c release         # or just build
open Package.swift             # or open it in Xcode and run the SquirrelFinder scheme
```

Needs macOS 14+ and the Ollama server on `127.0.0.1:11434` for the AI players. `macos/tools/` holds the
asset helper scripts (prop atlas builder, voxel room previewer).

## Credits

- Site icon: [Penguin icons created by G-CAT - Flaticon](https://www.flaticon.com/free-icons/penguin).
- Characters (wizard, ghosts): pixel art from [FreePixel](https://freepixel.art/), free for personal and commercial use, attribution optional ([asset licence](https://freepixel.art/license/)). The licence forbids redistributing the raw asset files as assets, so only the copies inside the built game (`docs/art`) are committed.
- Dungeon room models: [Tiny Voxel Dungeon](https://maxparata.itch.io/tinyvoxeldungeon) by **Max Parata**. The author states: "You can use it for any kind of project that you have (commercial or not): game, movies, sceneries, 3D printing etc." and asks for his name in the credits, which is why it is credited here and on the site. The room models (`docs/art/rooms`, `macos/Assets/rooms`) are included in this repository.
- Charts: [Plotly.js](https://plotly.com/javascript/) (MIT), vendored at `web/public/vendor/plotly/` with its licence.
- Runtime libraries: three.js (MIT), ONNX Runtime Web (MIT), Transformers.js (Apache-2.0). Licence texts are in `web/public/licenses/`.
- Julia 1 WebGPU export by Supersonic Labs (Apache-2.0), see `web/public/licenses/julia-1-README.md`.

## Art licences

The in-browser Laya path vendors code from layaForWeb (Apache-2.0; licence and notice in
`web/public/licenses/`) and downloads weights derived from `convaiinnovations/laya-typed-decisions` (Apache-2.0).

The art comes from these sources (all credited on the site):

- Characters (wizard, ghosts): [FreePixel](https://freepixel.art/), free for commercial use, attribution optional.
- Dungeon rooms: [Tiny Voxel Dungeon](https://maxparata.itch.io/tinyvoxeldungeon) by Max Parata, free for any project.
- Graveyard tiles and props: [Dark Horror Asset Pack 16x16](https://nonokaatm.itch.io/dark-horror-asset-pack-16x16) by nonokaatm, free to use with attribution (credited here and on the site). The pack's own page is the authority on its terms; this repo ships only the baked `ground.png`, `props.png` and `cracks.png` inside `docs/art`, not the original tileset.
- Penguin: generated with AI by the author.

Raw source copies in `macos/Assets` are git-ignored except the rooms and the penguin, so a fresh clone needs
its own copy of the other sprites there
