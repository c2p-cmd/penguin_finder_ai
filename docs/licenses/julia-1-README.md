---
license: apache-2.0
base_model: SupersonicLabs/Julia-1
library_name: onnx
tags:
  - decision-model
  - webgpu
  - onnx
  - multilingual
---

![Supersonic Labs logo](assets/supersonic-labs-dark.svg)

<h1 align="center">Julia 1 WebGPU</h1>
<p align="center">Julia 1 decisions in the browser.</p>

![Julia 1 artwork](assets/julia-banner.png)

**This repository is for browser WebGPU inference.** For the original Julia 1 Python CPU and CUDA runtime, use [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1). This export uses the same Julia 1 weights and decision graph. It accepts a state, a question, and 2–20 answer options, then scores the options in their supplied order.

## Try it

Run the included playground locally:

```bash
npm install
npm run dev
```

Open the URL printed by Vite and select `playground.html`. The page renders before the model downloads. Press **Load model** when you want to use Julia. The library downloads the ONNX graph, 551 MB external weights, and Rust WebAssembly tokenizer, creates a WebGPU session, and warms it. Later decisions reuse that session.

The playground lets you edit `state`, `question`, `type` (`choice`, `score`, or `noul`), and one option per line. It shows the selected answer, probabilities, and raw logits. For `noul`, provide exactly two options in false/true order.

## Use the library

For a page that appears immediately, start the download from a button click:

```js
let julia;
const loadButton = document.querySelector('#load-model');
const decideButton = document.querySelector('#decide');

decideButton.disabled = true;
loadButton.addEventListener('click', async () => {
  const { JuliaWebGPU } = await import('./index.js');
  julia = await JuliaWebGPU.load('/models/Julia-1-ONNX/');
  decideButton.disabled = false;
});

decideButton.addEventListener('click', async () => {
  const [decision] = await julia.predict([{
    state: 'The customer asks to reset their password.',
    question: 'Which team should handle this?',
    options: ['Account support', 'Billing'],
    type: 'choice',
  }]);
  console.log(decision.index, decision.probabilities);
});
```

`load(baseUrl)` downloads and warms the model once. Keep its returned instance in memory. `predict(rows)` returns indexes and display probabilities; `logits(rows)` returns raw scores. Strict encoding is enabled by default. All computation stays in the browser when these files are served locally. A browser with WebGPU is required.

The model files `model.onnx` and `model.onnx.data` must remain together. The browser package uses ONNX Runtime WebGPU to compile the model operations to WGSL. The bundled Rust WebAssembly tokenizer handles request encoding. The `rust/` source also exposes a Node N-API binding for apps that want resident native encoding. Building it is optional for browser use.

## How this differs from original Julia 1

| | Original Julia 1 | Julia 1 WebGPU |
| --- | --- | --- |
| Trained weights | Julia 1 checkpoint | Same weights exported to ONNX |
| Decision graph | PyTorch | Same graph exported to ONNX |
| Inference | Python CPU or CUDA | Browser WebGPU through WGSL kernels |
| Request encoding | Python tokenizer | Bundled Rust WebAssembly tokenizer |
| First use | Load resident model in Python | Download weights, initialize WebGPU, and warm session when requested |
| Later use | Reuse Python model | Reuse browser model session |

The model's decision quality should be effectively the same. Floating point rounding can change a choice if competing logits are nearly tied. The [original Julia 1 evaluation](https://huggingface.co/SupersonicLabs/Julia-1) reported 73.15% on typed decisions, 94% on an AG News pilot, and 86% on an Emotion pilot. Those published accuracy results were measured on the original runtime, **not rerun on WebGPU**. This repository tests output agreement against the original model.

## WebGPU benchmark

The browser benchmark ran this ONNX export through WebGPU in Brave on Linux. It used **100 real Julia validation requests**, batches of four, with five measured runs after warmup.

| Browser WebGPU ONNX | Result |
| --- | ---: |
| Median time for 100 decisions | **7.55 s** |
| Median time per decision | **75.47 ms** |
| Predictions matching original Julia 1 | **100 / 100** |
| Maximum absolute logit difference | **0.00225** |
| Model load and warmup, local browser cache | **5.84 s** |

[Full WebGPU measurements](benchmark-webgpu.json) · [the 100 test requests and original logits](parity-cases.json) · [rerun in the browser](benchmark-webgpu.html)

The original Julia 1 CPU run on the same requests took 18.23 s on this machine. It is a reference for the original model's output, **not a CPU mode of this WebGPU library**. CPU and CUDA deployment belong to [the original Julia 1 repository](https://huggingface.co/SupersonicLabs/Julia-1). Browser WebGPU and Python CPU have different runtime overhead, so this is not a controlled hardware speedup claim. A CUDA comparison was not available on this machine.

The original accuracy suite has not been rerun in WebGPU. The 100 matching predictions check output parity on this request set, not a new accuracy figure. A nearly tied decision can still change due to floating point rounding.

## Files

- `model.onnx` and `model.onnx.data`: full Julia 1 ONNX graph and weights.
- `tokenizer.json` and `tokenizer_config.json`: Julia 1 tokenizer.
- `index.js`: lazy WebGPU adapter and resident model API.
- `wasm/` and `rust/`: packaged Rust WebAssembly tokenizer and N-API source.
- `playground.html`: decision interface.
- `benchmark-webgpu.html`: repeat the browser WebGPU timing and output parity check.
- `export.py`: regenerates the ONNX graph from the original checkpoint.

## Limits

Julia chooses among supplied answers; it is not a text generation model. It works best when the state contains the needed evidence and the options are distinct. Each native call accepts 2–20 options. Browser GPU limits vary, and a device needs enough memory for the full model. The original Julia 1 model card contains the detailed evaluation protocol and limitations.
