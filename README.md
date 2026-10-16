# elysia-local-model [![npm version][npm-version-src]][npm-version-href]

An [Elysia](https://elysiajs.com) v2 plugin for running local Hugging Face transformer models in-process on CPU via [Transformers.js](https://huggingface.co/docs/transformers.js). Server-side sibling of [nuxt-local-model](https://github.com/Aft1n/nuxt-local-model).

## Install

```bash
bun add elysia-local-model
```

## Usage

```ts
import { Elysia } from "elysia"
import { localModel } from "elysia-local-model"

const app = new Elysia()
  .use(localModel({
    cacheDir: "./.ai-models",
    models: {
      embedding: {
        task: "feature-extraction",
        model: "Xenova/all-MiniLM-L6-v2",
        options: { dtype: "q8" },
      },
    },
    prewarm: ["embedding"],
  }))
  .get("/embed", async ({ localModel }) => {
    const output = await localModel.run("embedding", "hello world", {
      pooling: "mean",
      normalize: true,
    })
    return output
  })
  .listen(3000)
```

The plugin decorates `localModel`:

| Method | Description |
| --- | --- |
| `get(name)` | Cached Transformers.js pipeline for the alias; call it `(input, callOptions?)`. |
| `run(name, input, callOptions?)` | Load the alias's pipeline and run one input through it. |
| `decision(name, overrides?)` | Cached decision model adapter, see below. |

## Config

| Option | Default | Description |
| --- | --- | --- |
| `models` | `{}` | Named pipeline aliases: `alias -> { task, model, options }`. Any Transformers.js task/model pair works. |
| `decisionModels` | `{}` | Named decision model definitions, see below. |
| `cacheDir` | `./.ai-models` | Where downloaded weights are cached. Also `ELYSIA_LOCAL_MODEL_CACHE_DIR`. |
| `allowRemoteModels` | `true` | Allow fetching models from the Hugging Face Hub. |
| `allowLocalModels` | `true` | Allow loading model files from disk. |
| `defaultTask` | `feature-extraction` | Pipeline task used when an alias omits `task`. |
| `prewarm` | `false` | `true` warms all aliases on server start, or pass `["embedding"]` for specific aliases. |

## Decision models

Typed local judgements (yes/no, choice, score) executed over ONNX — the same wire format as `nuxt-local-model`. Declare a model by `source` (a local `.onnx` file or directory, an `https://` URL, or a Hugging Face id) and call it through the plugin:

```ts
const app = new Elysia()
  .use(localModel({
    decisionModels: {
      verdict: { source: "my-org/my-decision-model", adapter: "shared-prefix" },
    },
  }))
  .post("/decide", async ({ localModel, body }) => {
    const model = await localModel.decision("verdict")
    return model.decide({
      state: body.text,
      questions: {
        ok: { type: "noul", instructions: "Is this acceptable?" },
      },
    })
  })
```

`model.decide()` returns `{ answers }` with one answer per question — `noul` (number), `choice` (label + probabilities), or `score` (level + probabilities). `adapter: "shared-prefix"` selects the bundled manifest-directory adapter (`manifest.json` + `tokenizer.json` + `model.onnx`) and needs the optional peer `@huggingface/tokenizers`; the default adapter runs a single `.onnx` file. Custom runtimes: pass a factory to `localModel.decision(name, { adapter })` implementing `DecisionModelAdapter`.

## Notes

- Inference runs in-process and blocks the event loop while a model executes. `prewarm` on start to pay the load cost up front. If concurrent requests must never wait on inference, run models behind a Bun `Worker` instead — the pipeline API is identical.
- ONNX Runtime Node (`onnxruntime-node`) is used as the Transformers.js backend when available; otherwise Transformers.js falls back to its default backend.

## License

MIT

[npm-version-src]: https://img.shields.io/npm/v/elysia-local-model?style=flat-square
[npm-version-href]: https://www.npmjs.com/package/elysia-local-model
