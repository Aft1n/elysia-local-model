import { afterEach, describe, expect, it } from "bun:test"
import { Elysia } from "elysia"
import {
  getDecisionModel,
  localModel,
  resetDecisionModelCache,
  resolveLocalModelConfig,
  type DecisionAdapterFactory,
  type DecisionModelLoadOptions,
  type DecisionRequest,
  type DecisionResult,
  type LocalModelDefinition,
} from "../src/index"
import { resolveDecisionModelDefinition, resolveModelDefinition } from "../src/utils"

const REQUEST: DecisionRequest = {
  state: "the sky is blue",
  questions: { verdict: { type: "noul", instructions: "Is the sky blue?" } },
}

const RESULT: DecisionResult = { answers: { verdict: { type: "noul", noul: 1 } } }

function createStub() {
  const calls: DecisionModelLoadOptions[] = []
  const disposes: number[] = []

  const factory: DecisionAdapterFactory = (options) => {
    const index = calls.push(options) - 1
    return {
      decide: async () => RESULT,
      dispose: () => {
        disposes.push(index)
      },
    }
  }

  return { factory, calls, disposes }
}

afterEach(() => resetDecisionModelCache())

describe("plugin API config errors", () => {
  it("surfaces an unknown model alias as the documented config error", async () => {
    const app = new Elysia()
      .use(localModel({ models: { known: { task: "feature-extraction", model: "org/known" } } }))
      .get("/run", ({ localModel: api }) => api.run("missing", "hello"))

    const response = await app.handle(new Request("http://localhost/run"))

    expect(response.status).toBe(500)
    expect(await response.json()).toMatchObject({
      detail: 'Local model "missing" is not defined in the elysia-local-model config.',
    })
  })

  it("surfaces an unknown decision alias as the documented config error", async () => {
    const app = new Elysia()
      .use(localModel({ decisionModels: { verdict: { source: "./models/verdict.onnx" } } }))
      .post("/decide", ({ localModel: api }) => api.decision("missing"))

    const response = await app.handle(new Request("http://localhost/decide", { method: "POST" }))

    expect(response.status).toBe(500)
    expect(await response.json()).toMatchObject({
      detail: 'Decision model "missing" is not defined in the elysia-local-model config.',
    })
  })
})

describe("localModel.decision", () => {
  it("resolves the configured definition and runs decide() through the override adapter", async () => {
    const stub = createStub()
    const app = new Elysia()
      .use(
        localModel({
          decisionModels: { verdict: { source: "./models/verdict.onnx", maxTokens: 32 } },
        }),
      )
      .post("/decide", async ({ localModel: api }) => {
        const model = await api.decision("verdict", { adapter: stub.factory })
        return { definition: model.definition, answer: (await model.decide(REQUEST)).answers.verdict }
      })

    const response = await app.handle(new Request("http://localhost/decide", { method: "POST" }))

    expect(response.status).toBe(200)
    expect(await response.json()).toMatchObject({
      definition: { source: "./models/verdict.onnx", maxTokens: 32 },
      answer: { type: "noul", noul: 1 },
    })
    expect(stub.calls[0]).toMatchObject({ source: "./models/verdict.onnx", maxTokens: 32 })
  })

  it("lets an overrides.source win over the configured source", async () => {
    const stub = createStub()
    const config = resolveLocalModelConfig({
      decisionModels: { verdict: { source: "./models/verdict.onnx" } },
    })

    const model = await getDecisionModel("verdict", config, {
      source: "./models/override.onnx",
      adapter: stub.factory,
    })

    expect(model.definition.source).toBe("./models/override.onnx")
    expect(stub.calls[0]?.source).toBe("./models/override.onnx")
  })

  it("shares one adapter for identical definitions and disposes it on the last release", async () => {
    const stub = createStub()
    const config = resolveLocalModelConfig({
      decisionModels: { cached: { source: "./models/cached.onnx" } },
    })

    const first = await getDecisionModel("cached", config, { adapter: stub.factory })
    const second = await getDecisionModel("cached", config, { adapter: stub.factory })

    expect(first.definition).toEqual({ source: "./models/cached.onnx" })
    expect(stub.calls).toHaveLength(1)
    expect((await second.decide(REQUEST)).answers).toEqual(RESULT.answers)

    await first.dispose()
    expect(stub.disposes).toHaveLength(0)

    await second.dispose()
    expect(stub.disposes).toEqual([0])
  })

  it("keeps a distinct session per resolved definition", async () => {
    const stub = createStub()
    const config = resolveLocalModelConfig({
      decisionModels: { split: { source: "./models/a.onnx" } },
    })

    const first = await getDecisionModel("split", config, { adapter: stub.factory })
    const second = await getDecisionModel("split", config, {
      source: "./models/b.onnx",
      adapter: stub.factory,
    })

    expect(stub.calls.map((options) => options.source)).toEqual(["./models/a.onnx", "./models/b.onnx"])

    await first.dispose()
    await second.dispose()
    expect(stub.disposes).toEqual([0, 1])
  })
})

describe("resolveModelDefinition", () => {
  it("merges registry options with per-load options, per-load winning", () => {
    const config = resolveLocalModelConfig({
      models: {
        embed: { task: "feature-extraction", model: "org/embed", options: { dtype: "q8", revision: "main" } },
      },
    })

    expect(resolveModelDefinition("embed", config, { dtype: "fp32", local_files_only: true })).toEqual({
      task: "feature-extraction",
      model: "org/embed",
      options: { dtype: "fp32", revision: "main", local_files_only: true },
    })
  })

  it("falls back to defaultTask when the entry omits task", () => {
    const config = resolveLocalModelConfig({
      defaultTask: "image-classification",
      models: { embed: { model: "org/embed" } as LocalModelDefinition },
    })

    expect(resolveModelDefinition("embed", config).task).toBe("image-classification")
  })

  it("uses the alias as the model id when the entry omits model", () => {
    const config = resolveLocalModelConfig({
      models: { "Xenova/all-MiniLM-L6-v2": { task: "feature-extraction" } as LocalModelDefinition },
    })

    expect(resolveModelDefinition("Xenova/all-MiniLM-L6-v2", config).model).toBe("Xenova/all-MiniLM-L6-v2")
  })
})

describe("resolveDecisionModelDefinition", () => {
  it("throws when the entry declares only a blank source", () => {
    const config = resolveLocalModelConfig({ decisionModels: { blank: { source: "   " } } })

    expect(() => resolveDecisionModelDefinition("blank", config)).toThrow(
      'Decision model "blank" must declare a non-empty "source" in the config.',
    )
  })
})
