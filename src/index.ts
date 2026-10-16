import { Elysia } from "elysia"
import type { LocalModelConfig, LocalModelPipeline } from "./types"
import { loadLocalModel } from "./local-model"
import {
  getDecisionModel,
  type DecisionModelOverrides,
  type ServerDecisionModel,
} from "./decision-model"
import { resolveLocalModelConfig } from "./utils"

export interface LocalModelApi {
  /** Cached pipeline behind a configured alias; call it like a Transformers.js pipeline. */
  get(name: string): Promise<LocalModelPipeline>
  /** Convenience: load the alias's pipeline and run one input through it. */
  run(name: string, input: unknown, callOptions?: Record<string, unknown>): Promise<unknown>
  /** Cached decision model adapter behind a configured `decisionModels` alias. */
  decision(name: string, overrides?: DecisionModelOverrides): Promise<ServerDecisionModel>
}

/**
 * Elysia plugin running local Hugging Face models (Transformers.js) in-process.
 *
 * ```ts
 * new Elysia()
 *   .use(localModel({
 *     models: { embedding: { task: "feature-extraction", model: "Xenova/all-MiniLM-L6-v2" } },
 *   }))
 *   .get("/", async ({ localModel }) => localModel.run("embedding", "hello"))
 * ```
 */
export const localModel = (config: LocalModelConfig = {}) => {
  const resolved = resolveLocalModelConfig(config)

  const api: LocalModelApi = {
    get: (name) => loadLocalModel(name, resolved),
    run: async (name, input, callOptions) => (await loadLocalModel(name, resolved))(input, callOptions),
    decision: (name, overrides) => getDecisionModel(name, resolved, overrides),
  }

  return new Elysia({ name: "elysia-local-model" })
    .decorate("localModel", api)
    .onStart(async () => {
      const targets = resolved.prewarm === true ? Object.keys(resolved.models) : resolved.prewarm || []
      if (targets.length) await Promise.all(targets.map((name) => loadLocalModel(name, resolved)))
    })
}

export { loadLocalModel } from "./local-model"
export { getDecisionModel, resetDecisionModelCache } from "./decision-model"
export { resolveLocalModelConfig } from "./utils"
export type { DecisionAdapterFactory, DecisionModelOverrides, ServerDecisionModel } from "./decision-model"
export type { ResolvedLocalModelConfig } from "./utils"
export type {
  DecisionAnswer,
  DecisionAnswers,
  DecisionChoiceAnswer,
  DecisionChoiceQuestion,
  DecisionDecideOptions,
  DecisionModelAdapter,
  DecisionModelDefinition,
  DecisionModelLoadOptions,
  DecisionModelState,
  DecisionNoulAnswer,
  DecisionNoulQuestion,
  DecisionQuestion,
  DecisionQuestionMap,
  DecisionRequest,
  DecisionResult,
  DecisionScoreAnswer,
  DecisionScoreQuestion,
  DecisionUsage,
  LocalModelConfig,
  LocalModelDefinition,
  LocalModelModelRegistry,
  LocalModelName,
  LocalModelPipeline,
  LocalModelPipelineLoadOptions,
  LocalModelPrewarmTargets,
  LocalModelRunner,
  LocalModelTask,
} from "./types"
