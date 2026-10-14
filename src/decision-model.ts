import { AutoTokenizer } from "@huggingface/transformers"
import type {
  DecisionDecideOptions,
  DecisionModelAdapter,
  DecisionModelLoadOptions,
  DecisionRequest,
  DecisionResult,
} from "./types"
import { createDecisionOnnxAdapter, type DecisionEncodedInput } from "./decision/decision-onnx"
import { createDecisionSharedPrefixAdapter } from "./decision/decision-shared-prefix"
import {
  decisionSessionKey,
  getDecisionSession,
  releaseDecisionSession,
  resetDecisionSessions,
} from "./decision/decision-session"
import { resolveDecisionModelDefinition, type ResolvedLocalModelConfig } from "./utils"

const DEFAULT_TOKENIZER = "Xenova/all-MiniLM-L6-v2"

export { resetDecisionSessions as resetDecisionModelCache }

export type DecisionAdapterFactory = (
  options: DecisionModelLoadOptions,
) => Promise<DecisionModelAdapter> | DecisionModelAdapter

export interface DecisionModelOverrides {
  source?: string
  /** Custom runtime, replacing the configured adapter for this call. */
  adapter?: DecisionAdapterFactory
}

export interface ServerDecisionModel {
  readonly definition: DecisionModelLoadOptions
  decide(request: DecisionRequest, options?: DecisionDecideOptions): Promise<DecisionResult>
  dispose(): Promise<void>
}

/**
 * Load (and cache) the adapter behind a configured decision model alias.
 * Sessions are cached per resolved definition, so repeated calls share one
 * loaded ONNX session; `dispose()` releases it on the last holder.
 */
export async function getDecisionModel(
  name: string,
  config: ResolvedLocalModelConfig,
  overrides: DecisionModelOverrides = {},
): Promise<ServerDecisionModel> {
  const definition = resolveDecisionModelDefinition(
    name,
    config,
    overrides.source ? { source: overrides.source } : {},
  )

  const key = decisionSessionKey(name, definition, overrides.adapter)
  const session = getDecisionSession(key, () => createAdapter(definition, overrides.adapter))
  const adapter = await session

  let disposed = false
  return {
    definition,
    decide: (request, decideOptions) => adapter.decide(request, decideOptions),
    dispose: async () => {
      if (disposed) return
      disposed = true
      // Last holder disposes; a superseded session stays alive for its holders.
      if (releaseDecisionSession(key, session)) {
        await adapter.dispose?.()
      }
    },
  }
}

async function createAdapter(
  definition: DecisionModelLoadOptions,
  override?: DecisionAdapterFactory,
): Promise<DecisionModelAdapter> {
  if (override) return override(definition)
  // "shared-prefix" selects the bundled manifest-directory adapter (manifest +
  // tokenizer + model.onnx, local dir / URL / HF id); anything else is a plain
  // single-file ONNX decision model.
  if (definition.adapter?.toLowerCase() === "shared-prefix") {
    return createDecisionSharedPrefixAdapter(definition)
  }
  return createOnnxAdapter(definition)
}

async function createOnnxAdapter(options: DecisionModelLoadOptions): Promise<DecisionModelAdapter> {
  // Loaded once per adapter instance: a tokenizer is a multi-megabyte
  // vocabulary table. MiniLM is only the fallback, since a decision model
  // brings its own vocabulary.
  const tokenizer = await AutoTokenizer.from_pretrained(options.tokenizer ?? DEFAULT_TOKENIZER)
  return createDecisionOnnxAdapter({
    ...options,
    encode: async (text: string): Promise<DecisionEncodedInput> =>
      // Transformers.js takes the model's own context length by default,
      // which is exactly the silent truncation the adapter must report.
      tokenizer(text, {
        padding: true,
        truncation: true,
        ...(options.maxTokens === undefined ? {} : { max_length: options.maxTokens }),
      }) as unknown as DecisionEncodedInput,
  })
}
