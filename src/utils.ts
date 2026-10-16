import { env as transformersEnv } from "@huggingface/transformers"
import ort from "onnxruntime-node"
import type {
  DecisionModelLoadOptions,
  LocalModelConfig,
  LocalModelDefinition,
  LocalModelModelRegistry,
  LocalModelPipelineLoadOptions,
  LocalModelTask,
} from "./types"

export interface ResolvedLocalModelConfig {
  models: LocalModelModelRegistry
  decisionModels: Record<string, DecisionModelLoadOptions>
  cacheDir: string
  allowRemoteModels: boolean
  allowLocalModels: boolean
  defaultTask: LocalModelTask
  prewarm: boolean | string[]
}

export function resolveCacheDir(cacheDir?: string) {
  if (cacheDir?.trim()) return cacheDir.trim()
  return globalThis.process?.env?.ELYSIA_LOCAL_MODEL_CACHE_DIR?.trim() || "./.ai-models"
}

export function resolveLocalModelConfig(config?: LocalModelConfig): ResolvedLocalModelConfig {
  return {
    allowRemoteModels: config?.allowRemoteModels ?? true,
    allowLocalModels: config?.allowLocalModels ?? true,
    cacheDir: resolveCacheDir(config?.cacheDir),
    defaultTask: config?.defaultTask ?? "feature-extraction",
    prewarm: config?.prewarm ?? false,
    models: config?.models ?? {},
    decisionModels: config?.decisionModels ?? {},
  }
}

export function applyLocalModelEnvironment(
  config: Pick<ResolvedLocalModelConfig, "cacheDir" | "allowRemoteModels" | "allowLocalModels">,
) {
  transformersEnv.cacheDir = resolveCacheDir(config.cacheDir)
  transformersEnv.localModelPath = transformersEnv.cacheDir
  transformersEnv.allowRemoteModels = config.allowRemoteModels
  transformersEnv.allowLocalModels = config.allowLocalModels
}

export function ensurePreferredOnnxBackend() {
  // Registers native ORT once so Transformers.js prefers it over its wasm fallback.
  const symbol = Symbol.for("onnxruntime")
  if (!(symbol in globalThis)) {
    ;(globalThis as Record<PropertyKey, unknown>)[symbol] = ort
  }
}

export function resolveModelDefinition(
  name: string,
  config: ResolvedLocalModelConfig,
  loadOptions?: LocalModelPipelineLoadOptions,
): LocalModelDefinition {
  const registryEntry = config.models[name]

  if (!registryEntry) {
    throw new Error(`Local model "${name}" is not defined in the elysia-local-model config.`)
  }

  return {
    task: registryEntry.task || config.defaultTask,
    model: registryEntry.model || name,
    options: {
      ...registryEntry.options,
      ...loadOptions,
    },
  }
}

export function resolveDecisionModelDefinition(
  name: string,
  config: ResolvedLocalModelConfig,
  overrides?: Partial<DecisionModelLoadOptions>,
): DecisionModelLoadOptions {
  const registryEntry = config.decisionModels[name]

  if (!registryEntry) {
    throw new Error(`Decision model "${name}" is not defined in the elysia-local-model config.`)
  }

  if (!registryEntry.source || typeof registryEntry.source !== "string" || !registryEntry.source.trim()) {
    throw new Error(`Decision model "${name}" must declare a non-empty "source" in the config.`)
  }

  return {
    ...registryEntry,
    ...overrides,
    source: overrides?.source || registryEntry.source,
  }
}
