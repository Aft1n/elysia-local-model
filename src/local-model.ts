import { pipeline } from "@huggingface/transformers"
import type { LocalModelPipeline } from "./types"
import {
  applyLocalModelEnvironment,
  ensurePreferredOnnxBackend,
  resolveModelDefinition,
  type ResolvedLocalModelConfig,
} from "./utils"

const modelCache = new Map<string, Promise<LocalModelPipeline>>()

// Transformers.js types `pipeline` per-task; the registry stores task strings.
const createPipeline = pipeline as unknown as (
  task: string,
  model: string,
  options?: Record<string, unknown>,
) => Promise<LocalModelPipeline>

/**
 * Load (and cache) the pipeline behind a configured model alias. The returned
 * callable takes `(input, callOptions?)` like a native Transformers.js
 * pipeline; `dispose()` releases the model and evicts the cache entry.
 */
export async function loadLocalModel(
  name: string,
  config: ResolvedLocalModelConfig,
): Promise<LocalModelPipeline> {
  ensurePreferredOnnxBackend()
  applyLocalModelEnvironment(config)

  const definition = resolveModelDefinition(name, config)
  const key = [name, definition.task, definition.model, config.cacheDir].join("::")

  if (!modelCache.has(key)) {
    const modelPromise = (async () => {
      const loaded = await createPipeline(definition.task, definition.model, definition.options || {})
      const dispose = loaded.dispose?.bind(loaded)
      return Object.assign(loaded, {
        dispose: async () => {
          await dispose?.()
          modelCache.delete(key)
        },
      })
    })().catch((error) => {
      modelCache.delete(key)
      throw error
    })

    modelCache.set(key, modelPromise)
  }

  return modelCache.get(key) as Promise<LocalModelPipeline>
}
