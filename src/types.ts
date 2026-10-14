import type { PipelineType as TransformersPipelineType, PretrainedModelOptions } from "@huggingface/transformers"

// ---------------------------------------------------------------------------
// Transformers.js pipelines
// ---------------------------------------------------------------------------

export type LocalModelKnownTask = TransformersPipelineType
export type LocalModelTask = LocalModelKnownTask | (string & {})

type LocalModelCallable = (...args: unknown[]) => Promise<unknown>

export type LocalModelPipeline = LocalModelCallable & {
  dispose?: () => Promise<void> | void
}

export type LocalModelRunner = LocalModelPipeline

export type LocalModelPipelineLoadOptions = PretrainedModelOptions

export type LocalModelPrewarmTargets = boolean | string[]

export interface LocalModelDefinition<TTask extends LocalModelTask = LocalModelTask> {
  task: TTask
  model: string
  options?: LocalModelPipelineLoadOptions
}

export type LocalModelModelRegistry = Record<string, LocalModelDefinition>

export type LocalModelName = string

export type LocalModelConfig = {
  /** Named model aliases: `alias -> { task, model, options }`. */
  models?: LocalModelModelRegistry
  /** Named decision model definitions, keyed like `models`. */
  decisionModels?: DecisionModelRegistry
  /** Where downloaded weights are cached (default `./.ai-models`). */
  cacheDir?: string
  allowRemoteModels?: boolean
  allowLocalModels?: boolean
  /** Pipeline task used when a model entry omits `task` (default `feature-extraction`). */
  defaultTask?: LocalModelTask
  /** Warm pipelines on server start: `true` for all aliases or a list of aliases. */
  prewarm?: LocalModelPrewarmTargets
}

// ---------------------------------------------------------------------------
// Decision models (typed judgements, executed locally over ONNX)
// ---------------------------------------------------------------------------

export type DecisionModelState = string | Record<string, unknown> | unknown[]

export interface DecisionNoulQuestion {
  type: "noul"
  instructions: string
  /** Meanings for the two outcomes; defaults to generic Yes/No phrasing. */
  criteria?: { true?: string, false?: string, yes?: string, no?: string }
}

export interface DecisionChoiceQuestion {
  type: "choice"
  instructions: string
  criteria: Record<string, string>
}

export interface DecisionScoreQuestion {
  type: "score"
  instructions: string
  criteria: string[]
}

export type DecisionQuestion = DecisionNoulQuestion | DecisionChoiceQuestion | DecisionScoreQuestion
export type DecisionQuestionMap = Record<string, DecisionQuestion>

export interface DecisionAnswerBase {
  /** 0..1. Present for `choice` and `score`, where the answer is a distribution. */
  confidence?: number
  /** Per-option/per-level probabilities. */
  probabilities?: Record<string, number>
}

export interface DecisionNoulAnswer extends DecisionAnswerBase {
  type: "noul"
  noul: number
}

export interface DecisionChoiceAnswer extends DecisionAnswerBase {
  type: "choice"
  choice: string
  probabilities: Record<string, number>
}

export interface DecisionScoreAnswer extends DecisionAnswerBase {
  type: "score"
  score: number
  probabilities: Record<string, number>
  /** String level index -> level label, mirroring the question's criteria. */
  legend?: Record<string, string>
}

export type DecisionAnswer = DecisionNoulAnswer | DecisionChoiceAnswer | DecisionScoreAnswer
export type DecisionAnswers = Record<string, DecisionAnswer>

export interface DecisionRequest {
  state: DecisionModelState
  questions: DecisionQuestionMap
}

export interface DecisionUsage {
  stateTokens?: number
  truncated?: boolean
}

export interface DecisionResult {
  answers: DecisionAnswers
  usage?: DecisionUsage
}

export interface DecisionDecideOptions {
  /** Abort a request that outlives its budget. */
  signal?: AbortSignal
}

/**
 * A decision engine. Implement this over any local runtime (ONNX Runtime, a
 * vendored engine, a remote gateway) and register it in the plugin config.
 * The plugin ships default ONNX adapters, so users only implement this when
 * they want a different runtime.
 */
export interface DecisionModelAdapter {
  decide(request: DecisionRequest, options?: DecisionDecideOptions): Promise<DecisionResult>
  dispose?(): Promise<void> | void
}

export interface DecisionModelLoadOptions {
  /**
   * Where the model lives. Three forms: a local directory (or `.onnx` file),
   * a full `https://` URL (`.onnx` file or a shared-prefix manifest directory),
   * or a Hugging Face id such as `my-org/my-decision-model`
   * (optionally `id@revision`) resolved to `…/resolve/<revision>/`.
   * Shared-prefix models point at the manifest directory (or the HF id);
   * the adapter reads `manifest.json` + tokenizer + model from it.
   */
  source: string
  /**
   * Hugging Face revision (branch, tag or commit) used when `source` is a
   * Hugging Face id. Defaults to `main`.
   */
  revision?: string
  /**
   * Export subdirectory inside the repo for bare HF ids with the
   * shared-prefix adapter (default `onnx/`). Full URLs and local paths
   * already point at the directory, so they ignore it.
   */
  subdir?: string
  /**
   * Fetch implementation used to download remote models. Defaults to
   * `globalThis.fetch`; injectable for tests and for hosts proxying weights.
   */
  fetch?: typeof fetch
  /**
   * Adapter factory path, resolved at runtime — or `"shared-prefix"` for the
   * bundled manifest-directory adapter, which runs exports with one prompt
   * prefix per question. Defaults to the bundled ONNX adapter.
   */
  adapter?: string
  /**
   * Hugging Face id of the tokenizer matching this checkpoint, e.g.
   * `"Xenova/all-MiniLM-L6-v2"`. Defaults to that MiniLM tokenizer when unset.
   */
  tokenizer?: string
  /**
   * ONNX output name -> question id map. Only read by the bundled ONNX adapter:
   * real checkpoints name their outputs `logits`, `choice_0`, `head`, or
   * `output.1` rather than after the question id.
   */
  outputMap?: Record<string, string>
  /** Token budget passed to the tokenizer as its truncation limit. */
  maxTokens?: number
  /** Passed through to the adapter untouched. */
  sessionOptions?: Record<string, unknown>
}

export type DecisionModelDefinition = DecisionModelLoadOptions

export type DecisionModelRegistry = Record<string, DecisionModelDefinition>
