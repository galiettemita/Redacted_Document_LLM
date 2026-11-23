export interface PassageRect {
  x: number
  y: number
  width: number
  height: number
}

export interface Passage {
  id: string
  label: string
  page: number
  rects: PassageRect[]
}

export type PredictionBasis =
  | 'model_only_v1'
  | 'best_available_foiarchive_bm25_v1'
  | 'best_available_foiarchive_v1'
  | 'paper_exact_v1'

interface PredictionBase {
  passageId: string
  page: number
  model: string
  text: string
  usage?: {
    inputTokens: number
    outputTokens: number
    totalTokens: number
    reasoningTokens: number
    estimatedCostUsd: number
    pricingVersion: string
  }
}

export type Prediction = PredictionBase & (
  | {
    provenance: 'simulated'
    predictionBasis?: never
  }
  | {
    provenance: 'model'
    predictionBasis: PredictionBasis
  }
)

export type DocumentMode = 'sample' | 'uploaded'

export interface DocumentDescriptor {
  id: string
  filename: string
  source: string
  mode: DocumentMode
  pageCount?: number
  label?: string
  backendDocumentId?: string
}

export type PredictionPhase = 'idle' | 'predicting' | 'partial' | 'complete' | 'empty' | 'failed'
