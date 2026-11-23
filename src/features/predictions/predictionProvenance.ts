import type { Prediction, PredictionBasis } from '../../types'

const BASIS_LABELS: Record<Exclude<PredictionBasis, 'best_available_foiarchive_bm25_v1'>, string> = {
  model_only_v1: 'Model-only prediction — no archive evidence used',
  best_available_foiarchive_v1: 'Archive-assisted prediction — available FOIArchive snapshot used',
  paper_exact_v1: 'Paper-method prediction',
}

export function predictionProvenanceLabel(prediction: Prediction): string | null {
  if (prediction.provenance === 'simulated') return 'Simulated prediction'
  if (prediction.predictionBasis === 'best_available_foiarchive_bm25_v1') return null
  return BASIS_LABELS[prediction.predictionBasis]
}
