import type { Prediction, PredictionBasis } from '../../types'

const BASIS_LABELS: Record<PredictionBasis, string> = {
  model_only_v1: 'Model-only prediction — no archive evidence used',
  best_available_foiarchive_v1: 'Archive-assisted prediction — available FOIArchive snapshot used',
  paper_exact_v1: 'Paper-method prediction',
}

export function predictionProvenanceLabel(prediction: Prediction): string {
  if (prediction.provenance === 'simulated') return 'Simulated prediction'
  return BASIS_LABELS[prediction.predictionBasis]
}
