import type { DocumentDescriptor, Prediction, PredictionBasis } from '../types'

export interface PredictionService {
  stream(
    document: DocumentDescriptor,
    signal: AbortSignal,
    onPrediction: (prediction: Prediction) => void,
    options?: { jobBases?: Readonly<Record<string, PredictionBasis>> },
  ): Promise<void>
}
