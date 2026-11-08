import type { DocumentDescriptor, Prediction } from '../types'

export interface PredictionService {
  stream(
    document: DocumentDescriptor,
    signal: AbortSignal,
    onPrediction: (prediction: Prediction) => void,
  ): Promise<void>
}
