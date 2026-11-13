import type { DocumentDescriptor, Prediction } from '../types'

export interface PredictionService {
  stream(
    document: DocumentDescriptor,
    signal: AbortSignal,
    onPrediction: (prediction: Prediction) => void,
    options?: { jobIds?: readonly string[] },
  ): Promise<void>
}
