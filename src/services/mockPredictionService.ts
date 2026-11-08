import type { Prediction } from '../types'
import type { PredictionService } from './predictionService'

interface MockPredictionServiceOptions {
  predictionsByDocument: Record<string, Prediction[] | undefined>
  failureByDocument?: Record<string, string | undefined>
  delayMs?: number
}

function waitForDelay(delayMs: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('Prediction stream aborted.', 'AbortError'))
      return
    }

    const timer = window.setTimeout(resolve, delayMs)
    signal.addEventListener('abort', () => {
      window.clearTimeout(timer)
      reject(new DOMException('Prediction stream aborted.', 'AbortError'))
    }, { once: true })
  })
}

export function createMockPredictionService({
  predictionsByDocument,
  failureByDocument = {},
  delayMs = 650,
}: MockPredictionServiceOptions): PredictionService {
  return {
    async stream(document, signal, onPrediction) {
      const failure = failureByDocument[document.id]
      const predictions = predictionsByDocument[document.id] ?? []

      if (failure) {
        await waitForDelay(delayMs, signal)
        throw new Error(failure)
      }

      if (predictions.length === 0) {
        await waitForDelay(delayMs, signal)
        return
      }

      for (const prediction of predictions) {
        await waitForDelay(delayMs, signal)
        if (signal.aborted) return
        onPrediction(prediction)
      }
    },
  }
}
