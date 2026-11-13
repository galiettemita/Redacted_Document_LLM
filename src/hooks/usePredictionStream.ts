import { useEffect, useState } from 'react'
import type { PredictionService } from '../services/predictionService'
import type { DocumentDescriptor, Prediction, PredictionPhase } from '../types'

export function usePredictionStream(
  document: DocumentDescriptor | null,
  service: PredictionService,
  runKey: number,
  jobIds: readonly string[] = [],
) {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [phase, setPhase] = useState<PredictionPhase>('idle')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!document || runKey < 1) {
      setPredictions([])
      setPhase('idle')
      setError(null)
      return
    }
    const controller = new AbortController()
    let completedCount = 0
    setPredictions([])
    setPhase('predicting')
    setError(null)

    void service.stream(document, controller.signal, (prediction) => {
      if (controller.signal.aborted) return
      completedCount += 1
      setPredictions((current) => {
        const existing = current.findIndex(
          (item) => item.passageId === prediction.passageId,
        )
        if (existing < 0) return [...current, prediction]
        return current.map((item, index) => (
          index === existing ? prediction : item
        ))
      })
      setPhase('partial')
    }, { jobIds }).then(() => {
      if (controller.signal.aborted) return
      setPhase(completedCount > 0 ? 'complete' : 'empty')
    }).catch((reason: unknown) => {
      if (controller.signal.aborted) return
      setError(reason instanceof Error ? reason.message : 'Predictions are unavailable.')
      setPhase('failed')
    })

    return () => controller.abort()
  }, [document, jobIds, runKey, service])

  return { predictions, phase, error }
}
