import { useCallback, useEffect, useRef, useState } from 'react'
import type { PredictionService } from '../services/predictionService'
import type { DocumentDescriptor, Prediction, PredictionPhase } from '../types'

export function usePredictionSimulation(
  document: DocumentDescriptor | null,
  service: PredictionService,
) {
  const controllerRef = useRef<AbortController | null>(null)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [phase, setPhase] = useState<PredictionPhase>(document ? 'predicting' : 'idle')
  const [error, setError] = useState<string | null>(null)

  const reset = useCallback(() => {
    controllerRef.current?.abort()
    controllerRef.current = null
    setPredictions([])
    setPhase('idle')
    setError(null)
  }, [])

  useEffect(() => {
    if (!document) {
      reset()
      return
    }

    const controller = new AbortController()
    controllerRef.current = controller
    let completedCount = 0
    setPredictions([])
    setPhase('predicting')
    setError(null)

    void service.stream(document, controller.signal, (prediction) => {
      if (controller.signal.aborted) return
      completedCount += 1
      setPredictions((current) => [...current, prediction])
      setPhase('partial')
    }).then(() => {
      if (controller.signal.aborted) return
      setPhase(completedCount > 0 ? 'complete' : 'empty')
    }).catch((reason: unknown) => {
      if (controller.signal.aborted) return
      setError(reason instanceof Error ? reason.message : 'Predictions are unavailable.')
      setPhase('failed')
    })

    return () => controller.abort()
  }, [document, reset, service])

  return { predictions, phase, error, reset }
}
