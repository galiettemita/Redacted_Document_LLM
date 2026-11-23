import { renderHook, waitFor } from '@testing-library/react'
import type { PredictionService } from '../services/predictionService'
import type { DocumentDescriptor, PredictionBasis } from '../types'
import { usePredictionStream } from './usePredictionStream'

const document: DocumentDescriptor = {
  id: 'uploaded-1',
  filename: 'archive.pdf',
  source: 'blob:test',
  mode: 'uploaded',
  backendDocumentId: 'document-1',
}

describe('usePredictionStream', () => {
  it('keeps one latest model result per passage across replayed events', async () => {
    const service: PredictionService = {
      async stream(_document, _signal, onPrediction) {
        onPrediction({
          passageId: 'passage-1',
          page: 1,
          model: 'GPT-6 Astra',
          text: 'First result',
          provenance: 'model',
          predictionBasis: 'model_only_v1',
        })
        onPrediction({
          passageId: 'passage-1',
          page: 1,
          model: 'GPT-6 Astra',
          text: 'Replayed result',
          provenance: 'model',
          predictionBasis: 'model_only_v1',
        })
      },
    }

    const jobBases = { 'job-1': 'model_only_v1' } as const
    const { result } = renderHook(() => (
      usePredictionStream(document, service, 1, jobBases)
    ))

    await waitFor(() => expect(result.current.phase).toBe('complete'))
    expect(result.current.predictions).toEqual([
      expect.objectContaining({ passageId: 'passage-1', text: 'Replayed result' }),
    ])
  })

  it('keeps completed passage results when a later selected passage runs', async () => {
    const service: PredictionService = {
      async stream(_document, _signal, onPrediction, options) {
        const jobId = Object.keys(options?.jobBases ?? {})[0]
        onPrediction({
          passageId: jobId === 'job-2' ? 'passage-2' : 'passage-1',
          page: jobId === 'job-2' ? 2 : 1,
          model: 'GPT-6 Astra',
          text: jobId === 'job-2' ? 'Second result' : 'First result',
          provenance: 'model',
          predictionBasis: 'model_only_v1',
        })
      },
    }
    const { result, rerender } = renderHook(
      ({ runKey, jobBases }) => usePredictionStream(
        document,
        service,
        runKey,
        jobBases,
      ),
      {
        initialProps: {
          runKey: 1,
          jobBases: {
            'job-1': 'model_only_v1',
          } as Record<string, PredictionBasis>,
        },
      },
    )

    await waitFor(() => expect(result.current.phase).toBe('complete'))
    rerender({
      runKey: 2,
      jobBases: { 'job-2': 'model_only_v1' } as const,
    })

    await waitFor(() => expect(result.current.predictions).toHaveLength(2))
    expect(result.current.predictions).toEqual([
      expect.objectContaining({ passageId: 'passage-1', text: 'First result' }),
      expect.objectContaining({ passageId: 'passage-2', text: 'Second result' }),
    ])
  })
})
