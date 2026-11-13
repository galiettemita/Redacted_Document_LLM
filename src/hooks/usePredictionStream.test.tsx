import { renderHook, waitFor } from '@testing-library/react'
import type { PredictionService } from '../services/predictionService'
import type { DocumentDescriptor } from '../types'
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
        })
        onPrediction({
          passageId: 'passage-1',
          page: 1,
          model: 'GPT-6 Astra',
          text: 'Replayed result',
          provenance: 'model',
        })
      },
    }

    const jobIds = ['job-1']
    const { result } = renderHook(() => usePredictionStream(document, service, 1, jobIds))

    await waitFor(() => expect(result.current.phase).toBe('complete'))
    expect(result.current.predictions).toEqual([
      expect.objectContaining({ passageId: 'passage-1', text: 'Replayed result' }),
    ])
  })
})
