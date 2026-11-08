import { act, renderHook } from '@testing-library/react'
import type { DocumentDescriptor, Prediction } from '../types'
import { createMockPredictionService } from '../services/mockPredictionService'
import { usePredictionSimulation } from './usePredictionSimulation'

const documentOne: DocumentDescriptor = {
  id: 'document-one',
  filename: 'one.pdf',
  source: '/one.pdf',
  mode: 'sample',
}

const documentTwo: DocumentDescriptor = {
  ...documentOne,
  id: 'document-two',
  filename: 'two.pdf',
}

const first: Prediction = { passageId: 'passage-1', page: 1, model: 'Demo LLM', text: 'First prediction.' }
const second: Prediction = { passageId: 'passage-2', page: 2, model: 'Demo LLM', text: 'Second prediction.' }

describe('usePredictionSimulation', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('exposes completed predictions while the remaining passages are still processing', async () => {
    const service = createMockPredictionService({
      predictionsByDocument: { [documentOne.id]: [first, second] },
      delayMs: 100,
    })
    const { result } = renderHook(() => usePredictionSimulation(documentOne, service))
    expect(result.current).toMatchObject({ predictions: [], phase: 'predicting', error: null })

    await act(async () => vi.advanceTimersByTimeAsync(100))
    expect(result.current).toMatchObject({ predictions: [first], phase: 'partial' })

    await act(async () => vi.advanceTimersByTimeAsync(100))
    expect(result.current).toMatchObject({ predictions: [first, second], phase: 'complete' })
  })

  it('represents deterministic empty and failed service outcomes', async () => {
    const emptyService = createMockPredictionService({ predictionsByDocument: {}, delayMs: 50 })
    const empty = renderHook(() => usePredictionSimulation(documentOne, emptyService))
    await act(async () => vi.advanceTimersByTimeAsync(50))
    expect(empty.result.current.phase).toBe('empty')
    empty.unmount()

    const failedService = createMockPredictionService({
      predictionsByDocument: {},
      failureByDocument: { [documentOne.id]: 'Prediction preview unavailable.' },
      delayMs: 50,
    })
    const failed = renderHook(() => usePredictionSimulation(documentOne, failedService))
    await act(async () => vi.advanceTimersByTimeAsync(50))
    expect(failed.result.current).toMatchObject({ phase: 'failed', error: 'Prediction preview unavailable.' })
  })

  it('aborts an old stream before a replacement document can receive stale results', async () => {
    const service = createMockPredictionService({
      predictionsByDocument: {
        [documentOne.id]: [first],
        [documentTwo.id]: [second],
      },
      delayMs: 100,
    })
    const { result, rerender, unmount } = renderHook(
      ({ document }) => usePredictionSimulation(document, service),
      { initialProps: { document: documentOne } },
    )

    rerender({ document: documentTwo })
    await act(async () => vi.advanceTimersByTimeAsync(100))
    expect(result.current.predictions).toEqual([second])

    unmount()
    await act(async () => vi.runAllTimersAsync())
    expect(result.current.predictions).toEqual([second])
  })
})
