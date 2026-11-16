import type { DocumentDescriptor, Prediction } from '../types'
import { createMockPredictionService } from './mockPredictionService'

const demoDocument: DocumentDescriptor = {
  id: 'fictional-demo',
  filename: 'fictional.pdf',
  source: '/fictional.pdf',
  mode: 'sample',
}

const firstPrediction: Prediction = {
  passageId: 'passage-1',
  page: 1,
  model: 'Demo LLM',
  text: 'First simulated result.',
  provenance: 'simulated',
}

const secondPrediction: Prediction = {
  passageId: 'passage-2',
  page: 2,
  model: 'Demo LLM',
  text: 'Second simulated result.',
  provenance: 'simulated',
}

describe('createMockPredictionService', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('emits fictional predictions progressively in document order', async () => {
    const received: Prediction[] = []
    const service = createMockPredictionService({
      predictionsByDocument: {
        [demoDocument.id]: [firstPrediction, secondPrediction],
      },
      delayMs: 100,
    })
    const completion = service.stream(
      demoDocument,
      new AbortController().signal,
      (prediction) => received.push(prediction),
    )

    await vi.advanceTimersByTimeAsync(100)
    expect(received).toEqual([firstPrediction])

    await vi.advanceTimersByTimeAsync(100)
    await completion
    expect(received).toEqual([firstPrediction, secondPrediction])
  })

  it('stops an active fictional stream when its owner aborts', async () => {
    const received: Prediction[] = []
    const controller = new AbortController()
    const service = createMockPredictionService({
      predictionsByDocument: {
        [demoDocument.id]: [firstPrediction, secondPrediction],
      },
      delayMs: 100,
    })
    const completion = service.stream(
      demoDocument,
      controller.signal,
      (prediction) => received.push(prediction),
    )
    const aborted = expect(completion).rejects.toMatchObject({ name: 'AbortError' })

    await vi.advanceTimersByTimeAsync(100)
    controller.abort()
    await vi.runAllTimersAsync()

    await aborted
    expect(received).toEqual([firstPrediction])
  })

  it('returns empty and configured failure outcomes without inventing results', async () => {
    const empty = createMockPredictionService({
      predictionsByDocument: {},
      delayMs: 50,
    })
    const emptyCompletion = empty.stream(
      demoDocument,
      new AbortController().signal,
      vi.fn(),
    )
    await vi.advanceTimersByTimeAsync(50)
    await expect(emptyCompletion).resolves.toBeUndefined()

    const failed = createMockPredictionService({
      predictionsByDocument: {},
      failureByDocument: { [demoDocument.id]: 'Demo unavailable.' },
      delayMs: 50,
    })
    const failedCompletion = failed.stream(
      demoDocument,
      new AbortController().signal,
      vi.fn(),
    )
    const failedOutcome = expect(failedCompletion).rejects.toThrow('Demo unavailable.')
    await vi.advanceTimersByTimeAsync(50)
    await failedOutcome
  })
})
