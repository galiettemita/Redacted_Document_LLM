import { createHttpPredictionService } from './httpPredictionService'

describe('createHttpPredictionService', () => {
  it('maps a replayable SSE prediction to the real passage ID', async () => {
    const body = [
      'id: 4',
      'event: passage.status',
      'data: {"passage_id":"passage-1","status":"running"}',
      '',
      'id: 5',
      'event: passage.prediction',
      'data: {"job_id":"job-1","passage_id":"passage-1","page_number":2,"model":"gpt-6-astra","prediction_basis":"model_only_v1","prediction":"Recovered archival sentence.","usage":{"input_tokens":2000,"output_tokens":120,"total_tokens":2120,"reasoning_tokens":30},"estimated_cost_usd":0.026,"pricing_version":"2026-11-13"}',
      '',
      '',
    ].join('\n')
    const fetchMock = vi.fn(async () => new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    }))
    const service = createHttpPredictionService(fetchMock as typeof fetch)
    const predictions: unknown[] = []

    await service.stream(
      {
        id: 'uploaded-1',
        filename: 'archive.pdf',
        source: 'blob:test',
        mode: 'uploaded',
        backendDocumentId: 'document-1',
      },
      new AbortController().signal,
      (prediction) => predictions.push(prediction),
      { jobBases: { 'job-1': 'model_only_v1' } },
    )

    expect(fetchMock).toHaveBeenCalledWith(
      '/v1/documents/document-1/events',
      expect.objectContaining({
        headers: { Accept: 'text/event-stream' },
      }),
    )
    expect(predictions).toEqual([{
      passageId: 'passage-1',
      page: 2,
      model: 'GPT-6 Astra',
      text: 'Recovered archival sentence.',
      provenance: 'model',
      predictionBasis: 'model_only_v1',
      usage: {
        inputTokens: 2000,
        outputTokens: 120,
        totalTokens: 2120,
        reasoningTokens: 30,
        estimatedCostUsd: 0.026,
        pricingVersion: '2026-11-13',
      },
    }])
  })

  it('ignores prediction events from an older run of the same document', async () => {
    const body = [
      'event: passage.prediction',
      'data: {"job_id":"old-job","passage_id":"passage-1","page_number":1,"model":"gpt-6-astra","prediction_basis":"paper_exact_v1","prediction":"Stale result.","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"reasoning_tokens":0},"estimated_cost_usd":0.001,"pricing_version":"test"}',
      '',
      'event: passage.prediction',
      'data: {"job_id":"new-job","passage_id":"passage-1","page_number":1,"model":"gpt-6-astra","prediction_basis":"paper_exact_v1","prediction":"Current result.","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"reasoning_tokens":0},"estimated_cost_usd":0.001,"pricing_version":"test"}',
      '',
      '',
    ].join('\n')
    const service = createHttpPredictionService(vi.fn(async () => new Response(body)) as typeof fetch)
    const results: string[] = []

    await service.stream(
      {
        id: 'uploaded-1',
        filename: 'archive.pdf',
        source: 'blob:test',
        mode: 'uploaded',
        backendDocumentId: 'document-1',
      },
      new AbortController().signal,
      (prediction) => results.push(prediction.text),
      { jobBases: { 'new-job': 'paper_exact_v1' } },
    )

    expect(results).toEqual(['Current result.'])
  })

  it('accepts the staged BM25 basis only when it matches the scheduled job', async () => {
    const body = [
      'event: passage.prediction',
      'data: {"job_id":"job-1","passage_id":"passage-1","page_number":1,"model":"gpt-6-astra","prediction_basis":"best_available_foiarchive_bm25_v1","prediction":"Evidence-bound result.","usage":{"input_tokens":100,"output_tokens":10,"total_tokens":110,"reasoning_tokens":0},"estimated_cost_usd":0.002,"pricing_version":"test"}',
      '',
      '',
    ].join('\n')
    const service = createHttpPredictionService(
      vi.fn(async () => new Response(body)) as typeof fetch,
    )
    const predictions: unknown[] = []

    await service.stream(
      {
        id: 'uploaded-1',
        filename: 'archive.pdf',
        source: 'blob:test',
        mode: 'uploaded',
        backendDocumentId: 'document-1',
      },
      new AbortController().signal,
      (prediction) => predictions.push(prediction),
      { jobBases: { 'job-1': 'best_available_foiarchive_bm25_v1' } },
    )

    expect(predictions).toEqual([
      expect.objectContaining({
        text: 'Evidence-bound result.',
        predictionBasis: 'best_available_foiarchive_bm25_v1',
      }),
    ])
  })

  it('rejects malformed prediction events instead of displaying them', async () => {
    const fetchMock = vi.fn(async () => new Response(
      'event: passage.prediction\ndata: {"passage_id":"passage-1"}\n\n',
      { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
    ))
    const service = createHttpPredictionService(fetchMock as typeof fetch)

    await expect(service.stream(
      {
        id: 'uploaded-1',
        filename: 'archive.pdf',
        source: 'blob:test',
        mode: 'uploaded',
        backendDocumentId: 'document-1',
      },
      new AbortController().signal,
      vi.fn(),
    )).rejects.toThrow('invalid prediction event')
  })

  it.each([
    ['missing', ''],
    ['unknown', ',"prediction_basis":"experimental_v2"'],
  ])('rejects a %s prediction basis instead of guessing provenance', async (_name, basis) => {
    const event = `event: passage.prediction\ndata: {"job_id":"job-1","passage_id":"passage-1","page_number":1,"model":"gpt-6-astra"${basis},"prediction":"Unsafe result.","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"reasoning_tokens":0},"estimated_cost_usd":0.001,"pricing_version":"test"}\n\n`
    const service = createHttpPredictionService(
      vi.fn(async () => new Response(event)) as typeof fetch,
    )

    await expect(service.stream(
      {
        id: 'uploaded-1',
        filename: 'archive.pdf',
        source: 'blob:test',
        mode: 'uploaded',
        backendDocumentId: 'document-1',
      },
      new AbortController().signal,
      vi.fn(),
    )).rejects.toThrow('invalid prediction event')
  })

  it('rejects a recognized basis that differs from the scheduled job', async () => {
    const event = [
      'event: passage.prediction',
      'data: {"job_id":"job-1","passage_id":"passage-1","page_number":1,"model":"gpt-6-astra","prediction_basis":"paper_exact_v1","prediction":"Mismatched result.","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"reasoning_tokens":0},"estimated_cost_usd":0.001,"pricing_version":"test"}',
      '',
      '',
    ].join('\n')
    const service = createHttpPredictionService(
      vi.fn(async () => new Response(event)) as typeof fetch,
    )
    const onPrediction = vi.fn()

    await expect(service.stream(
      {
        id: 'uploaded-1',
        filename: 'archive.pdf',
        source: 'blob:test',
        mode: 'uploaded',
        backendDocumentId: 'document-1',
      },
      new AbortController().signal,
      onPrediction,
      { jobBases: { 'job-1': 'model_only_v1' } },
    )).rejects.toThrow('prediction basis changed')
    expect(onPrediction).not.toHaveBeenCalled()
  })

  it('reports worker error events after consuming the terminal stream', async () => {
    const body = [
      'id: 8',
      'event: passage.error',
      'data: {"job_id":"job-1","passage_id":"passage-1","code":"rate_limited","detail":"Provider retry budget was exhausted"}',
      '',
      '',
    ].join('\n')
    const service = createHttpPredictionService(
      vi.fn(async () => new Response(body)) as typeof fetch,
    )

    await expect(service.stream(
      {
        id: 'uploaded-1',
        filename: 'archive.pdf',
        source: 'blob:test',
        mode: 'uploaded',
        backendDocumentId: 'document-1',
      },
      new AbortController().signal,
      vi.fn(),
      { jobBases: { 'job-1': 'model_only_v1' } },
    )).rejects.toThrow('Provider retry budget was exhausted')
  })

  it('resumes a dropped stream from its last event ID', async () => {
    let pullCount = 0
    const interrupted = new ReadableStream<Uint8Array>({
      pull(controller) {
        pullCount += 1
        if (pullCount === 1) {
          controller.enqueue(new TextEncoder().encode([
            'id: 11',
            'event: passage.status',
            'data: {"job_id":"job-1","status":"running"}',
            '',
            '',
          ].join('\n')))
        } else {
          controller.error(new Error('connection dropped'))
        }
      },
    })
    const completed = [
      'id: 12',
      'event: passage.prediction',
      'data: {"job_id":"job-1","passage_id":"passage-1","page_number":1,"model":"gpt-6-astra","prediction_basis":"best_available_foiarchive_v1","prediction":"Resumed result.","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"reasoning_tokens":0},"estimated_cost_usd":0.001,"pricing_version":"test"}',
      '',
      '',
    ].join('\n')
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(interrupted))
      .mockResolvedValueOnce(new Response(completed))
    const service = createHttpPredictionService(fetchMock as typeof fetch)
    const results: string[] = []

    await service.stream(
      {
        id: 'uploaded-1',
        filename: 'archive.pdf',
        source: 'blob:test',
        mode: 'uploaded',
        backendDocumentId: 'document-1',
      },
      new AbortController().signal,
      (prediction) => results.push(prediction.text),
      { jobBases: { 'job-1': 'best_available_foiarchive_v1' } },
    )

    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/v1/documents/document-1/events',
      expect.objectContaining({
        headers: {
          Accept: 'text/event-stream',
          'Last-Event-ID': '11',
        },
      }),
    )
    expect(results).toEqual(['Resumed result.'])
  })
})
