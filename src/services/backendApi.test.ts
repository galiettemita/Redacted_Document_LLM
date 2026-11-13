import {
  BackendApiError,
  createBackendApi,
  type AnalysisJobRecord,
  type BackendDocumentRecord,
  type BackendPassageRecord,
  type PassageDetectionResult,
  type VisibleContextSummary,
} from './backendApi'

const documentRecord: BackendDocumentRecord = {
  id: '11111111-1111-4111-8111-111111111111',
  filename: 'record.pdf',
  sha256: 'a'.repeat(64),
  media_type: 'application/pdf',
  page_count: 2,
  status: 'uploaded',
  storage_key: 'sha256/aa/aaaaaaaa',
  visible_date: null,
  date_provenance: null,
  visible_date_evidence: null,
  created_at: '2026-11-11T12:00:00Z',
  expires_at: '2026-11-12T12:00:00Z',
}

const passageRecord: BackendPassageRecord = {
  id: '22222222-2222-4222-8222-222222222222',
  document_id: documentRecord.id,
  label: 'Passage 1',
  page_number: 1,
  rects: [{ x: 0.2, y: 0.3, width: 0.4, height: 0.08 }],
  source: 'manual',
  status: 'marked',
  revision: 1,
  merged_into: null,
  split_from: null,
  created_at: '2026-11-11T12:01:00Z',
  updated_at: '2026-11-11T12:01:00Z',
}

const visibleContext: VisibleContextSummary = {
  passage_id: passageRecord.id,
  passage_revision: 1,
  status: 'context_ready',
  document_date: '1945-05-08',
  length_cue: 'part of a sentence',
  estimated_words: 8,
  visible_word_count: 37,
  confirmed_mask_count: 2,
  serialization_version: 'visible-transcript-v1',
  context_query_status: 'ready',
  context_query_term_count: 4,
  context_query_sha256: '4e64699ee5403e5b9c960206e944cdfde3be184499c2552b508afdaf089746fb',
  context_query_version: 'context-query-v1',
  context_stopwords_version: 'visible-english-v1',
}

const blockedAnalysis: AnalysisJobRecord = {
  id: '33333333-3333-4333-8333-333333333333',
  passage_id: passageRecord.id,
  document_id: documentRecord.id,
  passage_revision: 1,
  status: 'blocked',
  stage: 'readiness',
  blockers: [
    { component: 'corpus', detail: 'corpus exclusion coverage is incomplete' },
  ],
  error_code: null,
  error_detail: null,
  created_at: '2026-11-13T12:00:00Z',
  updated_at: '2026-11-13T12:00:00Z',
}

const detectionResult: PassageDetectionResult = {
  document_id: documentRecord.id,
  status: 'detection_ready',
  pages_scanned: 2,
  coverage_confident: true,
  uncertain_pages: [],
  warnings: [],
  passages: [passageRecord],
}

const documentContexts = {
  document_id: documentRecord.id,
  status: 'contexts_ready' as const,
  passage_count: 1,
  contexts: [visibleContext],
}

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('BackendApi', () => {
  it('uploads the original File as multipart data and preserves the AbortSignal', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(documentRecord, 201))
    const api = createBackendApi(fetchMock)
    const file = new File(['%PDF-1.7'], 'record.pdf', { type: 'application/pdf' })
    const controller = new AbortController()

    await expect(api.uploadDocument(file, controller.signal)).resolves.toEqual(documentRecord)

    expect(fetchMock).toHaveBeenCalledOnce()
    const [url, options] = fetchMock.mock.calls[0]
    expect(url).toBe('/v1/documents')
    expect(options).toMatchObject({ method: 'POST', signal: controller.signal })
    expect(options?.headers).toBeUndefined()
    expect(options?.body).toBeInstanceOf(FormData)
    expect((options?.body as FormData).get('file')).toEqual(file)
  })

  it('creates and updates normalized passage geometry with stable API field names', async () => {
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(passageRecord, 201))
      .mockResolvedValueOnce(jsonResponse({ ...passageRecord, revision: 2 }))
    const api = createBackendApi(fetchMock)
    const rect = { x: 0.2, y: 0.3, width: 0.4, height: 0.08 }

    await api.createPassage(
      documentRecord.id,
      { label: 'Passage 1', page: 1, rects: [rect] },
      { idempotencyKey: 'mark-1' },
    )
    await api.updatePassage(passageRecord.id, { page: 1, rects: [rect] })

    expect(fetchMock).toHaveBeenNthCalledWith(1, `/v1/documents/${documentRecord.id}/passages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Idempotency-Key': 'mark-1',
      },
      body: JSON.stringify({ label: 'Passage 1', page_number: 1, rects: [rect] }),
      signal: undefined,
    })
    expect(fetchMock).toHaveBeenNthCalledWith(2, `/v1/passages/${passageRecord.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ page_number: 1, rects: [rect] }),
      signal: undefined,
    })
  })

  it('starts all-page passage detection for the exact persisted document', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(detectionResult))
    const api = createBackendApi(fetchMock)
    const controller = new AbortController()

    await expect(api.detectDocumentPassages(
      documentRecord.id,
      controller.signal,
    )).resolves.toEqual(detectionResult)

    expect(fetchMock).toHaveBeenCalledWith(
      `/v1/documents/${documentRecord.id}/passages:detect`,
      { method: 'POST', signal: controller.signal },
    )
  })

  it('prepares the exact passage without sending document date metadata', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(visibleContext))
    const api = createBackendApi(fetchMock)
    const controller = new AbortController()

    await expect(api.prepareVisibleContext(
      passageRecord.id,
      controller.signal,
    )).resolves.toEqual(visibleContext)

    expect(fetchMock).toHaveBeenCalledOnce()
    expect(fetchMock).toHaveBeenCalledWith(
      `/v1/passages/${passageRecord.id}:prepare-context`,
      { method: 'POST', signal: controller.signal },
    )
  })

  it('prepares every detected passage through one document request', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(documentContexts))
    const api = createBackendApi(fetchMock) as ReturnType<typeof createBackendApi> & {
      prepareDocumentContexts?: (documentId: string, signal?: AbortSignal) => Promise<unknown>
    }
    const controller = new AbortController()

    expect(typeof api.prepareDocumentContexts).toBe('function')
    await expect(api.prepareDocumentContexts?.(
      documentRecord.id,
      controller.signal,
    )).resolves.toEqual(documentContexts)
    expect(fetchMock).toHaveBeenCalledWith(
      `/v1/documents/${documentRecord.id}:prepare-contexts`,
      { method: 'POST', signal: controller.signal },
    )
  })

  it('schedules the exact prepared passage revision with an idempotency key', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(blockedAnalysis, 202))
    const api = createBackendApi(fetchMock)
    const controller = new AbortController()

    await expect(api.schedulePassageAnalysis(
      passageRecord.id,
      1,
      { signal: controller.signal, idempotencyKey: 'analysis-1' },
    )).resolves.toEqual(blockedAnalysis)

    expect(fetchMock).toHaveBeenCalledWith(
      `/v1/passages/${passageRecord.id}/analysis`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Idempotency-Key': 'analysis-1',
        },
        body: JSON.stringify({ expected_revision: 1 }),
        signal: controller.signal,
      },
    )
  })

  it('maps backend details into a typed error without hiding the status or code', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      detail: { code: 'malformed_pdf', message: 'PDF parser rejected the file' },
    }, 422))
    const api = createBackendApi(fetchMock)

    const failure = await api.uploadDocument(
      new File(['broken'], 'broken.pdf', { type: 'application/pdf' }),
    ).catch((error: unknown) => error)

    expect(failure).toBeInstanceOf(BackendApiError)
    expect(failure).toMatchObject({
      status: 422,
      code: 'malformed_pdf',
      message: 'PDF parser rejected the file',
    })
  })

  it('does not convert an abort into an application failure', async () => {
    const aborted = new DOMException('The operation was aborted.', 'AbortError')
    const fetchMock = vi.fn<typeof fetch>().mockRejectedValue(aborted)
    const api = createBackendApi(fetchMock)

    await expect(api.getReadiness(new AbortController().signal)).rejects.toBe(aborted)
  })

  it('deletes the exact persisted document and accepts an empty 204 response', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(null, { status: 204 }))
    const api = createBackendApi(fetchMock)
    const controller = new AbortController()

    await api.deleteDocument(documentRecord.id, controller.signal)

    expect(fetchMock).toHaveBeenCalledWith(`/v1/documents/${documentRecord.id}`, {
      method: 'DELETE',
      signal: controller.signal,
    })
  })
})
