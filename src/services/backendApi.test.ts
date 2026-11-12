import {
  BackendApiError,
  createBackendApi,
  type BackendDocumentRecord,
  type BackendPassageRecord,
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
