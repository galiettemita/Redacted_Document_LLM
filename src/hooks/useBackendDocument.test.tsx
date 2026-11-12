import { act, renderHook, waitFor } from '@testing-library/react'
import { BackendApiError, type BackendApi, type BackendDocumentRecord } from '../services/backendApi'
import { useBackendDocument } from './useBackendDocument'

const record: BackendDocumentRecord = {
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

function apiWith(overrides: Partial<BackendApi> = {}): BackendApi {
  return {
    getReadiness: vi.fn(),
    uploadDocument: vi.fn().mockResolvedValue(record),
    deleteDocument: vi.fn().mockResolvedValue(undefined),
    updateDocumentMetadata: vi.fn(),
    createPassage: vi.fn(),
    updatePassage: vi.fn(),
    prepareVisibleContext: vi.fn(),
    ...overrides,
  }
}

describe('useBackendDocument', () => {
  it('reports uploading immediately and then exposes the persisted record', async () => {
    let resolveUpload!: (value: BackendDocumentRecord) => void
    const uploadDocument = vi.fn(() => new Promise<BackendDocumentRecord>((resolve) => {
      resolveUpload = resolve
    }))
    const api = apiWith({ uploadDocument })
    const file = new File(['%PDF-1.7'], 'record.pdf', { type: 'application/pdf' })

    const { result } = renderHook(() => useBackendDocument(file, api))
    expect(result.current.status).toBe('uploading')

    await act(async () => resolveUpload(record))
    await waitFor(() => expect(result.current.status).toBe('ready'))
    expect(result.current.record).toEqual(record)
  })

  it('distinguishes an unavailable backend from a rejected PDF', async () => {
    const file = new File(['%PDF-1.7'], 'record.pdf', { type: 'application/pdf' })
    const unavailableApi = apiWith({
      uploadDocument: vi.fn().mockRejectedValue(
        new BackendApiError('The local backend is unavailable.', { status: 502, code: 'backend_error' }),
      ),
    })
    const rejectedApi = apiWith({
      uploadDocument: vi.fn().mockRejectedValue(
        new BackendApiError('PDF parser rejected the file', { status: 422, code: 'malformed_pdf' }),
      ),
    })

    const unavailable = renderHook(() => useBackendDocument(file, unavailableApi))
    await waitFor(() => expect(unavailable.result.current.status).toBe('unavailable'))
    expect(unavailable.result.current.error).toBe(
      'The local backend is unavailable. Start it and choose the PDF again.',
    )
    const rejected = renderHook(() => useBackendDocument(file, rejectedApi))
    await waitFor(() => expect(rejected.result.current.status).toBe('failed'))
  })

  it('deletes the persisted copy before reporting that removal is complete', async () => {
    let resolveDelete!: () => void
    const deleteDocument = vi.fn(() => new Promise<void>((resolve) => {
      resolveDelete = resolve
    }))
    const api = apiWith({ deleteDocument })
    const file = new File(['%PDF-1.7'], 'record.pdf', { type: 'application/pdf' })
    const { result } = renderHook(() => useBackendDocument(file, api))
    await waitFor(() => expect(result.current.status).toBe('ready'))

    let removal!: Promise<void>
    act(() => {
      removal = result.current.remove()
    })
    expect(result.current.status).toBe('deleting')
    expect(deleteDocument).toHaveBeenCalledWith(record.id)

    await act(async () => {
      resolveDelete()
      await removal
    })
    expect(result.current.status).toBe('idle')
  })

  it('waits for an in-flight upload and deletes the late backend record', async () => {
    let resolveUpload!: (value: BackendDocumentRecord) => void
    let observedSignal: AbortSignal | undefined
    const uploadDocument = vi.fn((_file: File, signal?: AbortSignal) => {
      observedSignal = signal
      return new Promise<BackendDocumentRecord>((resolve) => {
        resolveUpload = resolve
      })
    })
    const deleteDocument = vi.fn().mockResolvedValue(undefined)
    const api = apiWith({ uploadDocument, deleteDocument })
    const file = new File(['%PDF-1.7'], 'record.pdf', { type: 'application/pdf' })
    const { result } = renderHook(() => useBackendDocument(file, api))

    let removal!: Promise<void>
    act(() => {
      removal = result.current.remove()
    })

    expect(result.current.status).toBe('deleting')
    expect(observedSignal?.aborted).toBe(false)
    expect(deleteDocument).not.toHaveBeenCalled()

    await act(async () => {
      resolveUpload(record)
      await removal
    })

    expect(deleteDocument).toHaveBeenCalledWith(record.id)
    expect(result.current.status).toBe('idle')
  })

  it('aborts an upload when the selected file is cleared', async () => {
    let observedSignal: AbortSignal | undefined
    const uploadDocument = vi.fn((_file: File, signal?: AbortSignal) => {
      observedSignal = signal
      return new Promise<BackendDocumentRecord>(() => undefined)
    })
    const api = apiWith({ uploadDocument })
    const file = new File(['%PDF-1.7'], 'record.pdf', { type: 'application/pdf' })
    const { rerender } = renderHook(
      ({ selected }) => useBackendDocument(selected, api),
      { initialProps: { selected: file as File | null } },
    )

    expect(observedSignal?.aborted).toBe(false)
    rerender({ selected: null })
    expect(observedSignal?.aborted).toBe(true)
  })
})
