import type { PassageRect } from '../types'

export interface ComponentReadiness {
  ready: boolean
  detail: string
}

export interface BackendReadiness {
  http: ComponentReadiness
  database: ComponentReadiness
  worker: ComponentReadiness
  corpus: ComponentReadiness
  model: ComponentReadiness
  prediction_ready: boolean
}

export interface BackendDocumentRecord {
  id: string
  filename: string
  sha256: string
  media_type: string
  page_count: number
  status: 'uploaded' | 'extracting' | 'ready' | 'failed' | 'deleted'
  storage_key: string
  visible_date: string | null
  date_provenance: 'visible_text' | 'pdf_metadata_corroborated' | 'manual_visible_document' | null
  created_at: string
  expires_at: string
}

export interface BackendPassageRecord {
  id: string
  document_id: string
  label: string
  page_number: number
  rects: PassageRect[]
  source: 'manual' | 'detected'
  status: 'marked' | 'context_ready' | 'retrieving' | 'predicting' | 'complete' | 'failed' | 'deleted'
  revision: number
  merged_into: string | null
  split_from: string | null
  created_at: string
  updated_at: string
}

export interface PassageMutation {
  label?: string
  page: number
  rects: PassageRect[]
}

interface PassageRequestOptions {
  signal?: AbortSignal
  idempotencyKey?: string
}

interface BackendApiErrorOptions {
  status?: number
  code?: string
  cause?: unknown
}

export class BackendApiError extends Error {
  readonly status: number | null
  readonly code: string

  constructor(message: string, options: BackendApiErrorOptions = {}) {
    super(message, { cause: options.cause })
    this.name = 'BackendApiError'
    this.status = options.status ?? null
    this.code = options.code ?? 'backend_error'
  }
}

export interface BackendApi {
  getReadiness(signal?: AbortSignal): Promise<BackendReadiness>
  uploadDocument(file: File, signal?: AbortSignal): Promise<BackendDocumentRecord>
  deleteDocument(documentId: string, signal?: AbortSignal): Promise<void>
  createPassage(
    documentId: string,
    passage: PassageMutation & { label: string },
    options?: PassageRequestOptions,
  ): Promise<BackendPassageRecord>
  updatePassage(
    passageId: string,
    passage: PassageMutation,
    options?: Pick<PassageRequestOptions, 'signal'>,
  ): Promise<BackendPassageRecord>
}

export function createBackendApi(fetchImplementation?: typeof fetch): BackendApi {
  const fetchRequest: typeof fetch = (...arguments_) => (
    (fetchImplementation ?? globalThis.fetch)(...arguments_)
  )
  return {
    getReadiness(signal) {
      return requestJson<BackendReadiness>(fetchRequest, '/v1/readiness', { signal })
    },

    uploadDocument(file, signal) {
      const body = new FormData()
      body.append('file', file, file.name)
      return requestJson<BackendDocumentRecord>(fetchRequest, '/v1/documents', {
        method: 'POST',
        body,
        signal,
      })
    },

    async deleteDocument(documentId, signal) {
      await requestJson<void>(
        fetchRequest,
        `/v1/documents/${encodeURIComponent(documentId)}`,
        { method: 'DELETE', signal },
      )
    },

    createPassage(documentId, passage, options = {}) {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (options.idempotencyKey) headers['Idempotency-Key'] = options.idempotencyKey
      return requestJson<BackendPassageRecord>(
        fetchRequest,
        `/v1/documents/${encodeURIComponent(documentId)}/passages`,
        {
          method: 'POST',
          headers,
          body: JSON.stringify({
            label: passage.label,
            page_number: passage.page,
            rects: passage.rects,
          }),
          signal: options.signal,
        },
      )
    },

    updatePassage(passageId, passage, options = {}) {
      return requestJson<BackendPassageRecord>(
        fetchRequest,
        `/v1/passages/${encodeURIComponent(passageId)}`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...(passage.label ? { label: passage.label } : {}),
            page_number: passage.page,
            rects: passage.rects,
          }),
          signal: options.signal,
        },
      )
    },
  }
}

async function requestJson<T>(
  fetchImplementation: typeof fetch,
  input: string,
  init: RequestInit,
): Promise<T> {
  let response: Response
  try {
    response = await fetchImplementation(input, init)
  } catch (error) {
    if (isAbortError(error)) throw error
    throw new BackendApiError('The local backend is unavailable.', {
      code: 'backend_unavailable',
      cause: error,
    })
  }

  if (!response.ok) {
    const detail = await readErrorDetail(response)
    throw new BackendApiError(detail.message, {
      status: response.status,
      code: detail.code,
    })
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

function isAbortError(error: unknown): boolean {
  return typeof error === 'object'
    && error !== null
    && 'name' in error
    && error.name === 'AbortError'
}

async function readErrorDetail(response: Response): Promise<{ code: string; message: string }> {
  try {
    const body = await response.json() as {
      detail?: string | { code?: string; message?: string }
    }
    if (typeof body.detail === 'string') {
      return { code: 'backend_error', message: body.detail }
    }
    return {
      code: body.detail?.code ?? 'backend_error',
      message: body.detail?.message ?? `Backend request failed (${response.status})`,
    }
  } catch {
    return {
      code: 'backend_error',
      message: `Backend request failed (${response.status})`,
    }
  }
}

export const backendApi = createBackendApi()
