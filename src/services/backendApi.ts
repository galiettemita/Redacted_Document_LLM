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
  date_provenance: 'visible_text' | 'pdf_metadata_corroborated' | null
  visible_date_evidence: VisibleDateEvidence | null
  created_at: string
  expires_at: string
}

export interface VisibleDateEvidence {
  value: string
  page_number: number
  line_text: string
  words: Array<{
    text: string
    page_number: number
    x: number
    y: number
    width: number
    height: number
    block_id: number
    line_id: number
    source: 'embedded' | 'ocr'
    confidence: number
  }>
  parser_version: 'visible-date-v1'
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

export interface PassageDetectionResult {
  document_id: string
  status: 'detection_ready'
  pages_scanned: number
  coverage_confident: boolean
  uncertain_pages: number[]
  warnings: string[]
  passages: BackendPassageRecord[]
}

export interface VisibleContextSummary {
  passage_id: string
  passage_revision: number
  status: 'context_ready'
  document_date: string
  length_cue: 'a few words' | 'part of a sentence' | 'about a sentence' | 'a paragraph'
  estimated_words: number
  visible_word_count: number
  confirmed_mask_count: number
  serialization_version: 'visible-transcript-v1'
  context_query_status: 'ready'
  context_query_term_count: number
  context_query_sha256: string
  context_query_version: 'context-query-v1'
  context_stopwords_version: string
}

export interface AnalysisBlocker {
  component: string
  detail: string
}

export interface AnalysisJobRecord {
  id: string
  passage_id: string
  document_id: string
  passage_revision: number
  status: 'blocked' | 'queued' | 'running' | 'complete' | 'failed' | 'cancelled'
  stage: 'readiness' | 'query' | 'retrieval' | 'prediction' | 'complete'
  blockers: AnalysisBlocker[]
  error_code: string | null
  error_detail: string | null
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
  detectDocumentPassages(
    documentId: string,
    signal?: AbortSignal,
  ): Promise<PassageDetectionResult>
  prepareVisibleContext(
    passageId: string,
    signal?: AbortSignal,
  ): Promise<VisibleContextSummary>
  schedulePassageAnalysis(
    passageId: string,
    expectedRevision: number,
    options: Required<Pick<PassageRequestOptions, 'idempotencyKey'>>
      & Pick<PassageRequestOptions, 'signal'>,
  ): Promise<AnalysisJobRecord>
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

    detectDocumentPassages(documentId, signal) {
      return requestJson<PassageDetectionResult>(
        fetchRequest,
        `/v1/documents/${encodeURIComponent(documentId)}/passages:detect`,
        { method: 'POST', signal },
      )
    },

    prepareVisibleContext(passageId, signal) {
      return requestJson<VisibleContextSummary>(
        fetchRequest,
        `/v1/passages/${encodeURIComponent(passageId)}:prepare-context`,
        { method: 'POST', signal },
      )
    },

    schedulePassageAnalysis(passageId, expectedRevision, options) {
      return requestJson<AnalysisJobRecord>(
        fetchRequest,
        `/v1/passages/${encodeURIComponent(passageId)}/analysis`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Idempotency-Key': options.idempotencyKey,
          },
          body: JSON.stringify({ expected_revision: expectedRevision }),
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
