import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { SAMPLE_DOCUMENT } from '../../data/sampleDocument'
import type {
  AnalysisJobRecord,
  BackendApi,
  BackendPassageRecord,
  PassageDetectionResult,
  VisibleContextSummary,
} from '../../services/backendApi'
import type { BackendDocumentState } from '../../hooks/useBackendDocument'
import type { DocumentDescriptor } from '../../types'
import { DocumentWorkspace } from './DocumentWorkspace'

vi.mock('../viewer/PdfViewer', () => ({
  PdfViewer: ({ markingEnabled, onMarkPassage, passages = [] }: { markingEnabled?: boolean; onMarkPassage?: (page: number, rect: { x: number; y: number; width: number; height: number }) => void; passages?: Array<{ id: string; label: string }> }) => (
    <div className="pdf-page-shell" aria-label="Local PDF renderer">
      {passages.map((passage) => (
        <button key={passage.id} type="button" data-passage-id={passage.id} aria-label={`${passage.label} redaction`} />
      ))}
      {markingEnabled ? (
        <button type="button" onClick={() => onMarkPassage?.(1, { x: 0.2, y: 0.3, width: 0.4, height: 0.08 })}>
          Complete marked rectangle
        </button>
      ) : null}
    </div>
  ),
}))

const uploadedDocument: DocumentDescriptor = {
  id: 'uploaded-record',
  filename: 'committee-record.pdf',
  source: 'blob:committee-record',
  mode: 'uploaded',
  backendDocumentId: '11111111-1111-4111-8111-111111111111',
}

const readyBackendDocument: BackendDocumentState = {
  status: 'ready',
  record: {
    id: uploadedDocument.backendDocumentId!,
    filename: uploadedDocument.filename,
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
  },
  error: null,
  remove: vi.fn(),
}

const savedPassage: BackendPassageRecord = {
  id: '22222222-2222-4222-8222-222222222222',
  document_id: uploadedDocument.backendDocumentId!,
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

const preparedContext: VisibleContextSummary = {
  passage_id: savedPassage.id,
  passage_revision: 2,
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
  passage_id: savedPassage.id,
  document_id: uploadedDocument.backendDocumentId!,
  passage_revision: preparedContext.passage_revision,
  status: 'blocked',
  stage: 'readiness',
  blockers: [
    { component: 'corpus', detail: 'corpus exclusion coverage is incomplete' },
    { component: 'model', detail: 'model access is not configured' },
    { component: 'worker', detail: 'analysis worker execution is not connected' },
  ],
  error_code: null,
  error_detail: null,
  created_at: '2026-11-13T12:00:00Z',
  updated_at: '2026-11-13T12:00:00Z',
}

const emptyDetection: PassageDetectionResult = {
  document_id: uploadedDocument.backendDocumentId!,
  status: 'detection_ready',
  pages_scanned: 2,
  coverage_confident: true,
  uncertain_pages: [],
  warnings: [],
  passages: [],
}

const detectedSecondPassage: BackendPassageRecord = {
  ...savedPassage,
  id: '44444444-4444-4444-8444-444444444444',
  label: 'Passage 2',
  page_number: 2,
  rects: [{ x: 0.1, y: 0.4, width: 0.2, height: 0.03 }],
  source: 'detected',
}

const detectedPassages: PassageDetectionResult = {
  ...emptyDetection,
  passages: [
    { ...savedPassage, source: 'detected' },
    detectedSecondPassage,
  ],
}

function apiWithCreate(
  createPassage = vi.fn().mockResolvedValue(savedPassage),
  overrides: Partial<BackendApi> = {},
): BackendApi {
  return {
    getReadiness: vi.fn(),
    uploadDocument: vi.fn(),
    deleteDocument: vi.fn(),
    createPassage,
    updatePassage: vi.fn().mockResolvedValue({ ...savedPassage, revision: 2 }),
    detectDocumentPassages: vi.fn().mockResolvedValue(emptyDetection),
    prepareVisibleContext: vi.fn().mockResolvedValue(preparedContext),
    schedulePassageAnalysis: vi.fn().mockResolvedValue(blockedAnalysis),
    ...overrides,
  }
}

describe('DocumentWorkspace', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('starts sample predictions automatically and reveals cards progressively', async () => {
    render(<DocumentWorkspace document={SAMPLE_DOCUMENT} onReplace={vi.fn()} onViewSample={vi.fn()} />)
    expect(screen.getByText('Fictional demonstration document')).toBeInTheDocument()
    expect(screen.getByText('Predicting hidden passages…')).toBeInTheDocument()

    await act(async () => vi.advanceTimersByTimeAsync(700))
    expect(screen.getByRole('button', { name: 'Open Passage 1 on page 1' })).toBeInTheDocument()
    expect(screen.getByText('Predicting hidden passages…')).toBeInTheDocument()
  })

  it('is honest about uploaded files and offers the complete interactive sample', () => {
    const onViewSample = vi.fn()
    render(<DocumentWorkspace document={uploadedDocument} backendDocument={readyBackendDocument} backendApi={apiWithCreate()} onReplace={vi.fn()} onViewSample={onViewSample} />)

    expect(screen.getByText(/The PDF is stored temporarily by the local backend/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Prepare visible context' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'View interactive sample' }))
    expect(onViewSample).toHaveBeenCalledOnce()
  })

  it('automatically detects and displays every passage without manual highlighting', async () => {
    vi.useRealTimers()
    const detectDocumentPassages = vi.fn().mockResolvedValue(detectedPassages)
    const api = apiWithCreate(undefined, { detectDocumentPassages })
    const { container } = render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
        onViewSample={vi.fn()}
      />,
    )

    expect(await screen.findByRole('heading', {
      name: '2 passages detected',
    })).toBeInTheDocument()
    expect(detectDocumentPassages).toHaveBeenCalledWith(
      uploadedDocument.backendDocumentId,
      expect.any(AbortSignal),
    )
    expect(container.querySelectorAll('[data-passage-id]')).toHaveLength(2)
    expect(screen.getByText(/all 2 pages were scanned automatically/i)).toBeInTheDocument()
    expect(screen.queryByText('Predicted hidden passage')).not.toBeInTheDocument()
  })

  it('persists one drawn rectangle without inventing prediction text', async () => {
    vi.useRealTimers()
    const createPassage = vi.fn().mockResolvedValue(savedPassage)
    render(<DocumentWorkspace document={uploadedDocument} backendDocument={readyBackendDocument} backendApi={apiWithCreate(createPassage)} onReplace={vi.fn()} onViewSample={vi.fn()} />)

    const markButton = screen.getByRole('button', { name: 'Mark a passage' })
    await waitFor(() => expect(markButton).toBeEnabled())
    fireEvent.click(markButton)
    fireEvent.click(screen.getByRole('button', { name: 'Complete marked rectangle' }))

    await waitFor(() => expect(createPassage).toHaveBeenCalledWith(
      uploadedDocument.backendDocumentId,
      {
        label: 'Passage 1',
        page: 1,
        rects: [{ x: 0.2, y: 0.3, width: 0.4, height: 0.08 }],
      },
      expect.objectContaining({ idempotencyKey: expect.any(String) }),
    ))
    expect(await screen.findByRole('heading', { name: 'Selected passage is saved' })).toBeInTheDocument()
    expect(screen.queryByText(/sample prediction/i)).not.toBeInTheDocument()
    expect(screen.queryByText('Simulated prediction')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View interactive sample' })).toBeInTheDocument()
  })

  it('detects the visible date while preparing the real passage-linked context', async () => {
    vi.useRealTimers()
    const prepareVisibleContext = vi.fn().mockResolvedValue(preparedContext)
    const api = apiWithCreate(undefined, {
      prepareVisibleContext,
    })
    const { container } = render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
        onViewSample={vi.fn()}
      />,
    )

    expect(screen.queryByLabelText('Document date shown on PDF')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Save visible date' })).not.toBeInTheDocument()

    const markButton = screen.getByRole('button', { name: 'Mark a passage' })
    await waitFor(() => expect(markButton).toBeEnabled())
    fireEvent.click(markButton)
    fireEvent.click(screen.getByRole('button', { name: 'Complete marked rectangle' }))
    await screen.findByRole('heading', { name: 'Selected passage is saved' })

    const prepareButton = screen.getByRole('button', { name: 'Prepare visible context' })
    await waitFor(() => expect(prepareButton).toBeEnabled())
    fireEvent.click(prepareButton)

    expect(await screen.findByRole('heading', {
      name: 'Visible-only context ready',
    })).toBeInTheDocument()
    expect(prepareVisibleContext).toHaveBeenCalledWith(
      savedPassage.id,
      expect.any(AbortSignal),
    )
    expect(screen.getByText('part of a sentence')).toBeInTheDocument()
    expect(screen.getByText('1945-05-08')).toBeInTheDocument()
    expect(screen.getByText('8 estimated words')).toBeInTheDocument()
    expect(screen.getByText('37 visible words')).toBeInTheDocument()
    expect(screen.getByText('2 confirmed masks')).toBeInTheDocument()
    expect(screen.getByText('4 visible query terms')).toBeInTheDocument()
    expect(screen.getByText('Context query ready')).toBeInTheDocument()
    expect(screen.getByText(/The next action checks the real corpus, worker, and model gates/)).toBeInTheDocument()
    expect(screen.queryByText('visible before visible after')).not.toBeInTheDocument()
    expect(screen.queryByText(/search results/i)).not.toBeInTheDocument()
    expect(container.querySelector('.pdf-page-shell')).toHaveTextContent('')
  })

  it('keeps context preparation disabled until a real passage is saved', () => {
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={apiWithCreate()}
        onReplace={vi.fn()}
        onViewSample={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: 'Prepare visible context' })).toBeDisabled()
  })

  it('creates a durable analysis check and shows real blockers without PDF prediction text', async () => {
    vi.useRealTimers()
    const schedulePassageAnalysis = vi.fn().mockResolvedValue(blockedAnalysis)
    const api = apiWithCreate(undefined, { schedulePassageAnalysis })
    const { container } = render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
        onViewSample={vi.fn()}
      />,
    )

    const markButton = screen.getByRole('button', { name: 'Mark a passage' })
    await waitFor(() => expect(markButton).toBeEnabled())
    fireEvent.click(markButton)
    fireEvent.click(screen.getByRole('button', { name: 'Complete marked rectangle' }))
    await screen.findByRole('heading', { name: 'Selected passage is saved' })
    fireEvent.click(screen.getByRole('button', { name: 'Prepare visible context' }))
    await screen.findByRole('heading', { name: 'Visible-only context ready' })

    fireEvent.click(screen.getByRole('button', { name: 'Predict passage' }))

    expect(await screen.findByRole('heading', {
      name: 'Prediction is not ready',
    })).toBeInTheDocument()
    expect(schedulePassageAnalysis).toHaveBeenCalledWith(
      savedPassage.id,
      preparedContext.passage_revision,
      expect.objectContaining({ idempotencyKey: expect.any(String) }),
    )
    expect(screen.getByText(/corpus exclusion coverage is incomplete/)).toBeInTheDocument()
    expect(screen.getByText(/model access is not configured/)).toBeInTheDocument()
    expect(screen.getByText(/analysis worker execution is not connected/)).toBeInTheDocument()
    expect(container.querySelector('.pdf-page-shell')).toHaveTextContent('')
    expect(screen.queryByText('Predicted hidden passage')).not.toBeInTheDocument()
  })

  it('reuses the analysis idempotency key when an uncertain request is retried', async () => {
    vi.useRealTimers()
    const schedulePassageAnalysis = vi.fn()
      .mockRejectedValueOnce(new Error('The connection closed before a response arrived.'))
      .mockResolvedValueOnce(blockedAnalysis)
    const api = apiWithCreate(undefined, { schedulePassageAnalysis })
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
        onViewSample={vi.fn()}
      />,
    )

    const markButton = screen.getByRole('button', { name: 'Mark a passage' })
    await waitFor(() => expect(markButton).toBeEnabled())
    fireEvent.click(markButton)
    fireEvent.click(screen.getByRole('button', { name: 'Complete marked rectangle' }))
    await screen.findByRole('heading', { name: 'Selected passage is saved' })
    fireEvent.click(screen.getByRole('button', { name: 'Prepare visible context' }))
    await screen.findByRole('heading', { name: 'Visible-only context ready' })

    fireEvent.click(screen.getByRole('button', { name: 'Predict passage' }))
    expect(await screen.findByText('The connection closed before a response arrived.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Predict passage' }))
    await screen.findByRole('heading', { name: 'Prediction is not ready' })

    const firstOptions = schedulePassageAnalysis.mock.calls[0][2]
    const retryOptions = schedulePassageAnalysis.mock.calls[1][2]
    expect(retryOptions.idempotencyKey).toBe(firstOptions.idempotencyKey)
  })

  it('does not allow a passage mutation before the backend upload is ready', () => {
    const createPassage = vi.fn()
    const uploading: BackendDocumentState = {
      status: 'uploading',
      record: null,
      error: null,
      remove: vi.fn(),
    }
    render(<DocumentWorkspace document={{ ...uploadedDocument, backendDocumentId: undefined }} backendDocument={uploading} backendApi={apiWithCreate(createPassage)} onReplace={vi.fn()} onViewSample={vi.fn()} />)

    expect(screen.getAllByText(/Uploading to the loopback-only local backend/)).not.toHaveLength(0)
    expect(screen.getByRole('button', { name: 'Mark a passage' })).toBeDisabled()
    expect(createPassage).not.toHaveBeenCalled()
  })

  it('locks destructive navigation while the stored document is being removed', () => {
    const deleting: BackendDocumentState = {
      ...readyBackendDocument,
      status: 'deleting',
    }
    render(<DocumentWorkspace document={uploadedDocument} backendDocument={deleting} backendApi={apiWithCreate()} onReplace={vi.fn()} onViewSample={vi.fn()} />)

    expect(screen.getByRole('button', { name: 'Removing document…' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'View interactive sample' })).toBeDisabled()
    expect(screen.getByRole('heading', { name: 'Removing stored PDF' })).toBeInTheDocument()
  })

  it('returns to the document view when a prediction card is activated', async () => {
    render(<DocumentWorkspace document={SAMPLE_DOCUMENT} onReplace={vi.fn()} onViewSample={vi.fn()} />)
    await act(async () => vi.advanceTimersByTimeAsync(1400))

    fireEvent.click(screen.getByRole('button', { name: 'Predictions' }))
    expect(screen.getByRole('button', { name: 'Predictions' })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: 'Open Passage 2 on page 1' }))

    expect(screen.getByRole('button', { name: 'Document' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getAllByText('Passage 2')).toHaveLength(1)
  })
})
