import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { StrictMode } from 'react'
import type {
  AnalysisJobRecord,
  BackendApi,
  BackendReadiness,
  BackendPassageRecord,
  DocumentVisibleContextsSummary,
  PassageDetectionResult,
  VisibleContextSummary,
} from '../../services/backendApi'
import { BackendApiError } from '../../services/backendApi'
import type { PredictionService } from '../../services/predictionService'
import type { BackendDocumentState } from '../../hooks/useBackendDocument'
import type { DocumentDescriptor } from '../../types'
import { DocumentWorkspace } from './DocumentWorkspace'

vi.mock('../viewer/PdfViewer', () => ({
  PdfViewer: ({ markingEnabled, onMarkPassage, onPinPassage, passages = [] }: { markingEnabled?: boolean; onMarkPassage?: (page: number, rect: { x: number; y: number; width: number; height: number }) => void; onPinPassage?: (passageId: string) => void; passages?: Array<{ id: string; label: string }> }) => (
    <div className="pdf-page-shell" aria-label="Local PDF renderer">
      {passages.map((passage) => (
        <button key={passage.id} type="button" data-passage-id={passage.id} aria-label={`${passage.label} redaction`} onClick={() => onPinPassage?.(passage.id)} />
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

const preparedDocumentContexts: DocumentVisibleContextsSummary = {
  document_id: uploadedDocument.backendDocumentId!,
  status: 'contexts_ready',
  passage_count: 2,
  contexts: [
    preparedContext,
    {
      ...preparedContext,
      passage_id: detectedSecondPassage.id,
    },
  ],
}

const stagedReadiness: BackendReadiness = {
  http: { ready: true, detail: 'ready' },
  database: { ready: true, detail: 'ready' },
  worker: { ready: true, detail: 'heartbeat current' },
  corpus: { ready: false, detail: 'not configured' },
  model: { ready: true, detail: 'configured and paid calls approved' },
  prediction_ready: true,
  prediction_bases: {
    model_only_v1: {
      operational: true,
      approved: false,
      ready: false,
      detail: 'operational; spending not approved',
    },
    best_available_foiarchive_bm25_v1: {
      operational: true,
      approved: true,
      ready: true,
      detail: 'operational and approved',
    },
    best_available_foiarchive_v1: {
      operational: false,
      approved: false,
      ready: false,
      detail: 'corpus: dense artifacts not available',
    },
    paper_exact_v1: {
      operational: false,
      approved: false,
      ready: false,
      detail: 'corpus: not configured',
    },
  },
  strongest_ready_basis: 'best_available_foiarchive_bm25_v1',
}

const fullReadiness: BackendReadiness = {
  ...stagedReadiness,
  prediction_bases: {
    ...stagedReadiness.prediction_bases,
    best_available_foiarchive_v1: {
      operational: true,
      approved: true,
      ready: true,
      detail: 'operational and approved',
    },
  },
  strongest_ready_basis: 'best_available_foiarchive_v1',
}

function analysisJob(
  passageId: string,
  status: AnalysisJobRecord['status'] = 'queued',
): AnalysisJobRecord {
  return {
    id: `job-${passageId}`,
    passage_id: passageId,
    document_id: uploadedDocument.backendDocumentId!,
    passage_revision: 1,
    prediction_basis: 'model_only_v1',
    status,
    stage: status === 'blocked' ? 'readiness' : 'prediction',
    blockers: status === 'blocked'
      ? [{ component: 'corpus', detail: 'complete corpus indexes are not available' }]
      : [],
    error_code: null,
    error_detail: null,
    created_at: '2026-11-13T12:00:00Z',
    updated_at: '2026-11-13T12:00:00Z',
  }
}

function contextsFor(
  ...passages: BackendPassageRecord[]
): DocumentVisibleContextsSummary {
  return {
    ...preparedDocumentContexts,
    passage_count: passages.length,
    contexts: passages.map((passage) => ({
      ...preparedContext,
      passage_id: passage.id,
      passage_revision: passage.revision,
    })),
  }
}

function apiWithCreate(
  createPassage = vi.fn().mockResolvedValue(savedPassage),
  overrides: Partial<BackendApi> = {},
): BackendApi {
  return {
    getReadiness: vi.fn().mockResolvedValue(stagedReadiness),
    uploadDocument: vi.fn(),
    deleteDocument: vi.fn(),
    createPassage,
    updatePassage: vi.fn().mockResolvedValue({ ...savedPassage, revision: 2 }),
    deletePassage: vi.fn().mockResolvedValue({
      ...savedPassage,
      status: 'deleted',
      revision: 2,
    }),
    detectDocumentPassages: vi.fn().mockResolvedValue(emptyDetection),
    prepareVisibleContext: vi.fn().mockResolvedValue(preparedContext),
    prepareDocumentContexts: vi.fn().mockResolvedValue(preparedDocumentContexts),
    getDocumentAnalysis: vi.fn().mockResolvedValue({
      document_id: uploadedDocument.backendDocumentId!,
      jobs: [],
    }),
    scheduleDocumentAnalysis: vi.fn().mockRejectedValue(
      new Error('Document analysis should not start in this UI increment.'),
    ),
    schedulePassageAnalysis: vi.fn().mockRejectedValue(
      new Error('Analysis should not start in this UI increment.'),
    ),
    ...overrides,
  }
}

describe('DocumentWorkspace', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('keeps the fictional demo available without changing the real document', async () => {
    vi.useRealTimers()
    const onViewDemo = vi.fn()
    render(<DocumentWorkspace document={uploadedDocument} backendDocument={readyBackendDocument} backendApi={apiWithCreate()} onReplace={vi.fn()} onViewDemo={onViewDemo} />)

    expect(await screen.findByRole('heading', { name: 'No redactions detected' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Prepare visible context' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'View fictional demo' }))
    expect(onViewDemo).toHaveBeenCalledOnce()
  })

  it('does not clear the real selection when Escape closes an obscuring demo', async () => {
    vi.useRealTimers()
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={apiWithCreate(undefined, {
          detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
        })}
        isInteractive={false}
        onReplace={vi.fn()}
      />,
    )
    await screen.findByRole('heading', { name: 'Ready to predict' })
    expect(screen.getByRole('button', { name: 'Dismiss selected passage' })).toBeInTheDocument()

    fireEvent.keyDown(window, { key: 'Escape' })

    expect(screen.getByRole('button', { name: 'Dismiss selected passage' })).toBeInTheDocument()
  })

  it('automatically detects and displays every passage without manual highlighting', async () => {
    vi.useRealTimers()
    const detectDocumentPassages = vi.fn().mockResolvedValue(detectedPassages)
    const api = apiWithCreate(undefined, { detectDocumentPassages })
    const { container } = render(
      <StrictMode>
        <DocumentWorkspace
          document={uploadedDocument}
          backendDocument={readyBackendDocument}
          backendApi={api}
          onReplace={vi.fn()}
        />
      </StrictMode>,
    )

    expect(await screen.findByRole('heading', {
      name: 'Ready to predict',
    })).toBeInTheDocument()
    expect(detectDocumentPassages).toHaveBeenCalledWith(
      uploadedDocument.backendDocumentId,
      expect.any(AbortSignal),
    )
    expect(detectDocumentPassages).toHaveBeenCalledOnce()
    expect(container.querySelectorAll('[data-passage-id]')).toHaveLength(2)
    expect(screen.getByText('2 hidden passages are ready.')).toBeInTheDocument()
    expect(screen.queryByText(/visible-only context/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/search query/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/four-list retrieval/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/GPT-6 Astra/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/readiness gate/i)).not.toBeInTheDocument()
    expect(screen.queryByText('Predicted hidden passage')).not.toBeInTheDocument()
  })

  it('automatically prepares visible-only context for every detected passage', async () => {
    vi.useRealTimers()
    const prepareDocumentContexts = vi.fn().mockResolvedValue(preparedDocumentContexts)
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      prepareDocumentContexts,
    })
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )

    expect(await screen.findByRole('heading', {
      name: 'Ready to predict',
    })).toBeInTheDocument()
    expect(prepareDocumentContexts).toHaveBeenCalledOnce()
    expect(prepareDocumentContexts).toHaveBeenCalledWith(
      uploadedDocument.backendDocumentId,
      expect.any(AbortSignal),
    )
    expect(screen.queryByRole('button', { name: 'Prepare visible context' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Predict passage' })).toBeInTheDocument()
  })

  it('adds a missed passage without replacing an automatic detection', async () => {
    vi.useRealTimers()
    const addedPassage: BackendPassageRecord = {
      ...savedPassage,
      id: '55555555-5555-4555-8555-555555555555',
      label: 'Passage 3',
    }
    const createPassage = vi.fn().mockResolvedValue(addedPassage)
    const updatePassage = vi.fn()
    const api = apiWithCreate(createPassage, {
      detectDocumentPassages: vi.fn().mockResolvedValue({
        ...detectedPassages,
        passages: [detectedSecondPassage],
      }),
      prepareDocumentContexts: vi.fn().mockResolvedValue(
        contextsFor(detectedSecondPassage),
      ),
      updatePassage,
    })
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )
    await screen.findByRole('heading', { name: 'Ready to predict' })

    fireEvent.click(screen.getByRole('button', { name: 'Add missed passage' }))
    fireEvent.click(screen.getByRole('button', { name: 'Complete marked rectangle' }))

    await waitFor(() => expect(createPassage).toHaveBeenCalledWith(
      uploadedDocument.backendDocumentId,
      expect.objectContaining({ label: 'Passage 3' }),
      expect.objectContaining({ idempotencyKey: expect.any(String) }),
    ))
    expect(updatePassage).not.toHaveBeenCalled()
  })

  it('corrects the selected automatic passage without dropping the others', async () => {
    vi.useRealTimers()
    const updatePassage = vi.fn().mockResolvedValue({
      ...savedPassage,
      source: 'manual',
      revision: 2,
    })
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      updatePassage,
    })
    const { container } = render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )
    await screen.findByRole('heading', { name: 'Ready to predict' })

    fireEvent.click(screen.getByRole('button', { name: 'Correct selected area' }))
    fireEvent.click(screen.getByRole('button', { name: 'Complete marked rectangle' }))

    await waitFor(() => expect(updatePassage).toHaveBeenCalledWith(
      savedPassage.id,
      expect.objectContaining({ page: 1 }),
      expect.any(Object),
    ))
    expect(container.querySelectorAll('[data-passage-id]')).toHaveLength(2)
  })

  it('confirms a false-positive removal and rebuilds only the remaining contexts', async () => {
    vi.useRealTimers()
    const deletedPassage: BackendPassageRecord = {
      ...savedPassage,
      status: 'deleted',
      revision: 2,
    }
    const deletePassage = vi.fn().mockResolvedValue(deletedPassage)
    const prepareDocumentContexts = vi.fn()
      .mockResolvedValueOnce(preparedDocumentContexts)
      .mockResolvedValueOnce(contextsFor(detectedSecondPassage))
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      prepareDocumentContexts,
      deletePassage,
    })
    const { container } = render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )
    await screen.findByRole('heading', { name: 'Ready to predict' })

    fireEvent.click(screen.getByRole('button', { name: 'Remove selected passage' }))
    expect(screen.getByRole('alertdialog', { name: 'Remove Passage 1?' })).toBeInTheDocument()
    expect(deletePassage).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Keep passage' })).toHaveFocus()

    fireEvent.click(screen.getByRole('button', { name: 'Keep passage' }))
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
    await waitFor(() => expect(
      screen.getByRole('button', { name: 'Remove selected passage' }),
    ).toHaveFocus())

    fireEvent.click(screen.getByRole('button', { name: 'Remove selected passage' }))
    fireEvent.click(screen.getByRole('button', { name: 'Confirm removal' }))

    await waitFor(() => expect(deletePassage).toHaveBeenCalledWith(
      savedPassage.id,
      expect.any(AbortSignal),
    ))
    expect(await screen.findByRole('heading', {
      name: 'Ready to predict',
    })).toBeInTheDocument()
    expect(container.querySelectorAll('[data-passage-id]')).toHaveLength(1)
    expect(container.querySelector('[data-passage-id]')).toHaveAttribute(
      'data-passage-id',
      detectedSecondPassage.id,
    )
    expect(prepareDocumentContexts).toHaveBeenCalledTimes(2)
  })

  it('keeps the passage and prepared context when removal is not confirmed by the backend', async () => {
    vi.useRealTimers()
    const deletePassage = vi.fn().mockRejectedValue(
      new Error('The local backend could not remove this passage.'),
    )
    const prepareDocumentContexts = vi.fn().mockResolvedValue(preparedDocumentContexts)
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      prepareDocumentContexts,
      deletePassage,
    })
    const { container } = render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )
    await screen.findByRole('heading', { name: 'Ready to predict' })

    fireEvent.click(screen.getByRole('button', { name: 'Remove selected passage' }))
    fireEvent.click(screen.getByRole('button', { name: 'Confirm removal' }))

    expect(await screen.findByText(
      'The local backend could not remove this passage.',
    )).toBeInTheDocument()
    expect(container.querySelectorAll('[data-passage-id]')).toHaveLength(2)
    expect(screen.getByRole('heading', { name: 'Ready to predict' })).toBeInTheDocument()
    expect(screen.queryByText(/visible-only context/i)).not.toBeInTheDocument()
    expect(prepareDocumentContexts).toHaveBeenCalledOnce()
  })

  it('persists one drawn rectangle without inventing prediction text', async () => {
    vi.useRealTimers()
    const createPassage = vi.fn().mockResolvedValue(savedPassage)
    render(<DocumentWorkspace document={uploadedDocument} backendDocument={readyBackendDocument} backendApi={apiWithCreate(createPassage)} onReplace={vi.fn()} />)

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
    expect(await screen.findByRole('heading', { name: 'Ready to predict' })).toBeInTheDocument()
    expect(screen.queryByText(/sample prediction/i)).not.toBeInTheDocument()
    expect(screen.queryByText('Simulated prediction')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'View interactive sample' })).not.toBeInTheDocument()
  })

  it('keeps preparation metadata out of the customer-facing prediction UI', async () => {
    vi.useRealTimers()
    const prepareDocumentContexts = vi.fn().mockResolvedValue(contextsFor(savedPassage))
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue({
        ...detectedPassages,
        passages: [{ ...savedPassage, source: 'detected' }],
      }),
      prepareDocumentContexts,
    })
    const { container } = render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )

    expect(screen.queryByLabelText('Document date shown on PDF')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Save visible date' })).not.toBeInTheDocument()

    expect(await screen.findByRole('heading', { name: 'Ready to predict' })).toBeInTheDocument()
    expect(prepareDocumentContexts).toHaveBeenCalledWith(
      uploadedDocument.backendDocumentId,
      expect.any(AbortSignal),
    )
    expect(screen.queryByText('part of a sentence')).not.toBeInTheDocument()
    expect(screen.queryByText('1945-05-08')).not.toBeInTheDocument()
    expect(screen.queryByText(/estimated words/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/visible words/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/confirmed masks/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/query terms/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/context query/i)).not.toBeInTheDocument()
    expect(screen.queryByText('visible before visible after')).not.toBeInTheDocument()
    expect(screen.queryByText(/search results/i)).not.toBeInTheDocument()
    expect(container.querySelector('.pdf-page-shell')).toHaveTextContent('')
  })

  it('never exposes a manual context-preparation control', async () => {
    vi.useRealTimers()
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={apiWithCreate()}
        onReplace={vi.fn()}
      />,
    )

    expect(await screen.findByRole('heading', { name: 'No redactions detected' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Prepare visible context' })).not.toBeInTheDocument()
  })

  it('does not start retrieval or a model call until the user requests it', async () => {
    vi.useRealTimers()
    const schedulePassageAnalysis = vi.fn()
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      schedulePassageAnalysis,
    })
    const { container } = render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )

    expect(await screen.findByRole('heading', {
      name: 'Ready to predict',
    })).toBeInTheDocument()
    expect(schedulePassageAnalysis).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Predict passage' })).toBeInTheDocument()
    expect(container.querySelector('.pdf-page-shell')).toHaveTextContent('')
    expect(screen.queryByText('Predicted hidden passage')).not.toBeInTheDocument()
  })

  it('keeps staged retrieval in selected-passage mode', async () => {
    vi.useRealTimers()
    const getReadiness = vi.fn().mockResolvedValue(stagedReadiness)
    const api = apiWithCreate(undefined, {
      getReadiness,
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
    })
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )

    expect(await screen.findByRole('button', { name: 'Predict passage' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Predict all passages' })).not.toBeInTheDocument()
    expect(getReadiness).toHaveBeenCalledWith(expect.any(AbortSignal))
  })

  it('offers all-passage mode only when a full dense basis is ready', async () => {
    vi.useRealTimers()
    const api = apiWithCreate(undefined, {
      getReadiness: vi.fn().mockResolvedValue(fullReadiness),
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
    })
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )

    expect(await screen.findByRole('button', {
      name: 'Predict all passages',
    })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Predict passage' })).not.toBeInTheDocument()
    expect(screen.queryByText(/dense artifacts/i)).not.toBeInTheDocument()
  })

  it('schedules only the passage the user selected', async () => {
    vi.useRealTimers()
    const accepted = {
      ...analysisJob(detectedSecondPassage.id),
      passage_revision: preparedContext.passage_revision,
    }
    const schedulePassageAnalysis = vi.fn().mockResolvedValue(accepted)
    const scheduleDocumentAnalysis = vi.fn().mockRejectedValue(
      new Error('The document-wide endpoint must not be used for a selected passage.'),
    )
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      scheduleDocumentAnalysis,
      schedulePassageAnalysis,
    })
    const stream = vi.fn<PredictionService['stream']>(
      async (_document, _signal, onPrediction) => {
        onPrediction({
          passageId: detectedSecondPassage.id,
          page: 2,
          model: 'GPT-6 Astra',
          text: 'Recovered archival sentence.',
          provenance: 'model',
          predictionBasis: 'model_only_v1',
        })
      },
    )
    const predictionService: PredictionService = { stream }
    const { container } = render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        predictionService={predictionService}
        onReplace={vi.fn()}
      />,
    )
    await screen.findByRole('heading', { name: 'Ready to predict' })

    fireEvent.click(screen.getByRole('button', { name: 'Passage 2 redaction' }))
    fireEvent.click(screen.getByRole('button', { name: 'Predict passage' }))

    await waitFor(() => expect(schedulePassageAnalysis).toHaveBeenCalledOnce())
    await waitFor(() => expect(stream).toHaveBeenCalledWith(
      uploadedDocument,
      expect.any(AbortSignal),
      expect.any(Function),
      {
        jobBases: {
          [`job-${detectedSecondPassage.id}`]: 'model_only_v1',
        },
      },
    ))
    expect(schedulePassageAnalysis).toHaveBeenCalledWith(
      detectedSecondPassage.id,
      preparedContext.passage_revision,
      expect.objectContaining({
        idempotencyKey: expect.any(String),
        signal: expect.any(AbortSignal),
      }),
    )
    expect(scheduleDocumentAnalysis).not.toHaveBeenCalled()
    expect(await screen.findByText('Recovered archival sentence.')).toBeInTheDocument()
    expect(screen.getByText(
      'Model-only prediction — no archive evidence used',
    )).toBeInTheDocument()
    expect(screen.queryByText('Simulated prediction')).not.toBeInTheDocument()
    expect(container.querySelector('.pdf-page-shell')).not.toHaveTextContent(
      'Recovered archival sentence.',
    )
  })

  it('shows a concise unavailable state without exposing infrastructure blockers', async () => {
    vi.useRealTimers()
    const schedulePassageAnalysis = vi.fn().mockResolvedValue({
      ...analysisJob(savedPassage.id, 'blocked'),
      passage_revision: preparedContext.passage_revision,
    })
    const predictionService: PredictionService = { stream: vi.fn() }
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      schedulePassageAnalysis,
    })
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        predictionService={predictionService}
        onReplace={vi.fn()}
      />,
    )
    await screen.findByRole('heading', { name: 'Ready to predict' })

    fireEvent.click(screen.getByRole('button', { name: 'Predict passage' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Predictions are not available yet. Please try again later.',
    )
    expect(screen.getByRole('heading', { name: 'Predictions unavailable' })).toBeInTheDocument()
    expect(screen.queryByText(/corpus indexes/i)).not.toBeInTheDocument()
    expect(predictionService.stream).not.toHaveBeenCalled()
  })

  it('reuses the selected-passage idempotency key after an uncertain response', async () => {
    vi.useRealTimers()
    const schedulePassageAnalysis = vi.fn()
      .mockRejectedValueOnce(new Error('The response was interrupted.'))
      .mockResolvedValueOnce({
        ...analysisJob(savedPassage.id),
        passage_revision: preparedContext.passage_revision,
      })
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      schedulePassageAnalysis,
    })
    const predictionService: PredictionService = {
      async stream() {},
    }
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        predictionService={predictionService}
        onReplace={vi.fn()}
      />,
    )
    await screen.findByRole('heading', { name: 'Ready to predict' })

    fireEvent.click(screen.getByRole('button', { name: 'Predict passage' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('The response was interrupted.')
    const firstKey = schedulePassageAnalysis.mock.calls[0][2].idempotencyKey

    fireEvent.click(screen.getByRole('button', { name: 'Predict passage' }))

    await waitFor(() => expect(schedulePassageAnalysis).toHaveBeenCalledTimes(2))
    expect(schedulePassageAnalysis.mock.calls[1][2].idempotencyKey).toBe(firstKey)
  })

  it('recovers a completed prediction and leaves another passage available', async () => {
    vi.useRealTimers()
    const recovered = {
      ...analysisJob(savedPassage.id),
      passage_revision: preparedContext.passage_revision,
      status: 'complete' as const,
      stage: 'complete' as const,
    }
    const schedulePassageAnalysis = vi.fn()
    const getDocumentAnalysis = vi.fn().mockResolvedValue({
      document_id: uploadedDocument.backendDocumentId!,
      jobs: [recovered],
    })
    const stream = vi.fn<PredictionService['stream']>(
      async (_document, _signal, onPrediction) => {
        onPrediction({
          passageId: savedPassage.id,
          page: 1,
          model: 'GPT-6 Astra',
          text: 'Stored real prediction.',
          provenance: 'model',
          predictionBasis: recovered.prediction_basis,
        })
      },
    )
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      getDocumentAnalysis,
      schedulePassageAnalysis,
    })

    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        predictionService={{ stream }}
        onReplace={vi.fn()}
      />,
    )

    expect(await screen.findByText('Stored real prediction.')).toBeInTheDocument()
    expect(getDocumentAnalysis).toHaveBeenCalledWith(
      uploadedDocument.backendDocumentId,
      expect.any(AbortSignal),
    )
    fireEvent.click(screen.getByRole('button', { name: 'Passage 2 redaction' }))
    expect(screen.getByRole('button', { name: 'Predict passage' })).toBeEnabled()
    expect(schedulePassageAnalysis).not.toHaveBeenCalled()
  })

  it('suppresses duplicate prediction clicks while passage scheduling is pending', async () => {
    vi.useRealTimers()
    const schedulePassageAnalysis = vi.fn<BackendApi['schedulePassageAnalysis']>(
      () => new Promise(() => {}),
    )
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      schedulePassageAnalysis,
    })
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )
    const button = await screen.findByRole('button', { name: 'Predict passage' })

    fireEvent.click(button)
    fireEvent.click(button)

    expect(schedulePassageAnalysis).toHaveBeenCalledOnce()
    expect(button).toBeDisabled()
  })

  it('locks passage changes and replacement while an accepted prediction is active', async () => {
    vi.useRealTimers()
    const accepted = {
      ...analysisJob(savedPassage.id),
      passage_revision: preparedContext.passage_revision,
    }
    const schedulePassageAnalysis = vi.fn().mockResolvedValue(accepted)
    const stream = vi.fn<PredictionService['stream']>(
      async (_document, signal) => new Promise<void>((resolve) => {
        signal.addEventListener('abort', () => resolve(), { once: true })
      }),
    )
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      schedulePassageAnalysis,
    })
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        predictionService={{ stream }}
        onReplace={vi.fn()}
      />,
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Predict passage' }))
    await waitFor(() => expect(stream).toHaveBeenCalledOnce())

    expect(screen.getByRole('button', { name: 'Correct selected area' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Remove selected passage' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Add missed passage' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Replace document' })).toBeDisabled()
  })

  it('fails closed when automatic preparation returns the wrong passage set', async () => {
    vi.useRealTimers()
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      prepareDocumentContexts: vi.fn().mockResolvedValue(contextsFor(savedPassage)),
    })
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )

    expect(await screen.findByText(
      'We couldn’t prepare this document for prediction. Please try again.',
    )).toBeInTheDocument()
    expect(screen.queryByText('Context query ready')).not.toBeInTheDocument()
  })

  it('shows the backend preparation reason instead of a generic error', async () => {
    vi.useRealTimers()
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      prepareDocumentContexts: vi.fn().mockRejectedValue(new BackendApiError(
        'No supported date was found in visible first-page header text',
        { status: 422, code: 'visible_date_not_found' },
      )),
    })
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )

    expect(await screen.findByText(
      'No supported date was found in visible first-page header text',
    )).toBeInTheDocument()
    expect(screen.queryByText(
      'We couldn’t prepare this document for prediction. Please try again.',
    )).not.toBeInTheDocument()
  })

  it('retries preparation for the already-open document', async () => {
    vi.useRealTimers()
    const prepareDocumentContexts = vi.fn()
      .mockRejectedValueOnce(new BackendApiError(
        'No supported date was found in visible first-page header text',
        { status: 422, code: 'visible_date_not_found' },
      ))
      .mockResolvedValueOnce(preparedDocumentContexts)
    const api = apiWithCreate(undefined, {
      detectDocumentPassages: vi.fn().mockResolvedValue(detectedPassages),
      prepareDocumentContexts,
    })
    render(
      <DocumentWorkspace
        document={uploadedDocument}
        backendDocument={readyBackendDocument}
        backendApi={api}
        onReplace={vi.fn()}
      />,
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Try again' }))

    expect(await screen.findByRole('heading', { name: 'Ready to predict' }))
      .toBeInTheDocument()
    expect(prepareDocumentContexts).toHaveBeenCalledTimes(2)
  })

  it('does not allow a passage mutation before the backend upload is ready', () => {
    const createPassage = vi.fn()
    const uploading: BackendDocumentState = {
      status: 'uploading',
      record: null,
      error: null,
      remove: vi.fn(),
    }
    render(<DocumentWorkspace document={{ ...uploadedDocument, backendDocumentId: undefined }} backendDocument={uploading} backendApi={apiWithCreate(createPassage)} onReplace={vi.fn()} />)

    expect(screen.getAllByText(/Uploading to the loopback-only local backend/)).not.toHaveLength(0)
    expect(screen.getByRole('button', { name: 'Mark a passage' })).toBeDisabled()
    expect(createPassage).not.toHaveBeenCalled()
  })

  it('locks destructive navigation while the stored document is being removed', () => {
    const deleting: BackendDocumentState = {
      ...readyBackendDocument,
      status: 'deleting',
    }
    render(<DocumentWorkspace document={uploadedDocument} backendDocument={deleting} backendApi={apiWithCreate()} onReplace={vi.fn()} />)

    expect(screen.getByRole('button', { name: 'Removing document…' })).toBeDisabled()
    expect(screen.queryByRole('button', { name: 'View interactive sample' })).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Removing document…' })).toBeInTheDocument()
  })
})
