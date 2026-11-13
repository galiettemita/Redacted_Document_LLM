import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { StrictMode } from 'react'
import { SAMPLE_DOCUMENT } from '../../data/sampleDocument'
import type {
  BackendApi,
  BackendPassageRecord,
  DocumentVisibleContextsSummary,
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
    getReadiness: vi.fn(),
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
    schedulePassageAnalysis: vi.fn().mockRejectedValue(
      new Error('Analysis should not start in this UI increment.'),
    ),
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

  it('is honest about uploaded files and offers the complete interactive sample', async () => {
    vi.useRealTimers()
    const onViewSample = vi.fn()
    render(<DocumentWorkspace document={uploadedDocument} backendDocument={readyBackendDocument} backendApi={apiWithCreate()} onReplace={vi.fn()} onViewSample={onViewSample} />)

    expect(await screen.findByRole('heading', { name: 'No redactions detected' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Prepare visible context' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'View interactive sample' }))
    expect(onViewSample).toHaveBeenCalledOnce()
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
          onViewSample={vi.fn()}
        />
      </StrictMode>,
    )

    expect(await screen.findByRole('heading', {
      name: '2 passages ready for retrieval',
    })).toBeInTheDocument()
    expect(detectDocumentPassages).toHaveBeenCalledWith(
      uploadedDocument.backendDocumentId,
      expect.any(AbortSignal),
    )
    expect(detectDocumentPassages).toHaveBeenCalledOnce()
    expect(container.querySelectorAll('[data-passage-id]')).toHaveLength(2)
    expect(screen.getByText(/every detected passage now has visible-only context/i)).toBeInTheDocument()
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
        onViewSample={vi.fn()}
      />,
    )

    expect(await screen.findByRole('heading', {
      name: '2 passages ready for retrieval',
    })).toBeInTheDocument()
    expect(prepareDocumentContexts).toHaveBeenCalledOnce()
    expect(prepareDocumentContexts).toHaveBeenCalledWith(
      uploadedDocument.backendDocumentId,
      expect.any(AbortSignal),
    )
    expect(screen.queryByRole('button', { name: 'Prepare visible context' })).not.toBeInTheDocument()
    expect(screen.getAllByText(
      /retrieval, worker, and approved model are not connected yet/i,
    )).not.toHaveLength(0)
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
        onViewSample={vi.fn()}
      />,
    )
    await screen.findByRole('heading', { name: '1 passage ready for retrieval' })

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
        onViewSample={vi.fn()}
      />,
    )
    await screen.findByRole('heading', { name: '2 passages ready for retrieval' })

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
        onViewSample={vi.fn()}
      />,
    )
    await screen.findByRole('heading', { name: '2 passages ready for retrieval' })

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
      name: '1 passage ready for retrieval',
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
        onViewSample={vi.fn()}
      />,
    )
    await screen.findByRole('heading', { name: '2 passages ready for retrieval' })

    fireEvent.click(screen.getByRole('button', { name: 'Remove selected passage' }))
    fireEvent.click(screen.getByRole('button', { name: 'Confirm removal' }))

    expect(await screen.findByText(
      'The local backend could not remove this passage.',
    )).toBeInTheDocument()
    expect(container.querySelectorAll('[data-passage-id]')).toHaveLength(2)
    expect(screen.getByRole('heading', { name: 'Visible-only context ready' })).toBeInTheDocument()
    expect(screen.getByRole('heading', {
      name: '2 passages ready for retrieval',
    })).toBeInTheDocument()
    expect(prepareDocumentContexts).toHaveBeenCalledOnce()
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
    expect(await screen.findByRole('heading', { name: '1 passage ready for retrieval' })).toBeInTheDocument()
    expect(screen.queryByText(/sample prediction/i)).not.toBeInTheDocument()
    expect(screen.queryByText('Simulated prediction')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View interactive sample' })).toBeInTheDocument()
  })

  it('shows the visible date after automatic document context preparation', async () => {
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
        onViewSample={vi.fn()}
      />,
    )

    expect(screen.queryByLabelText('Document date shown on PDF')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Save visible date' })).not.toBeInTheDocument()

    expect(await screen.findByRole('heading', {
      name: 'Visible-only context ready',
    })).toBeInTheDocument()
    expect(prepareDocumentContexts).toHaveBeenCalledWith(
      uploadedDocument.backendDocumentId,
      expect.any(AbortSignal),
    )
    expect(screen.getByText('part of a sentence')).toBeInTheDocument()
    expect(screen.getByText('1945-05-08')).toBeInTheDocument()
    expect(screen.getByText('8 estimated words')).toBeInTheDocument()
    expect(screen.getByText('37 visible words')).toBeInTheDocument()
    expect(screen.getByText('2 confirmed masks')).toBeInTheDocument()
    expect(screen.getByText('4 visible query terms')).toBeInTheDocument()
    expect(screen.getByText('Context query ready')).toBeInTheDocument()
    expect(screen.getAllByText(/approved model are not connected yet/i)).not.toHaveLength(0)
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
        onViewSample={vi.fn()}
      />,
    )

    expect(await screen.findByRole('heading', { name: 'No redactions detected' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Prepare visible context' })).not.toBeInTheDocument()
  })

  it('does not start retrieval or a model call without the real service connection', async () => {
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
        onViewSample={vi.fn()}
      />,
    )

    expect(await screen.findByRole('heading', {
      name: '2 passages ready for retrieval',
    })).toBeInTheDocument()
    expect(schedulePassageAnalysis).not.toHaveBeenCalled()
    expect(screen.queryByRole('button', { name: 'Predict passage' })).not.toBeInTheDocument()
    expect(container.querySelector('.pdf-page-shell')).toHaveTextContent('')
    expect(screen.queryByText('Predicted hidden passage')).not.toBeInTheDocument()
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
        onViewSample={vi.fn()}
      />,
    )

    expect(await screen.findByText(
      'The backend returned contexts for the wrong passages.',
    )).toBeInTheDocument()
    expect(screen.queryByText('Context query ready')).not.toBeInTheDocument()
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
