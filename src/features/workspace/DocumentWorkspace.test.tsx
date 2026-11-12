import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { SAMPLE_DOCUMENT } from '../../data/sampleDocument'
import type {
  BackendApi,
  BackendPassageRecord,
  VisibleContextSummary,
} from '../../services/backendApi'
import type { BackendDocumentState } from '../../hooks/useBackendDocument'
import type { DocumentDescriptor } from '../../types'
import { DocumentWorkspace } from './DocumentWorkspace'

vi.mock('../viewer/PdfViewer', () => ({
  PdfViewer: ({ markingEnabled, onMarkPassage }: { markingEnabled?: boolean; onMarkPassage?: (page: number, rect: { x: number; y: number; width: number; height: number }) => void }) => (
    <div className="pdf-page-shell" aria-label="Local PDF renderer">
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
    updateDocumentMetadata: vi.fn().mockResolvedValue({
      ...readyBackendDocument.record,
      visible_date: '1945-05-08',
      date_provenance: 'manual_visible_document',
    }),
    prepareVisibleContext: vi.fn().mockResolvedValue(preparedContext),
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
    expect(screen.getByText(/retrieval and model prediction are not connected yet/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'View interactive sample' }))
    expect(onViewSample).toHaveBeenCalledOnce()
  })

  it('persists one drawn rectangle without inventing prediction text', async () => {
    vi.useRealTimers()
    const createPassage = vi.fn().mockResolvedValue(savedPassage)
    render(<DocumentWorkspace document={uploadedDocument} backendDocument={readyBackendDocument} backendApi={apiWithCreate(createPassage)} onReplace={vi.fn()} onViewSample={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Mark a passage' }))
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
    expect(await screen.findByRole('heading', { name: 'Passage 1 is saved' })).toBeInTheDocument()
    expect(screen.queryByText(/sample prediction/i)).not.toBeInTheDocument()
    expect(screen.queryByText('Simulated prediction')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View interactive sample' })).toBeInTheDocument()
  })

  it('saves the visible date and shows the real passage-linked context checkpoint', async () => {
    vi.useRealTimers()
    const updateDocumentMetadata = vi.fn().mockResolvedValue({
      ...readyBackendDocument.record,
      visible_date: '1945-05-08',
      date_provenance: 'manual_visible_document',
    })
    const prepareVisibleContext = vi.fn().mockResolvedValue(preparedContext)
    const api = apiWithCreate(undefined, {
      updateDocumentMetadata,
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

    fireEvent.change(screen.getByLabelText('Document date shown on PDF'), {
      target: { value: '1945-05-08' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Save visible date' }))
    await waitFor(() => expect(updateDocumentMetadata).toHaveBeenCalledWith(
      uploadedDocument.backendDocumentId,
      '1945-05-08',
      expect.any(AbortSignal),
    ))

    fireEvent.click(screen.getByRole('button', { name: 'Mark a passage' }))
    fireEvent.click(screen.getByRole('button', { name: 'Complete marked rectangle' }))
    await screen.findByRole('heading', { name: 'Passage 1 is saved' })

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
    expect(screen.getByText('8 estimated words')).toBeInTheDocument()
    expect(screen.getByText('37 visible words')).toBeInTheDocument()
    expect(screen.getByText('2 confirmed masks')).toBeInTheDocument()
    expect(screen.getAllByText(/Retrieval and model prediction are not connected yet/)).not.toHaveLength(0)
    expect(container.querySelector('.pdf-page-shell')).toHaveTextContent('')
  })

  it('keeps context preparation disabled until both date and passage are saved', () => {
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
