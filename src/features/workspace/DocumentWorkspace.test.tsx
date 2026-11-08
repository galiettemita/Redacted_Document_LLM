import { act, fireEvent, render, screen } from '@testing-library/react'
import { SAMPLE_DOCUMENT } from '../../data/sampleDocument'
import type { DocumentDescriptor } from '../../types'
import { DocumentWorkspace } from './DocumentWorkspace'

vi.mock('../viewer/PdfViewer', () => ({
  PdfViewer: ({ markingEnabled, onMarkPassage }: { markingEnabled?: boolean; onMarkPassage?: (page: number, rect: { x: number; y: number; width: number; height: number }) => void }) => (
    <div aria-label="Local PDF renderer">
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
    render(<DocumentWorkspace document={uploadedDocument} onReplace={vi.fn()} onViewSample={onViewSample} />)

    expect(screen.getByText(/Automatic passage detection and predictions aren’t connected yet/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'View interactive sample' }))
    expect(onViewSample).toHaveBeenCalledOnce()
  })

  it('turns one drawn rectangle into a clearly labeled sample prediction', () => {
    render(<DocumentWorkspace document={uploadedDocument} onReplace={vi.fn()} onViewSample={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Mark a passage' }))
    fireEvent.click(screen.getByRole('button', { name: 'Complete marked rectangle' }))

    expect(screen.getByText(/This marked area uses a sample prediction/)).toBeInTheDocument()
    expect(screen.getByText('Simulated prediction')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View interactive sample' })).toBeInTheDocument()
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
