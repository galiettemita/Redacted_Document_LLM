import { act, fireEvent, render, screen } from '@testing-library/react'
import { FictionalDemoWorkspace } from './FictionalDemoWorkspace'

vi.mock('../viewer/PdfViewer', () => ({
  PdfViewer: ({
    page,
    passages,
    pinnedId,
    onHoverPassage,
    onPinPassage,
  }: {
    page: number
    passages: Array<{ id: string; label: string; page: number }>
    pinnedId?: string | null
    onHoverPassage?: (passageId: string | null) => void
    onPinPassage?: (passageId: string | null) => void
  }) => (
    <section aria-label="Fictional PDF viewer">
      <p>Demo page {page}</p>
      <p>Pinned {pinnedId ?? 'none'}</p>
      {passages.filter((passage) => passage.page === page).map((passage) => (
        <button
          key={passage.id}
          type="button"
          onMouseEnter={() => onHoverPassage?.(passage.id)}
          onMouseLeave={() => onHoverPassage?.(null)}
          onClick={() => onPinPassage?.(passage.id)}
        >
          {passage.label} redaction
        </button>
      ))}
    </section>
  ),
}))

describe('FictionalDemoWorkspace', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('progressively presents clearly simulated predictions linked to the fictional PDF', async () => {
    render(
      <FictionalDemoWorkspace
        onClose={vi.fn()}
        returnLabel="Return to upload"
      />,
    )

    expect(screen.getByRole('dialog', { name: 'Fictional interactive demo' })).toBeInTheDocument()
    expect(screen.getByText('This demonstration uses invented content and simulated predictions.')).toBeInTheDocument()
    expect(screen.getByText('Predicting hidden passages…')).toBeInTheDocument()

    await act(async () => vi.advanceTimersByTimeAsync(650))

    expect(screen.getByRole('button', { name: 'Open Passage 1 on page 1' })).toBeInTheDocument()
    expect(screen.getByText('Simulated prediction')).toBeInTheDocument()
    expect(screen.queryByText('Model prediction')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Passage 1 redaction' }))
    expect(screen.getByText('Pinned passage-1')).toBeInTheDocument()
    expect(screen.getAllByText('Passage 1')).toHaveLength(1)
  })

  it('keeps document navigation interactive and closes by control or Escape', async () => {
    const onClose = vi.fn()
    render(
      <FictionalDemoWorkspace
        onClose={onClose}
        returnLabel="Return to your document"
      />,
    )

    await act(async () => vi.advanceTimersByTimeAsync(1300))
    fireEvent.click(screen.getByRole('button', { name: 'Predictions' }))
    expect(screen.getByRole('button', { name: 'Predictions' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open Passage 2 on page 1' }))
    expect(screen.getByRole('button', { name: 'Document' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(screen.getByText('Pinned passage-2')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Return to your document' }))
    expect(onClose).toHaveBeenCalledOnce()

    onClose.mockClear()
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledOnce()
  })
})
