import { fireEvent, render, screen } from '@testing-library/react'
import { SAMPLE_PASSAGES, SAMPLE_PREDICTIONS } from '../../data/sampleDocument'
import { PredictionSidebar } from './PredictionSidebar'

describe('PredictionSidebar', () => {
  it('shows the active passage once and keeps the remaining cards in document order', () => {
    render(
      <PredictionSidebar
        passages={SAMPLE_PASSAGES}
        predictions={SAMPLE_PREDICTIONS.slice(0, 3)}
        phase="partial"
        error={null}
        activePassageId="passage-2"
        onSelectPassage={vi.fn()}
        onDismissSelection={vi.fn()}
      />,
    )

    expect(screen.getByRole('heading', { name: 'Selected passage' })).toBeInTheDocument()
    expect(screen.getAllByText('Passage 2')).toHaveLength(1)
    const cards = screen.getAllByRole('button', { name: /Open Passage/ })
    expect(cards.map((card) => card.getAttribute('aria-label'))).toEqual([
      'Open Passage 1 on page 1',
      'Open Passage 3 on page 2',
    ])
    expect(screen.getByText('Predicting hidden passages…')).toBeInTheDocument()
  })

  it('navigates only when a card is activated, not when it is hovered', () => {
    const onSelectPassage = vi.fn()
    render(
      <PredictionSidebar
        passages={SAMPLE_PASSAGES}
        predictions={SAMPLE_PREDICTIONS.slice(0, 2)}
        phase="complete"
        error={null}
        activePassageId={null}
        onSelectPassage={onSelectPassage}
        onDismissSelection={vi.fn()}
      />,
    )

    const firstCard = screen.getByRole('button', { name: 'Open Passage 1 on page 1' })
    fireEvent.mouseEnter(firstCard)
    expect(onSelectPassage).not.toHaveBeenCalled()
    fireEvent.click(firstCard)
    expect(onSelectPassage).toHaveBeenCalledWith('passage-1')
  })

  it('offers a clear selection action from the dedicated selected area', () => {
    const onDismissSelection = vi.fn()
    render(
      <PredictionSidebar
        passages={SAMPLE_PASSAGES}
        predictions={SAMPLE_PREDICTIONS.slice(0, 1)}
        phase="complete"
        error={null}
        activePassageId="passage-1"
        onSelectPassage={vi.fn()}
        onDismissSelection={onDismissSelection}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Dismiss selected passage' }))
    expect(onDismissSelection).toHaveBeenCalledOnce()
  })

  it('labels provider output as a model prediction and never as simulated', () => {
    render(
      <PredictionSidebar
        passages={SAMPLE_PASSAGES}
        predictions={[{
          ...SAMPLE_PREDICTIONS[0],
          model: 'GPT-6 Astra',
          provenance: 'model',
          usage: {
            inputTokens: 2_000,
            outputTokens: 120,
            totalTokens: 2_120,
            reasoningTokens: 30,
            estimatedCostUsd: 0.026,
            pricingVersion: '2026-11-13',
          },
        }]}
        phase="complete"
        error={null}
        activePassageId="passage-1"
        onSelectPassage={vi.fn()}
        onDismissSelection={vi.fn()}
      />,
    )

    expect(screen.getByText('Model prediction')).toBeInTheDocument()
    expect(screen.getByText('2,120 tokens · $0.0260 estimated')).toBeInTheDocument()
    expect(screen.queryByText('Simulated prediction')).not.toBeInTheDocument()
  })
})
