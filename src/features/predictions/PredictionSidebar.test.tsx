import { fireEvent, render, screen } from '@testing-library/react'
import { TEST_PASSAGES, TEST_PREDICTIONS } from '../../test/fixtures'
import type { PredictionBasis } from '../../types'
import { PredictionSidebar } from './PredictionSidebar'

describe('PredictionSidebar', () => {
  it('shows the active passage once and keeps the remaining cards in document order', () => {
    render(
      <PredictionSidebar
        passages={TEST_PASSAGES}
        predictions={TEST_PREDICTIONS.slice(0, 3)}
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
        passages={TEST_PASSAGES}
        predictions={TEST_PREDICTIONS.slice(0, 2)}
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
        passages={TEST_PASSAGES}
        predictions={TEST_PREDICTIONS.slice(0, 1)}
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

  it('labels provider output without exposing its model, token, or cost metadata', () => {
    render(
      <PredictionSidebar
        passages={TEST_PASSAGES}
        predictions={TEST_PREDICTIONS.slice(0, 2).map((prediction) => ({
          ...prediction,
          model: 'GPT-6 Astra',
          provenance: 'model' as const,
          predictionBasis: 'model_only_v1' as const,
          usage: {
            inputTokens: 2_000,
            outputTokens: 120,
            totalTokens: 2_120,
            reasoningTokens: 30,
            estimatedCostUsd: 0.026,
            pricingVersion: '2026-11-13',
          },
        }))}
        phase="complete"
        error={null}
        activePassageId="passage-1"
        onSelectPassage={vi.fn()}
        onDismissSelection={vi.fn()}
      />,
    )

    expect(screen.getAllByText(
      'Model-only prediction — no archive evidence used',
    )).toHaveLength(2)
    expect(screen.queryByText('GPT-6 Astra')).not.toBeInTheDocument()
    expect(screen.queryByText(/2,120 tokens/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/\$0\.0260 estimated/i)).not.toBeInTheDocument()
    expect(screen.queryByText('Simulated prediction')).not.toBeInTheDocument()
  })

  it('labels invented output as simulated and never as a model result', () => {
    render(
      <PredictionSidebar
        passages={TEST_PASSAGES}
        predictions={[{
          passageId: TEST_PREDICTIONS[0].passageId,
          page: TEST_PREDICTIONS[0].page,
          model: 'Demo LLM',
          text: TEST_PREDICTIONS[0].text,
          provenance: 'simulated',
        }]}
        phase="complete"
        error={null}
        activePassageId="passage-1"
        onSelectPassage={vi.fn()}
        onDismissSelection={vi.fn()}
      />,
    )

    expect(screen.getByText('Simulated prediction')).toBeInTheDocument()
    expect(screen.queryByText(
      'Model-only prediction — no archive evidence used',
    )).not.toBeInTheDocument()
  })

  it.each<[PredictionBasis, string]>([
    [
      'best_available_foiarchive_v1',
      'Archive-assisted prediction — available FOIArchive snapshot used',
    ],
    ['paper_exact_v1', 'Paper-method prediction'],
  ])('labels %s output with its exact evidence basis', (predictionBasis, label) => {
    render(
      <PredictionSidebar
        passages={TEST_PASSAGES}
        predictions={[{
          passageId: TEST_PREDICTIONS[0].passageId,
          page: TEST_PREDICTIONS[0].page,
          model: 'GPT-6 Astra',
          text: TEST_PREDICTIONS[0].text,
          provenance: 'model',
          predictionBasis,
        }]}
        phase="complete"
        error={null}
        activePassageId={null}
        onSelectPassage={vi.fn()}
        onDismissSelection={vi.fn()}
      />,
    )

    expect(screen.getByText(label)).toBeInTheDocument()
  })
})
