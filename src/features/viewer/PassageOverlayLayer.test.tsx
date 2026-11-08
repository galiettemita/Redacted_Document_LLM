import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { SAMPLE_PASSAGES, SAMPLE_PREDICTIONS } from '../../data/sampleDocument'
import { PassageOverlayLayer } from './PassageOverlayLayer'

function OverlayHarness() {
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [pinnedId, setPinnedId] = useState<string | null>('passage-1')
  return (
    <div style={{ position: 'relative', width: 600, height: 800 }}>
      <PassageOverlayLayer
        page={1}
        passages={SAMPLE_PASSAGES}
        predictions={SAMPLE_PREDICTIONS}
        hoveredId={hoveredId}
        pinnedId={pinnedId}
        onHover={setHoveredId}
        onPin={setPinnedId}
        onDismiss={() => setPinnedId(null)}
      />
    </div>
  )
}

describe('PassageOverlayLayer', () => {
  it('activates every rectangle in a multi-line passage and restores the pin after hover', () => {
    render(<OverlayHarness />)
    const passageOne = screen.getAllByRole('button', { name: 'Passage 1 redaction' })
    const passageTwo = screen.getAllByRole('button', { name: 'Passage 2 redaction' })
    expect(passageTwo).toHaveLength(2)
    expect(passageOne.every((rect) => rect.dataset.active === 'true')).toBe(true)

    fireEvent.mouseEnter(passageTwo[1])
    expect(passageTwo.every((rect) => rect.dataset.active === 'true')).toBe(true)
    expect(passageOne.every((rect) => rect.dataset.active === 'false')).toBe(true)
    expect(screen.getByRole('status')).toHaveTextContent('Field offices were directed')

    fireEvent.mouseLeave(passageTwo[1])
    expect(passageOne.every((rect) => rect.dataset.active === 'true')).toBe(true)
  })

  it('pins with Enter or Space and dismisses with Escape', () => {
    render(<OverlayHarness />)
    const target = screen.getAllByRole('button', { name: 'Passage 2 redaction' })[0]

    fireEvent.keyDown(target, { key: 'Enter' })
    expect(screen.getAllByRole('button', { name: 'Passage 2 redaction' })[0]).toHaveAttribute('data-active', 'true')

    fireEvent.keyDown(target, { key: 'Escape' })
    expect(screen.getAllByRole('button', { name: 'Passage 2 redaction' })[0]).toHaveAttribute('data-active', 'false')
  })
})
