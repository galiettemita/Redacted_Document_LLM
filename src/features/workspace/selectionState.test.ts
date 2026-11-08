import { getEffectivePassageId } from './selectionState'

describe('getEffectivePassageId', () => {
  it('temporarily prefers hover and restores the pinned passage afterward', () => {
    expect(getEffectivePassageId('hovered', 'pinned')).toBe('hovered')
    expect(getEffectivePassageId(null, 'pinned')).toBe('pinned')
    expect(getEffectivePassageId(null, null)).toBeNull()
  })
})
