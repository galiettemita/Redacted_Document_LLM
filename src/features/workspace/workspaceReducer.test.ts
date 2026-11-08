import { initialWorkspaceState, workspaceReducer } from './workspaceReducer'

describe('workspaceReducer', () => {
  it('navigates, pins, and returns to the document view as one prediction-selection action', () => {
    const state = workspaceReducer(
      { ...initialWorkspaceState, activeView: 'predictions' },
      { type: 'select-passage', passageId: 'passage-3', page: 2 },
    )
    expect(state).toMatchObject({ pinnedId: 'passage-3', page: 2, activeView: 'document' })
  })

  it('clears document-specific interaction state on replacement', () => {
    const dirtyState = {
      ...initialWorkspaceState,
      page: 3,
      hoveredId: 'passage-4',
      pinnedId: 'passage-4',
      activeView: 'predictions' as const,
      markingEnabled: true,
      markedPassage: { id: 'user-passage-1', label: 'Passage 1', page: 3, rects: [] },
    }
    expect(workspaceReducer(dirtyState, { type: 'reset' })).toEqual(initialWorkspaceState)
  })
})
