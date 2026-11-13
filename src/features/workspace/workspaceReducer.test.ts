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
      passages: [{ id: 'user-passage-1', label: 'Passage 1', page: 3, rects: [] }],
    }
    expect(workspaceReducer(dirtyState, { type: 'reset' })).toEqual(initialWorkspaceState)
  })

  it('stores every detected passage in backend document order', () => {
    const passages = [
      { id: 'passage-1', label: 'Passage 1', page: 1, rects: [] },
      { id: 'passage-2', label: 'Passage 2', page: 2, rects: [] },
    ]

    const state = workspaceReducer(initialWorkspaceState, {
      type: 'set-passages',
      passages,
    })

    expect(state.passages).toEqual(passages)
    expect(state.pinnedId).toBe('passage-1')
  })

  it('updates one corrected passage without dropping the others', () => {
    const initial = workspaceReducer(initialWorkspaceState, {
      type: 'set-passages',
      passages: [
        { id: 'passage-1', label: 'Passage 1', page: 1, rects: [] },
        { id: 'passage-2', label: 'Passage 2', page: 2, rects: [] },
      ],
    })

    const state = workspaceReducer(initial, {
      type: 'upsert-passage',
      passage: {
        id: 'passage-2',
        label: 'Passage 2',
        page: 2,
        rects: [{ x: 0.1, y: 0.2, width: 0.2, height: 0.03 }],
      },
    })

    expect(state.passages).toHaveLength(2)
    expect(state.passages[1].rects).toHaveLength(1)
  })

  it('replaces an optimistic passage ID with its durable backend ID', () => {
    const optimistic = workspaceReducer(initialWorkspaceState, {
      type: 'upsert-passage',
      passage: {
        id: 'pending-1',
        label: 'Passage 1',
        page: 1,
        rects: [{ x: 0.1, y: 0.2, width: 0.2, height: 0.03 }],
      },
    })

    const state = workspaceReducer(optimistic, {
      type: 'replace-passage',
      previousId: 'pending-1',
      passage: {
        id: 'durable-1',
        label: 'Passage 1',
        page: 1,
        rects: [{ x: 0.1, y: 0.2, width: 0.2, height: 0.03 }],
      },
    })

    expect(state.passages.map((passage) => passage.id)).toEqual(['durable-1'])
    expect(state.pinnedId).toBe('durable-1')
  })

  it('distinguishes adding a missed passage from correcting the selected passage', () => {
    const detected = workspaceReducer(initialWorkspaceState, {
      type: 'set-passages',
      passages: [
        { id: 'passage-1', label: 'Passage 1', page: 1, rects: [] },
        { id: 'passage-2', label: 'Passage 2', page: 2, rects: [] },
      ],
    })

    const adding = workspaceReducer(detected, {
      type: 'start-marking',
      passageId: null,
    })
    const correcting = workspaceReducer(detected, {
      type: 'start-marking',
      passageId: 'passage-2',
    })

    expect(adding).toMatchObject({
      markingEnabled: true,
      markingPassageId: null,
    })
    expect(correcting).toMatchObject({
      markingEnabled: true,
      markingPassageId: 'passage-2',
    })
  })
})
