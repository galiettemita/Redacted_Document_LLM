import type { Passage } from '../../types'

export interface WorkspaceState {
  page: number
  hoveredId: string | null
  pinnedId: string | null
  activeView: 'document' | 'predictions'
  markingEnabled: boolean
  markedPassage: Passage | null
}

export const initialWorkspaceState: WorkspaceState = {
  page: 1,
  hoveredId: null,
  pinnedId: null,
  activeView: 'document',
  markingEnabled: false,
  markedPassage: null,
}

type WorkspaceAction =
  | { type: 'reset' }
  | { type: 'set-page'; page: number }
  | { type: 'hover'; passageId: string | null }
  | { type: 'pin'; passageId: string | null }
  | { type: 'dismiss' }
  | { type: 'set-view'; view: WorkspaceState['activeView'] }
  | { type: 'start-marking' }
  | { type: 'mark-passage'; passage: Passage }
  | { type: 'select-passage'; passageId: string; page: number }

export function workspaceReducer(state: WorkspaceState, action: WorkspaceAction): WorkspaceState {
  switch (action.type) {
    case 'reset':
      return initialWorkspaceState
    case 'set-page':
      return { ...state, page: action.page }
    case 'hover':
      return { ...state, hoveredId: action.passageId }
    case 'pin':
      return { ...state, pinnedId: action.passageId }
    case 'dismiss':
      return { ...state, hoveredId: null, pinnedId: null }
    case 'set-view':
      return { ...state, activeView: action.view }
    case 'start-marking':
      return { ...state, activeView: 'document', markingEnabled: true, hoveredId: null, pinnedId: null }
    case 'mark-passage':
      return {
        ...state,
        page: action.passage.page,
        markedPassage: action.passage,
        markingEnabled: false,
        pinnedId: action.passage.id,
      }
    case 'select-passage':
      return {
        ...state,
        page: action.page,
        pinnedId: action.passageId,
        hoveredId: null,
        activeView: 'document',
      }
  }
}
