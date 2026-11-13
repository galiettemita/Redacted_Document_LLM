import type { Passage } from '../../types'

export interface WorkspaceState {
  page: number
  hoveredId: string | null
  pinnedId: string | null
  activeView: 'document' | 'predictions'
  markingEnabled: boolean
  markingPassageId: string | null
  passages: Passage[]
}

export const initialWorkspaceState: WorkspaceState = {
  page: 1,
  hoveredId: null,
  pinnedId: null,
  activeView: 'document',
  markingEnabled: false,
  markingPassageId: null,
  passages: [],
}

type WorkspaceAction =
  | { type: 'reset' }
  | { type: 'set-page'; page: number }
  | { type: 'hover'; passageId: string | null }
  | { type: 'pin'; passageId: string | null }
  | { type: 'dismiss' }
  | { type: 'set-view'; view: WorkspaceState['activeView'] }
  | { type: 'start-marking'; passageId: string | null }
  | { type: 'set-passages'; passages: Passage[] }
  | { type: 'upsert-passage'; passage: Passage }
  | { type: 'replace-passage'; previousId: string; passage: Passage }
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
      return {
        ...state,
        activeView: 'document',
        markingEnabled: true,
        markingPassageId: action.passageId,
        hoveredId: null,
      }
    case 'set-passages':
      return {
        ...state,
        passages: action.passages,
        markingEnabled: false,
        markingPassageId: null,
        pinnedId: action.passages.some((passage) => passage.id === state.pinnedId)
          ? state.pinnedId
          : action.passages[0]?.id ?? null,
      }
    case 'upsert-passage':
      return {
        ...state,
        page: action.passage.page,
        passages: state.passages.some((passage) => passage.id === action.passage.id)
          ? state.passages.map((passage) => (
              passage.id === action.passage.id ? action.passage : passage
            ))
          : [...state.passages, action.passage],
        markingEnabled: false,
        markingPassageId: null,
        pinnedId: action.passage.id,
      }
    case 'replace-passage':
      return {
        ...state,
        page: action.passage.page,
        passages: state.passages.map((passage) => (
          passage.id === action.previousId ? action.passage : passage
        )),
        markingEnabled: false,
        markingPassageId: null,
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
