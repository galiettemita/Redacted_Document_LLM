import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { SAMPLE_PASSAGES, SAMPLE_PREDICTIONS } from '../../data/sampleDocument'
import type { BackendDocumentState } from '../../hooks/useBackendDocument'
import { usePredictionSimulation } from '../../hooks/usePredictionSimulation'
import {
  BackendApiError,
  backendApi as defaultBackendApi,
  type BackendApi,
  type BackendPassageRecord,
} from '../../services/backendApi'
import { createMockPredictionService } from '../../services/mockPredictionService'
import type { DocumentDescriptor, Passage, PassageRect } from '../../types'
import { PredictionSidebar } from '../predictions/PredictionSidebar'
import { PdfViewer } from '../viewer/PdfViewer'
import { getEffectivePassageId } from './selectionState'
import { initialWorkspaceState, workspaceReducer } from './workspaceReducer'

interface DocumentWorkspaceProps {
  document: DocumentDescriptor
  onReplace: () => void
  onViewSample: () => void
  backendDocument?: BackendDocumentState
  backendApi?: BackendApi
}

type PassageSaveStatus = 'idle' | 'saving' | 'saved' | 'failed'

export function DocumentWorkspace({
  document,
  onReplace,
  onViewSample,
  backendDocument,
  backendApi = defaultBackendApi,
}: DocumentWorkspaceProps) {
  const [state, dispatch] = useReducer(workspaceReducer, initialWorkspaceState)
  const [passageSaveStatus, setPassageSaveStatus] = useState<PassageSaveStatus>('idle')
  const [passageError, setPassageError] = useState<string | null>(null)
  const passageTargets = useRef(new Map<string, HTMLElement>())
  const passageRequest = useRef<AbortController | null>(null)
  const sampleService = useMemo(() => createMockPredictionService({
    predictionsByDocument: { [document.id]: document.mode === 'sample' ? SAMPLE_PREDICTIONS : [] },
    delayMs: 650,
  }), [document.id, document.mode])
  const simulation = usePredictionSimulation(document.mode === 'sample' ? document : null, sampleService)

  useEffect(() => {
    dispatch({ type: 'reset' })
    setPassageSaveStatus('idle')
    setPassageError(null)
    passageRequest.current?.abort()
    passageRequest.current = null
  }, [document.id])

  useEffect(() => () => passageRequest.current?.abort(), [])

  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') dispatch({ type: 'dismiss' })
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [])

  useEffect(() => {
    if (!state.pinnedId) return
    const target = passageTargets.current.get(state.pinnedId)
    target?.scrollIntoView?.({ block: 'center', behavior: 'smooth' })
    target?.focus({ preventScroll: true })
  }, [state.page, state.pinnedId])

  const uploadedPassages = state.markedPassage ? [state.markedPassage] : []
  const passages = document.mode === 'sample' ? SAMPLE_PASSAGES : uploadedPassages
  const predictions = document.mode === 'sample' ? simulation.predictions : []
  const phase = document.mode === 'sample'
    ? simulation.phase
    : 'idle' as const
  const activePassageId = getEffectivePassageId(state.hoveredId, state.pinnedId)
  const backendReady = document.mode === 'uploaded'
    && backendDocument?.status === 'ready'
    && Boolean(document.backendDocumentId)
  const markingUnavailable = !backendReady || passageSaveStatus === 'saving'

  async function handleMarkPassage(page: number, rect: PassageRect) {
    if (!backendReady || !document.backendDocumentId) return
    passageRequest.current?.abort()
    const controller = new AbortController()
    passageRequest.current = controller
    const existingPassage = state.markedPassage
    const passage: Passage = {
      id: existingPassage?.id ?? `pending-${createIdempotencyKey()}`,
      label: 'Passage 1',
      page,
      rects: [rect],
    }
    dispatch({ type: 'mark-passage', passage })
    setPassageSaveStatus('saving')
    setPassageError(null)

    try {
      let saved: BackendPassageRecord
      if (existingPassage && !existingPassage.id.startsWith('pending-')) {
        saved = await backendApi.updatePassage(
          existingPassage.id,
          { page, rects: [rect] },
          { signal: controller.signal },
        )
      } else {
        saved = await backendApi.createPassage(
          document.backendDocumentId,
          { label: passage.label, page, rects: [rect] },
          { signal: controller.signal, idempotencyKey: createIdempotencyKey() },
        )
      }
      dispatch({ type: 'mark-passage', passage: toPassage(saved) })
      setPassageSaveStatus('saved')
    } catch (error) {
      if (isAbortError(error)) return
      setPassageSaveStatus('failed')
      setPassageError(
        error instanceof BackendApiError || error instanceof Error
          ? error.message
          : 'The passage could not be saved.',
      )
    } finally {
      if (passageRequest.current === controller) passageRequest.current = null
    }
  }

  function handleSelectPassage(passageId: string) {
    const passage = passages.find((item) => item.id === passageId)
    if (!passage) return
    dispatch({ type: 'select-passage', passageId, page: passage.page })
  }

  return (
    <main className="workspace-shell">
      <header className="workspace-header">
        <div className="workspace-identity">
          <span className="workspace-brand">Redacted Document LLM</span>
          <span className="header-divider" aria-hidden="true" />
          <span className="filename" title={document.filename}>{document.filename}</span>
          {document.label ? <span className="fictional-badge">{document.label}</span> : null}
        </div>
        <button
          className="secondary-button"
          type="button"
          aria-label="Replace document"
          onClick={onReplace}
        >
          Replace document
        </button>
      </header>

      <nav className="mobile-view-switch" aria-label="Workspace view">
        <button
          type="button"
          aria-pressed={state.activeView === 'document'}
          onClick={() => dispatch({ type: 'set-view', view: 'document' })}
        >
          Document
        </button>
        <button
          type="button"
          aria-pressed={state.activeView === 'predictions'}
          onClick={() => dispatch({ type: 'set-view', view: 'predictions' })}
        >
          Predictions
          {predictions.length ? <span aria-hidden="true">{predictions.length}</span> : null}
        </button>
      </nav>

      <div className="workspace-grid">
        <div className="document-column" data-active={state.activeView === 'document'}>
          {document.mode === 'uploaded' ? (
            <div className="marking-toolbar">
              <p>
                <strong>{markingToolbarTitle(backendDocument, state.markingEnabled, passageSaveStatus)}</strong>
                {markingToolbarMessage(backendDocument, state.markingEnabled, passageSaveStatus)}
              </p>
              <button
                type="button"
                disabled={markingUnavailable}
                onClick={() => dispatch({ type: 'start-marking' })}
              >
                {state.markedPassage ? 'Mark another area' : 'Mark a passage'}
              </button>
            </div>
          ) : null}
          <PdfViewer
            source={document.source}
            filename={document.filename}
            passages={passages}
            page={state.page}
            onPageChange={(page) => dispatch({ type: 'set-page', page })}
            markingEnabled={state.markingEnabled}
            onMarkPassage={handleMarkPassage}
            hoveredId={state.hoveredId}
            pinnedId={state.pinnedId}
            onHoverPassage={(passageId) => dispatch({ type: 'hover', passageId })}
            onPinPassage={(passageId) => dispatch({ type: 'pin', passageId })}
            onDismissSelection={() => dispatch({ type: 'dismiss' })}
            registerPassageTarget={(passageId, element) => {
              if (element) passageTargets.current.set(passageId, element)
              else passageTargets.current.delete(passageId)
            }}
          />
        </div>
        <div className="predictions-column" data-active={state.activeView === 'predictions'}>
          <PredictionSidebar
            passages={passages}
            predictions={predictions}
            phase={phase}
            error={document.mode === 'sample'
              ? simulation.error
              : passageError ?? backendDocumentError(backendDocument)}
            activePassageId={activePassageId}
            onSelectPassage={handleSelectPassage}
            onDismissSelection={() => dispatch({ type: 'dismiss' })}
            missingPredictionMessage={document.mode === 'uploaded'
              ? passageSaveStatus === 'saving'
                ? 'Saving this passage to the backend…'
                : 'No prediction has been requested. Retrieval and model prediction are not connected yet.'
              : undefined}
          >
            {document.mode === 'uploaded' ? (
              <div className="uploaded-document-note">
                <h3>{backendNoteTitle(backendDocument, passageSaveStatus)}</h3>
                <p>{backendNoteMessage(backendDocument, passageSaveStatus)}</p>
                <button className="text-button" type="button" onClick={onViewSample}>View interactive sample</button>
              </div>
            ) : null}
          </PredictionSidebar>
        </div>
      </div>
    </main>
  )
}

function toPassage(record: BackendPassageRecord): Passage {
  return {
    id: record.id,
    label: record.label,
    page: record.page_number,
    rects: record.rects,
  }
}

function createIdempotencyKey(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') return globalThis.crypto.randomUUID()
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function isAbortError(error: unknown): boolean {
  return typeof error === 'object'
    && error !== null
    && 'name' in error
    && error.name === 'AbortError'
}

function markingToolbarTitle(
  backendDocument: BackendDocumentState | undefined,
  markingEnabled: boolean,
  passageSaveStatus: PassageSaveStatus,
): string {
  if (backendDocument?.status === 'uploading') return 'Uploading PDF'
  if (backendDocument?.status === 'deleting') return 'Removing PDF'
  if (backendDocument?.status === 'unavailable') return 'Backend unavailable'
  if (backendDocument?.status === 'failed') return 'Upload not saved'
  if (passageSaveStatus === 'saving') return 'Saving passage'
  return markingEnabled ? 'Draw over one passage' : 'Mark a real passage'
}

function markingToolbarMessage(
  backendDocument: BackendDocumentState | undefined,
  markingEnabled: boolean,
  passageSaveStatus: PassageSaveStatus,
): string {
  if (backendDocument?.status === 'uploading') {
    return ' Uploading to the loopback-only local backend. Marking becomes available when persistence finishes.'
  }
  if (backendDocument?.status === 'unavailable') {
    return ' The local PDF remains visible, but passage persistence needs the local backend.'
  }
  if (backendDocument?.status === 'failed') {
    return ` ${backendDocument.error ?? 'The PDF could not be persisted.'}`
  }
  if (passageSaveStatus === 'saving') return ' Keeping this rectangle linked to its stable backend passage.'
  return markingEnabled
    ? ' Drag a rectangle over the area you want to save.'
    : ' Mark one area to save its normalized coordinates under a stable passage ID.'
}

function backendNoteTitle(
  backendDocument: BackendDocumentState | undefined,
  passageSaveStatus: PassageSaveStatus,
): string {
  if (backendDocument?.status === 'uploading') return 'Backend upload in progress'
  if (backendDocument?.status === 'unavailable') return 'Local backend unavailable'
  if (backendDocument?.status === 'failed') return 'PDF upload failed'
  if (passageSaveStatus === 'saving') return 'Saving Passage 1'
  if (passageSaveStatus === 'saved') return 'Passage 1 is saved'
  if (passageSaveStatus === 'failed') return 'Passage save failed'
  return 'PDF upload complete'
}

function backendNoteMessage(
  backendDocument: BackendDocumentState | undefined,
  passageSaveStatus: PassageSaveStatus,
): string {
  if (backendDocument?.status === 'uploading') {
    return 'Uploading to the loopback-only local backend while the browser displays its local copy.'
  }
  if (backendDocument?.status === 'unavailable') {
    return 'The PDF is displayed locally, but the backend copy and passage coordinates have not been saved.'
  }
  if (backendDocument?.status === 'failed') {
    return backendDocument.error ?? 'The backend rejected this PDF.'
  }
  if (passageSaveStatus === 'saved') {
    return 'Passage 1 is saved. The visible-only extraction engine is installed; automatic processing, retrieval and model prediction are not connected in this screen yet.'
  }
  if (passageSaveStatus === 'failed') {
    return 'The rectangle remains visible locally, but it was not saved. Try marking the passage again.'
  }
  return 'The PDF is stored temporarily by the local backend. You can save one passage now; automatic retrieval and model prediction are not connected yet.'
}

function backendDocumentError(backendDocument: BackendDocumentState | undefined): string | null {
  if (backendDocument?.status !== 'failed' && backendDocument?.status !== 'unavailable') return null
  return backendDocument.error ?? 'The local backend is unavailable.'
}
