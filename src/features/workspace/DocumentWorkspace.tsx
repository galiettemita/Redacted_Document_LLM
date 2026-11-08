import { useEffect, useMemo, useReducer, useRef } from 'react'
import { MODEL_NAME } from '../../config'
import { SAMPLE_PASSAGES, SAMPLE_PREDICTIONS } from '../../data/sampleDocument'
import { usePredictionSimulation } from '../../hooks/usePredictionSimulation'
import { createMockPredictionService } from '../../services/mockPredictionService'
import type { DocumentDescriptor, Passage, PassageRect, Prediction } from '../../types'
import { PredictionSidebar } from '../predictions/PredictionSidebar'
import { PdfViewer } from '../viewer/PdfViewer'
import { getEffectivePassageId } from './selectionState'
import { initialWorkspaceState, workspaceReducer } from './workspaceReducer'

interface DocumentWorkspaceProps {
  document: DocumentDescriptor
  onReplace: () => void
  onViewSample: () => void
}

const UPLOADED_SAMPLE_TEXT = 'This marked area uses a sample prediction to demonstrate how a future analysis could connect a passage on the page with readable predicted text. No part of this PDF was analyzed.'

export function DocumentWorkspace({ document, onReplace, onViewSample }: DocumentWorkspaceProps) {
  const [state, dispatch] = useReducer(workspaceReducer, initialWorkspaceState)
  const passageTargets = useRef(new Map<string, HTMLElement>())
  const sampleService = useMemo(() => createMockPredictionService({
    predictionsByDocument: { [document.id]: document.mode === 'sample' ? SAMPLE_PREDICTIONS : [] },
    delayMs: 650,
  }), [document.id, document.mode])
  const simulation = usePredictionSimulation(document.mode === 'sample' ? document : null, sampleService)

  useEffect(() => {
    dispatch({ type: 'reset' })
  }, [document.id])

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
  const uploadedPredictions: Prediction[] = state.markedPassage ? [{
    passageId: state.markedPassage.id,
    page: state.markedPassage.page,
    model: MODEL_NAME,
    text: UPLOADED_SAMPLE_TEXT,
  }] : []
  const passages = document.mode === 'sample' ? SAMPLE_PASSAGES : uploadedPassages
  const predictions = document.mode === 'sample' ? simulation.predictions : uploadedPredictions
  const phase = document.mode === 'sample'
    ? simulation.phase
    : (uploadedPredictions.length ? 'complete' as const : 'idle' as const)
  const activePassageId = getEffectivePassageId(state.hoveredId, state.pinnedId)

  function handleMarkPassage(page: number, rect: PassageRect) {
    const passage: Passage = {
      id: 'user-passage-1',
      label: 'Passage 1',
      page,
      rects: [rect],
    }
    dispatch({ type: 'mark-passage', passage })
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
          onClick={() => {
            dispatch({ type: 'reset' })
            simulation.reset()
            onReplace()
          }}
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
                <strong>{state.markingEnabled ? 'Draw over one passage' : 'Interaction preview'}</strong>
                {state.markingEnabled
                  ? ' Drag a rectangle over the area you want to mark.'
                  : ' Mark one area to see how a future prediction could connect to it.'}
              </p>
              <button type="button" onClick={() => dispatch({ type: 'start-marking' })}>
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
            error={simulation.error}
            activePassageId={activePassageId}
            onSelectPassage={handleSelectPassage}
            onDismissSelection={() => dispatch({ type: 'dismiss' })}
          >
            {document.mode === 'uploaded' ? (
              <div className="uploaded-document-note">
                {!state.markedPassage ? (
                  <>
                    <h3>This PDF is displayed locally</h3>
                    <p>Automatic passage detection and predictions aren’t connected yet. Mark one area to preview the interaction, or explore the complete fictional sample.</p>
                  </>
                ) : (
                  <p>Your marked area is only an interaction preview. Explore the fictional sample for the complete automatic experience.</p>
                )}
                <button className="text-button" type="button" onClick={onViewSample}>View interactive sample</button>
              </div>
            ) : null}
          </PredictionSidebar>
        </div>
      </div>
    </main>
  )
}
