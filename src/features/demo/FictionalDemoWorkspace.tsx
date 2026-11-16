import { useEffect, useMemo, useReducer, useRef } from 'react'
import {
  SAMPLE_DOCUMENT,
  SAMPLE_PASSAGES,
  SAMPLE_PREDICTIONS,
} from '../../data/sampleDocument'
import { usePredictionSimulation } from '../../hooks/usePredictionSimulation'
import { createMockPredictionService } from '../../services/mockPredictionService'
import { PredictionSidebar } from '../predictions/PredictionSidebar'
import { PdfViewer } from '../viewer/PdfViewer'
import { getEffectivePassageId } from '../workspace/selectionState'
import {
  initialWorkspaceState,
  workspaceReducer,
} from '../workspace/workspaceReducer'

interface FictionalDemoWorkspaceProps {
  onClose: () => void
  returnLabel: string
}

export function FictionalDemoWorkspace({
  onClose,
  returnLabel,
}: FictionalDemoWorkspaceProps) {
  const [state, dispatch] = useReducer(workspaceReducer, {
    ...initialWorkspaceState,
    passages: SAMPLE_PASSAGES,
  })
  const passageTargets = useRef(new Map<string, HTMLElement>())
  const service = useMemo(
    () => createMockPredictionService({
      predictionsByDocument: {
        [SAMPLE_DOCUMENT.id]: SAMPLE_PREDICTIONS,
      },
    }),
    [],
  )
  const simulation = usePredictionSimulation(SAMPLE_DOCUMENT, service)
  const activePassageId = getEffectivePassageId(
    state.hoveredId,
    state.pinnedId,
  )

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [onClose])

  useEffect(() => {
    if (!state.pinnedId) return
    const target = passageTargets.current.get(state.pinnedId)
    target?.scrollIntoView?.({ block: 'center', behavior: 'smooth' })
    target?.focus({ preventScroll: true })
  }, [state.page, state.pinnedId])

  function selectPassage(passageId: string) {
    const passage = SAMPLE_PASSAGES.find((item) => item.id === passageId)
    if (!passage) return
    dispatch({
      type: 'select-passage',
      passageId,
      page: passage.page,
    })
  }

  return (
    <main
      className="workspace-shell fictional-demo-workspace"
      role="dialog"
      aria-modal="true"
      aria-labelledby="fictional-demo-title"
    >
      <header className="workspace-header">
        <div className="workspace-identity">
          <span className="workspace-brand">Redacted Document LLM</span>
          <span className="header-divider" aria-hidden="true" />
          <span id="fictional-demo-title" className="filename">
            Fictional interactive demo
          </span>
          <span className="fictional-badge">Simulated</span>
        </div>
        <button
          autoFocus
          className="secondary-button"
          type="button"
          onClick={onClose}
        >
          {returnLabel}
        </button>
      </header>

      <p className="demo-disclosure">
        This demonstration uses invented content and simulated predictions.
      </p>

      <nav className="mobile-view-switch" aria-label="Demo workspace view">
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
          {simulation.predictions.length ? (
            <span aria-hidden="true">{simulation.predictions.length}</span>
          ) : null}
        </button>
      </nav>

      <div className="workspace-grid">
        <div
          className="document-column"
          data-active={state.activeView === 'document'}
        >
          <PdfViewer
            source={SAMPLE_DOCUMENT.source}
            filename={SAMPLE_DOCUMENT.filename}
            passages={SAMPLE_PASSAGES}
            page={state.page}
            onPageChange={(page) => dispatch({ type: 'set-page', page })}
            hoveredId={state.hoveredId}
            pinnedId={state.pinnedId}
            onHoverPassage={(passageId) => dispatch({
              type: 'hover',
              passageId,
            })}
            onPinPassage={(passageId) => dispatch({
              type: 'pin',
              passageId,
            })}
            onDismissSelection={() => dispatch({ type: 'dismiss' })}
            registerPassageTarget={(passageId, element) => {
              if (element) passageTargets.current.set(passageId, element)
              else passageTargets.current.delete(passageId)
            }}
          />
        </div>
        <div
          className="predictions-column"
          data-active={state.activeView === 'predictions'}
        >
          <PredictionSidebar
            passages={SAMPLE_PASSAGES}
            predictions={simulation.predictions}
            phase={simulation.phase}
            error={simulation.error}
            activePassageId={activePassageId}
            onSelectPassage={selectPassage}
            onDismissSelection={() => dispatch({ type: 'dismiss' })}
            missingPredictionMessage="This simulated prediction is still processing…"
          />
        </div>
      </div>
    </main>
  )
}
