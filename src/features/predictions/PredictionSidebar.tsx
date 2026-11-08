import type { Passage, Prediction, PredictionPhase } from '../../types'
import { PredictionCard } from './PredictionCard'
import { SelectedPassage } from './SelectedPassage'

interface PredictionSidebarProps {
  passages: Passage[]
  predictions: Prediction[]
  phase: PredictionPhase
  error: string | null
  activePassageId: string | null
  onSelectPassage: (passageId: string) => void
  onDismissSelection: () => void
  children?: React.ReactNode
}

export function PredictionSidebar({
  passages,
  predictions,
  phase,
  error,
  activePassageId,
  onSelectPassage,
  onDismissSelection,
  children,
}: PredictionSidebarProps) {
  const activePassage = passages.find((passage) => passage.id === activePassageId) ?? null
  const activePrediction = predictions.find((prediction) => prediction.passageId === activePassageId) ?? null
  const visiblePredictions = predictions.filter((prediction) => prediction.passageId !== activePassageId)

  return (
    <aside className="prediction-sidebar" aria-label="Passage predictions">
      <div className="sidebar-scroll">
        <SelectedPassage
          passage={activePassage}
          prediction={activePrediction}
          onDismiss={onDismissSelection}
        />

        <div className="prediction-list-heading">
          <h2>Predictions</h2>
          {(phase === 'predicting' || phase === 'partial') ? (
            <p className="processing-label"><span aria-hidden="true" />Predicting hidden passages…</p>
          ) : null}
        </div>

        {error ? <p className="sidebar-state is-error" role="alert">{error}</p> : null}
        {phase === 'empty' ? <p className="sidebar-state">No simulated passages are available for this document.</p> : null}
        {children}
        <div className="prediction-list">
          {visiblePredictions.map((prediction) => {
            const passage = passages.find((item) => item.id === prediction.passageId)
            return passage ? (
              <PredictionCard
                key={prediction.passageId}
                passage={passage}
                prediction={prediction}
                onSelect={onSelectPassage}
              />
            ) : null
          })}
        </div>
      </div>
    </aside>
  )
}
