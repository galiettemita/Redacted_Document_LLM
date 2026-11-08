import type { Passage, Prediction } from '../../types'

interface SelectedPassageProps {
  passage: Passage | null
  prediction: Prediction | null
  onDismiss: () => void
}

export function SelectedPassage({ passage, prediction, onDismiss }: SelectedPassageProps) {
  return (
    <section className="selected-passage" aria-labelledby="selected-passage-heading">
      <div className="selected-heading-row">
        <h2 id="selected-passage-heading">Selected passage</h2>
        {passage ? (
          <button className="dismiss-selection" type="button" aria-label="Dismiss selected passage" onClick={onDismiss}>
            ×
          </button>
        ) : null}
      </div>
      {passage ? (
        <div className="selected-content">
          <div className="prediction-card-heading">
            <span className="prediction-label">{passage.label}</span>
            <span className="page-label">Page {passage.page}</span>
          </div>
          {prediction ? (
            <>
              <p className="prediction-model">{prediction.model}</p>
              <p className="prediction-text">{prediction.text}</p>
              <p className="simulated-label">Simulated prediction</p>
            </>
          ) : (
            <p className="selected-empty">Prediction is still processing…</p>
          )}
        </div>
      ) : (
        <p className="selected-empty">Hover over or select a redacted passage to inspect its prediction.</p>
      )}
    </section>
  )
}
