import type { Passage, Prediction } from '../../types'
import { predictionProvenanceLabel } from './predictionProvenance'

interface PredictionCardProps {
  passage: Passage
  prediction: Prediction
  onSelect: (passageId: string) => void
}

export function PredictionCard({ passage, prediction, onSelect }: PredictionCardProps) {
  const provenanceLabel = predictionProvenanceLabel(prediction)
  return (
    <button
      className="prediction-card"
      type="button"
      aria-label={`Open ${passage.label} on page ${passage.page}`}
      onClick={() => onSelect(passage.id)}
    >
      <span className="prediction-card-heading">
        <span className="prediction-label">{passage.label}</span>
        <span className="page-label">Page {passage.page}</span>
      </span>
      <span className="prediction-text">{prediction.text}</span>
      {provenanceLabel ? (
        <span className="prediction-provenance">{provenanceLabel}</span>
      ) : null}
    </button>
  )
}
