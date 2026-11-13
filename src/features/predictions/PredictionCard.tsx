import type { Passage, Prediction } from '../../types'

interface PredictionCardProps {
  passage: Passage
  prediction: Prediction
  onSelect: (passageId: string) => void
}

export function PredictionCard({ passage, prediction, onSelect }: PredictionCardProps) {
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
      <span className="prediction-model">{prediction.model}</span>
      <span className="prediction-text">{prediction.text}</span>
      {prediction.usage ? (
        <span className="prediction-usage">
          {prediction.usage.totalTokens.toLocaleString()} tokens
          {' · '}
          ${prediction.usage.estimatedCostUsd.toFixed(4)} estimated
        </span>
      ) : null}
      <span className="simulated-label">
        {prediction.provenance === 'model' ? 'Model prediction' : 'Simulated prediction'}
      </span>
    </button>
  )
}
