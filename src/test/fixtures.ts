import type { Passage, Prediction } from '../types'

export const TEST_PASSAGES: Passage[] = [
  {
    id: 'passage-1',
    label: 'Passage 1',
    page: 1,
    rects: [{ x: 0.1, y: 0.2, width: 0.4, height: 0.06 }],
  },
  {
    id: 'passage-2',
    label: 'Passage 2',
    page: 1,
    rects: [
      { x: 0.2, y: 0.4, width: 0.5, height: 0.04 },
      { x: 0.2, y: 0.45, width: 0.35, height: 0.04 },
    ],
  },
  {
    id: 'passage-3',
    label: 'Passage 3',
    page: 2,
    rects: [{ x: 0.15, y: 0.3, width: 0.45, height: 0.07 }],
  },
]

export const TEST_PREDICTIONS: Prediction[] = TEST_PASSAGES.map((passage) => ({
  passageId: passage.id,
  page: passage.page,
  model: 'GPT-6 Astra',
  text: `Provider prediction for ${passage.label}.`,
  provenance: 'model',
}))
