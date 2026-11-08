import { MODEL_NAME } from '../config'
import { SAMPLE_DOCUMENT, SAMPLE_PASSAGES, SAMPLE_PREDICTIONS } from './sampleDocument'

describe('fictional sample fixture', () => {
  it('keeps every prediction aligned to one ordered passage with valid page-relative geometry', () => {
    expect(SAMPLE_DOCUMENT).toMatchObject({
      mode: 'sample',
      filename: 'fictional-briefing.pdf',
      pageCount: 3,
      label: 'Fictional demonstration document',
    })

    const ids = SAMPLE_PASSAGES.map((passage) => passage.id)
    expect(new Set(ids).size).toBe(ids.length)
    expect(SAMPLE_PASSAGES.map((passage) => passage.page)).toEqual([1, 1, 2, 2, 3])
    expect(SAMPLE_PASSAGES.some((passage) => passage.rects.length > 1)).toBe(true)

    for (const passage of SAMPLE_PASSAGES) {
      for (const rect of passage.rects) {
        expect(rect.x).toBeGreaterThanOrEqual(0)
        expect(rect.y).toBeGreaterThanOrEqual(0)
        expect(rect.x + rect.width).toBeLessThanOrEqual(1)
        expect(rect.y + rect.height).toBeLessThanOrEqual(1)
      }
    }

    expect(SAMPLE_PREDICTIONS.map((prediction) => prediction.passageId)).toEqual(ids)
    expect(SAMPLE_PREDICTIONS.every((prediction) => prediction.model === MODEL_NAME)).toBe(true)
    expect(SAMPLE_PREDICTIONS.every((prediction) => prediction.text.length > 80)).toBe(true)
  })
})
