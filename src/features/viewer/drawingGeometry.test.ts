import { normalizeDrag } from './drawingGeometry'

const bounds = { left: 100, top: 50, width: 200, height: 400 }

describe('normalizeDrag', () => {
  it('normalizes a reverse-direction drag into page-relative coordinates', () => {
    expect(normalizeDrag({ x: 250, y: 350 }, { x: 150, y: 150 }, bounds)).toEqual({
      x: 0.25,
      y: 0.25,
      width: 0.5,
      height: 0.5,
    })
  })

  it('clamps pointer positions outside the rendered page', () => {
    expect(normalizeDrag({ x: 50, y: 0 }, { x: 350, y: 500 }, bounds)).toEqual({
      x: 0,
      y: 0,
      width: 1,
      height: 1,
    })
  })

  it('rejects a drag too small to mark intentionally', () => {
    expect(normalizeDrag({ x: 120, y: 70 }, { x: 122, y: 72 }, bounds)).toBeNull()
  })
})
