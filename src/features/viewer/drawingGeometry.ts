import type { PassageRect } from '../../types'

interface Point {
  x: number
  y: number
}

interface Bounds {
  left: number
  top: number
  width: number
  height: number
}

const clamp = (value: number, minimum: number, maximum: number) =>
  Math.min(maximum, Math.max(minimum, value))

export function normalizeDrag(
  start: Point,
  end: Point,
  bounds: Bounds,
  minimumPixels = 4,
): PassageRect | null {
  const startX = clamp(start.x, bounds.left, bounds.left + bounds.width)
  const startY = clamp(start.y, bounds.top, bounds.top + bounds.height)
  const endX = clamp(end.x, bounds.left, bounds.left + bounds.width)
  const endY = clamp(end.y, bounds.top, bounds.top + bounds.height)
  const left = Math.min(startX, endX)
  const top = Math.min(startY, endY)
  const width = Math.abs(endX - startX)
  const height = Math.abs(endY - startY)

  if (width < minimumPixels || height < minimumPixels || bounds.width <= 0 || bounds.height <= 0) {
    return null
  }

  return {
    x: (left - bounds.left) / bounds.width,
    y: (top - bounds.top) / bounds.height,
    width: width / bounds.width,
    height: height / bounds.height,
  }
}
