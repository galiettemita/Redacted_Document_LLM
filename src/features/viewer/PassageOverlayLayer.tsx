import { useRef, useState, type PointerEvent } from 'react'
import type { Passage, PassageRect, Prediction } from '../../types'
import { getEffectivePassageId } from '../workspace/selectionState'
import { normalizeDrag } from './drawingGeometry'

interface Point {
  x: number
  y: number
}

interface PassageOverlayLayerProps {
  page: number
  passages: Passage[]
  predictions?: Prediction[]
  hoveredId?: string | null
  pinnedId?: string | null
  onHover?: (passageId: string | null) => void
  onPin?: (passageId: string | null) => void
  onDismiss?: () => void
  markingEnabled?: boolean
  onMarkPassage?: (page: number, rect: PassageRect) => void
  registerPassageTarget?: (passageId: string, element: HTMLElement | null) => void
}

export function PassageOverlayLayer({
  page,
  passages,
  predictions = [],
  hoveredId = null,
  pinnedId = null,
  onHover,
  onPin,
  onDismiss,
  markingEnabled = false,
  onMarkPassage,
  registerPassageTarget,
}: PassageOverlayLayerProps) {
  const layerRef = useRef<HTMLDivElement>(null)
  const [dragStart, setDragStart] = useState<Point | null>(null)
  const [dragCurrent, setDragCurrent] = useState<Point | null>(null)

  function pointFromEvent(event: PointerEvent<HTMLDivElement>): Point {
    return { x: event.clientX, y: event.clientY }
  }

  function finishMark(event: PointerEvent<HTMLDivElement>) {
    if (!dragStart || !layerRef.current || !markingEnabled) return
    const bounds = layerRef.current.getBoundingClientRect()
    const rect = normalizeDrag(dragStart, pointFromEvent(event), bounds)
    setDragStart(null)
    setDragCurrent(null)
    if (rect) onMarkPassage?.(page, rect)
  }

  let draftRect: PassageRect | null = null
  if (dragStart && dragCurrent && layerRef.current) {
    draftRect = normalizeDrag(dragStart, dragCurrent, layerRef.current.getBoundingClientRect(), 0)
  }

  const pagePassages = passages.filter((passage) => passage.page === page)
  const activeId = getEffectivePassageId(hoveredId, pinnedId)
  const activePassage = pagePassages.find((passage) => passage.id === activeId) ?? null
  const activePrediction = predictions.find((prediction) => prediction.passageId === activeId)
  const previewPosition = activePassage ? getPreviewPosition(activePassage) : null

  return (
    <div
      ref={layerRef}
      className={`passage-overlay-layer${markingEnabled ? ' is-marking' : ''}`}
      data-testid="passage-overlay-layer"
      style={{ touchAction: markingEnabled ? 'none' : undefined }}
      onPointerDown={(event) => {
        if (!markingEnabled || event.button !== 0) return
        event.currentTarget.setPointerCapture(event.pointerId)
        const point = pointFromEvent(event)
        setDragStart(point)
        setDragCurrent(point)
      }}
      onPointerMove={(event) => {
        if (dragStart) setDragCurrent(pointFromEvent(event))
      }}
      onPointerUp={finishMark}
      onPointerCancel={() => {
        setDragStart(null)
        setDragCurrent(null)
      }}
    >
      {pagePassages.flatMap((passage) =>
        passage.rects.map((rect, index) => (
          <button
            key={`${passage.id}-${index}`}
            ref={index === 0 ? (element) => registerPassageTarget?.(passage.id, element) : undefined}
            className="passage-rect"
            type="button"
            aria-label={`${passage.label} redaction`}
            tabIndex={index === 0 ? 0 : -1}
            data-passage-id={passage.id}
            data-active={activeId === passage.id}
            onMouseEnter={() => onHover?.(passage.id)}
            onMouseLeave={() => onHover?.(null)}
            onFocus={() => onHover?.(passage.id)}
            onBlur={() => onHover?.(null)}
            onClick={(event) => {
              event.stopPropagation()
              onPin?.(passage.id)
            }}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                onPin?.(passage.id)
              }
              if (event.key === 'Escape') {
                event.preventDefault()
                onHover?.(null)
                onDismiss?.()
              }
            }}
            style={{
              left: `${rect.x * 100}%`,
              top: `${rect.y * 100}%`,
              width: `${rect.width * 100}%`,
              height: `${rect.height * 100}%`,
            }}
          />
        )),
      )}
      {activePassage && previewPosition ? (
        <div
          className="passage-preview"
          role="status"
          style={{ left: `${previewPosition.left * 100}%`, top: `${previewPosition.top * 100}%` }}
        >
          <span>{activePassage.label} · Page {activePassage.page}</span>
          <p>{activePrediction?.text ?? 'Prediction is still processing…'}</p>
        </div>
      ) : null}
      {draftRect ? (
        <span
          className="passage-draft"
          style={{
            left: `${draftRect.x * 100}%`,
            top: `${draftRect.y * 100}%`,
            width: `${draftRect.width * 100}%`,
            height: `${draftRect.height * 100}%`,
          }}
        />
      ) : null}
    </div>
  )
}

function getPreviewPosition(passage: Passage) {
  const left = Math.max(0.02, Math.min(0.7, Math.min(...passage.rects.map((rect) => rect.x))))
  const topEdge = Math.min(...passage.rects.map((rect) => rect.y))
  const bottomEdge = Math.max(...passage.rects.map((rect) => rect.y + rect.height))
  return { left, top: topEdge > 0.16 ? topEdge - 0.13 : Math.min(0.84, bottomEdge + 0.025) }
}
