import { useEffect, useRef, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import { configurePdfViewerWorker } from '../../pdfRuntime'
import type { Passage, PassageRect, Prediction } from '../../types'
import { PassageOverlayLayer } from './PassageOverlayLayer'
import { PdfToolbar } from './PdfToolbar'

configurePdfViewerWorker(pdfjs)

interface PdfViewerProps {
  source: string
  filename: string
  passages: Passage[]
  predictions?: Prediction[]
  hoveredId?: string | null
  pinnedId?: string | null
  onHoverPassage?: (passageId: string | null) => void
  onPinPassage?: (passageId: string | null) => void
  onDismissSelection?: () => void
  page: number
  onPageChange: (page: number) => void
  markingEnabled?: boolean
  onMarkPassage?: (page: number, rect: PassageRect) => void
  registerPassageTarget?: (passageId: string, element: HTMLElement | null) => void
}

export function PdfViewer({
  source,
  filename,
  passages,
  predictions = [],
  hoveredId = null,
  pinnedId = null,
  onHoverPassage,
  onPinPassage,
  onDismissSelection,
  page,
  onPageChange,
  markingEnabled = false,
  onMarkPassage,
  registerPassageTarget,
}: PdfViewerProps) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const [pageCount, setPageCount] = useState(0)
  const [scale, setScale] = useState(1)
  const [availableWidth, setAvailableWidth] = useState(760)

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    const updateWidth = () => setAvailableWidth(Math.max(320, viewport.clientWidth - 64))
    updateWidth()
    const observer = new ResizeObserver(updateWidth)
    observer.observe(viewport)
    return () => observer.disconnect()
  }, [])

  const renderedWidth = Math.min(760, availableWidth) * scale

  return (
    <section className="pdf-viewer" aria-label={`Document viewer for ${filename}`}>
      <PdfToolbar
        page={page}
        pageCount={pageCount}
        scale={scale}
        onPageChange={onPageChange}
        onScaleChange={setScale}
        onFitWidth={() => setScale(1)}
      />
      <div ref={viewportRef} className="pdf-viewport">
        <Document
          file={source}
          loading={<p className="viewer-state">Opening PDF…</p>}
          error={<p className="viewer-state is-error">This PDF could not be displayed. It has not left your device.</p>}
          onLoadSuccess={({ numPages }) => {
            setPageCount(numPages)
            if (page > numPages) onPageChange(1)
          }}
        >
          <div className="pdf-page-shell" style={{ width: renderedWidth }}>
            <Page
              pageNumber={page}
              width={renderedWidth}
              renderAnnotationLayer={false}
              renderTextLayer={false}
              loading={<p className="viewer-state">Rendering page {page}…</p>}
            />
            <PassageOverlayLayer
              page={page}
              passages={passages}
              predictions={predictions}
              hoveredId={hoveredId}
              pinnedId={pinnedId}
              onHover={onHoverPassage}
              onPin={onPinPassage}
              onDismiss={onDismissSelection}
              markingEnabled={markingEnabled}
              onMarkPassage={onMarkPassage}
              registerPassageTarget={registerPassageTarget}
            />
          </div>
        </Document>
      </div>
    </section>
  )
}
