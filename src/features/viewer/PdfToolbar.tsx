interface PdfToolbarProps {
  page: number
  pageCount: number
  scale: number
  onPageChange: (page: number) => void
  onScaleChange: (scale: number) => void
  onFitWidth: () => void
}

const clampScale = (scale: number) => Math.min(2, Math.max(0.6, Math.round(scale * 10) / 10))

export function PdfToolbar({
  page,
  pageCount,
  scale,
  onPageChange,
  onScaleChange,
  onFitWidth,
}: PdfToolbarProps) {
  return (
    <div className="pdf-toolbar" aria-label="PDF controls">
      <div className="toolbar-group">
        <button
          className="icon-button"
          type="button"
          aria-label="Previous page"
          disabled={page <= 1}
          onClick={() => onPageChange(Math.max(1, page - 1))}
        >
          ‹
        </button>
        <span className="page-indicator">Page {page} of {pageCount || '—'}</span>
        <button
          className="icon-button"
          type="button"
          aria-label="Next page"
          disabled={!pageCount || page >= pageCount}
          onClick={() => onPageChange(Math.min(pageCount, page + 1))}
        >
          ›
        </button>
      </div>
      <div className="toolbar-separator" aria-hidden="true" />
      <div className="toolbar-group">
        <button
          className="icon-button"
          type="button"
          aria-label="Zoom out"
          disabled={scale <= 0.6}
          onClick={() => onScaleChange(clampScale(scale - 0.1))}
        >
          −
        </button>
        <span className="zoom-indicator">{Math.round(scale * 100)}%</span>
        <button
          className="icon-button"
          type="button"
          aria-label="Zoom in"
          disabled={scale >= 2}
          onClick={() => onScaleChange(clampScale(scale + 0.1))}
        >
          +
        </button>
        <button className="fit-button" type="button" aria-label="Fit to width" onClick={onFitWidth}>
          Fit width
        </button>
      </div>
    </div>
  )
}
