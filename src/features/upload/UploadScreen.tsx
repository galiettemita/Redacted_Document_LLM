import { useRef, useState, type DragEvent } from 'react'
import { validatePdfFile, type PdfValidationResult } from './fileValidation'

interface UploadScreenProps {
  onFileAccepted: (file: File) => void
  validate?: (file: File) => PdfValidationResult
}

export function UploadScreen({ onFileAccepted, validate = validatePdfFile }: UploadScreenProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function handleFile(file?: File) {
    if (!file) return
    setError(null)
    try {
      const result = validate(file)
      if (!result.ok) {
        setError(result.message)
        return
      }
      onFileAccepted(file)
    } catch {
      setError('We couldn’t check that PDF. Try choosing it again.')
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setIsDragging(false)
    handleFile(event.dataTransfer.files[0])
  }

  return (
    <main className="upload-shell">
      <section className="upload-panel" aria-labelledby="upload-heading">
        <p className="brand">Redacted Document LLM</p>
        <div className="upload-copy">
          <h1 id="upload-heading">Explore what’s behind the redactions</h1>
          <p>Upload a redacted PDF to explore AI predictions for its hidden passages.</p>
        </div>
        <div
          className="drop-zone"
          data-drag-active={isDragging}
          data-testid="pdf-drop-zone"
          onDragEnter={(event) => {
            event.preventDefault()
            setIsDragging(true)
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setIsDragging(false)
          }}
          onDrop={handleDrop}
        >
          <div className="document-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <p className="drop-title">{isDragging ? 'Release to open this PDF' : 'Drop your redacted document here'}</p>
          <p className="drop-support">PDF files only · stored temporarily by this local app</p>
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept="application/pdf,.pdf"
            aria-label="Choose PDF"
            onChange={(event) => handleFile(event.target.files?.[0])}
          />
          <button
            className="primary-button"
            type="button"
            onClick={() => inputRef.current?.click()}
          >
            Choose PDF
          </button>
          {error ? <p className="upload-error" role="alert">{error}</p> : null}
        </div>
        <p className="demo-note"><span aria-hidden="true" />Demo — predictions are simulated</p>
      </section>
    </main>
  )
}
