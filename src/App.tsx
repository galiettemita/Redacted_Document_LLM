import { useEffect, useMemo, useRef, useState } from 'react'
import { FictionalDemoWorkspace } from './features/demo/FictionalDemoWorkspace'
import { UploadScreen } from './features/upload/UploadScreen'
import { DocumentWorkspace } from './features/workspace/DocumentWorkspace'
import { useLocalPdf } from './hooks/useLocalPdf'
import { useBackendDocument } from './hooks/useBackendDocument'
import type { DocumentDescriptor } from './types'

export function App() {
  const localPdf = useLocalPdf()
  const [document, setDocument] = useState<DocumentDescriptor | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [demoOpen, setDemoOpen] = useState(false)
  const demoOpener = useRef<HTMLElement | null>(null)
  const wasDemoOpen = useRef(false)
  const backendDocument = useBackendDocument(uploadedFile)

  const connectedDocument = useMemo(() => {
    if (!document || document.mode !== 'uploaded') return document
    return {
      ...document,
      backendDocumentId: backendDocument.record?.id,
      pageCount: backendDocument.record?.page_count,
    }
  }, [backendDocument.record, document])

  function openUploadedFile(file: File) {
    const source = localPdf.openValidatedFile(file)
    setUploadedFile(file)
    setDocument({
      id: `uploaded-${file.name}-${file.lastModified}`,
      filename: file.name,
      source,
      mode: 'uploaded',
    })
  }

  function openDemo(opener: HTMLButtonElement) {
    demoOpener.current = opener
    setDemoOpen(true)
  }

  useEffect(() => {
    if (!demoOpen && wasDemoOpen.current) demoOpener.current?.focus()
    wasDemoOpen.current = demoOpen
  }, [demoOpen])

  const realScreen = connectedDocument ? (
    <DocumentWorkspace
      document={connectedDocument}
      backendDocument={backendDocument}
      isInteractive={!demoOpen}
      onViewDemo={openDemo}
      onReplace={async () => {
        try {
          await backendDocument.remove()
        } catch {
          return
        }
        localPdf.clear()
        setUploadedFile(null)
        setDocument(null)
      }}
    />
  ) : (
    <UploadScreen onFileAccepted={openUploadedFile} onViewDemo={openDemo} />
  )

  return (
    <>
      <div
        className="app-surface"
        aria-hidden={demoOpen ? 'true' : undefined}
        inert={demoOpen ? true : undefined}
      >
        {realScreen}
      </div>
      {demoOpen ? (
        <div className="demo-layer">
          <FictionalDemoWorkspace
            onClose={() => setDemoOpen(false)}
            returnLabel={connectedDocument ? 'Return to document' : 'Back to upload'}
          />
        </div>
      ) : null}
    </>
  )
}
