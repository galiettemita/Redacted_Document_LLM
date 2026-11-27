import { useEffect, useMemo, useRef, useState } from 'react'
import { FictionalDemoWorkspace } from './features/demo/FictionalDemoWorkspace'
import { UploadScreen } from './features/upload/UploadScreen'
import { DocumentWorkspace } from './features/workspace/DocumentWorkspace'
import { useLocalPdf } from './hooks/useLocalPdf'
import { useBackendDocument } from './hooks/useBackendDocument'
import { BackendApiError, backendApi } from './services/backendApi'
import type { DocumentDescriptor } from './types'

const ACTIVE_DOCUMENT_KEY = 'redacted-document-llm.active-document-v1'

export function App() {
  const localPdf = useLocalPdf()
  const [document, setDocument] = useState<DocumentDescriptor | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [restoreDocumentId, setRestoreDocumentId] = useState<string | null>(
    readActiveDocumentId,
  )
  const [demoOpen, setDemoOpen] = useState(false)
  const demoOpener = useRef<HTMLElement | null>(null)
  const wasDemoOpen = useRef(false)
  const backendDocument = useBackendDocument(
    uploadedFile,
    backendApi,
    uploadedFile ? null : restoreDocumentId,
  )

  const connectedDocument = useMemo(() => {
    if (!document || document.mode !== 'uploaded') return document
    return {
      ...document,
      backendDocumentId: backendDocument.record?.id,
      pageCount: backendDocument.record?.page_count,
    }
  }, [backendDocument.record, document])

  function openUploadedFile(file: File) {
    clearActiveDocumentId()
    setRestoreDocumentId(null)
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

  useEffect(() => {
    if (backendDocument.record) {
      writeActiveDocumentId(backendDocument.record.id)
    }
  }, [backendDocument.record])

  useEffect(() => {
    if (
      document
      || uploadedFile
      || !restoreDocumentId
      || backendDocument.record?.id !== restoreDocumentId
    ) return
    const controller = new AbortController()
    void backendApi.getDocumentContent(restoreDocumentId, controller.signal)
      .then((content) => {
        if (controller.signal.aborted || !backendDocument.record) return
        const file = new File(
          [content],
          backendDocument.record.filename,
          { type: backendDocument.record.media_type },
        )
        const source = localPdf.openValidatedFile(file)
        setDocument({
          id: `uploaded-${backendDocument.record.id}`,
          filename: backendDocument.record.filename,
          source,
          mode: 'uploaded',
        })
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return
        if (error instanceof BackendApiError && error.status === 404) {
          clearActiveDocumentId()
          setRestoreDocumentId(null)
        }
      })
    return () => controller.abort()
  }, [
    backendDocument.record,
    document,
    localPdf.openValidatedFile,
    restoreDocumentId,
    uploadedFile,
  ])

  useEffect(() => {
    if (
      restoreDocumentId
      && !uploadedFile
      && backendDocument.status === 'failed'
    ) {
      clearActiveDocumentId()
      setRestoreDocumentId(null)
    }
  }, [backendDocument.status, restoreDocumentId, uploadedFile])

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
        clearActiveDocumentId()
        setRestoreDocumentId(null)
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
            returnLabel={connectedDocument ? 'Return to your document' : 'Return to upload'}
          />
        </div>
      ) : null}
    </>
  )
}

function readActiveDocumentId(): string | null {
  try {
    return globalThis.localStorage?.getItem(ACTIVE_DOCUMENT_KEY) ?? null
  } catch {
    return null
  }
}

function writeActiveDocumentId(documentId: string): void {
  try {
    globalThis.localStorage?.setItem(ACTIVE_DOCUMENT_KEY, documentId)
  } catch {
    // Browser storage is optional; the live workspace remains usable without it.
  }
}

function clearActiveDocumentId(): void {
  try {
    globalThis.localStorage?.removeItem(ACTIVE_DOCUMENT_KEY)
  } catch {
    // Browser storage is optional; backend deletion remains authoritative.
  }
}
