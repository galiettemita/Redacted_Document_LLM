import { useMemo, useState } from 'react'
import { SAMPLE_DOCUMENT } from './data/sampleDocument'
import { UploadScreen } from './features/upload/UploadScreen'
import { DocumentWorkspace } from './features/workspace/DocumentWorkspace'
import { useLocalPdf } from './hooks/useLocalPdf'
import { useBackendDocument } from './hooks/useBackendDocument'
import type { DocumentDescriptor } from './types'

export function App() {
  const localPdf = useLocalPdf()
  const [document, setDocument] = useState<DocumentDescriptor | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
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

  if (!connectedDocument) {
    return <UploadScreen onFileAccepted={openUploadedFile} />
  }

  return (
    <DocumentWorkspace
      document={connectedDocument}
      backendDocument={backendDocument}
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
      onViewSample={async () => {
        try {
          await backendDocument.remove()
        } catch {
          return
        }
        localPdf.clear()
        setUploadedFile(null)
        setDocument(SAMPLE_DOCUMENT)
      }}
    />
  )
}
