import { useState } from 'react'
import { SAMPLE_DOCUMENT } from './data/sampleDocument'
import { UploadScreen } from './features/upload/UploadScreen'
import { DocumentWorkspace } from './features/workspace/DocumentWorkspace'
import { useLocalPdf } from './hooks/useLocalPdf'
import type { DocumentDescriptor } from './types'

export function App() {
  const localPdf = useLocalPdf()
  const [document, setDocument] = useState<DocumentDescriptor | null>(null)

  function openUploadedFile(file: File) {
    const source = localPdf.openValidatedFile(file)
    setDocument({
      id: `uploaded-${file.name}-${file.lastModified}`,
      filename: file.name,
      source,
      mode: 'uploaded',
    })
  }

  if (!document) {
    return <UploadScreen onFileAccepted={openUploadedFile} />
  }

  return (
    <DocumentWorkspace
      document={document}
      onReplace={() => {
        localPdf.clear()
        setDocument(null)
      }}
      onViewSample={() => {
        localPdf.clear()
        setDocument(SAMPLE_DOCUMENT)
      }}
    />
  )
}
