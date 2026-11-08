import { useCallback, useEffect, useRef, useState } from 'react'
import { validatePdfFile } from '../features/upload/fileValidation'

type LocalPdfStatus = 'idle' | 'loading' | 'ready' | 'error'

export function useLocalPdf() {
  const currentUrl = useRef<string | null>(null)
  const [objectUrl, setObjectUrl] = useState<string | null>(null)
  const [status, setStatus] = useState<LocalPdfStatus>('idle')
  const [error, setError] = useState<string | null>(null)

  const revokeCurrentUrl = useCallback(() => {
    if (currentUrl.current) {
      URL.revokeObjectURL(currentUrl.current)
      currentUrl.current = null
    }
  }, [])

  const clear = useCallback(() => {
    revokeCurrentUrl()
    setObjectUrl(null)
    setStatus('idle')
    setError(null)
  }, [revokeCurrentUrl])

  const openFile = useCallback(async (file: File) => {
    setStatus('loading')
    setError(null)
    const validation = await validatePdfFile(file)
    if (!validation.ok) {
      setStatus('error')
      setError(validation.message)
      return null
    }

    revokeCurrentUrl()
    const nextUrl = URL.createObjectURL(file)
    currentUrl.current = nextUrl
    setObjectUrl(nextUrl)
    setStatus('ready')
    return nextUrl
  }, [revokeCurrentUrl])

  useEffect(() => revokeCurrentUrl, [revokeCurrentUrl])

  return { openFile, objectUrl, status, error, clear }
}
