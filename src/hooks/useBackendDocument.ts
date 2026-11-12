import { useCallback, useEffect, useRef, useState } from 'react'
import {
  BackendApiError,
  backendApi,
  type BackendApi,
  type BackendDocumentRecord,
} from '../services/backendApi'

export type BackendDocumentStatus =
  | 'idle'
  | 'uploading'
  | 'ready'
  | 'unavailable'
  | 'failed'
  | 'deleting'

export interface BackendDocumentState {
  status: BackendDocumentStatus
  record: BackendDocumentRecord | null
  error: string | null
  remove(): Promise<void>
}

interface BackendDocumentSnapshot {
  status: BackendDocumentStatus
  record: BackendDocumentRecord | null
  error: string | null
}

const IDLE_SNAPSHOT: BackendDocumentSnapshot = {
  status: 'idle',
  record: null,
  error: null,
}

export function useBackendDocument(
  file: File | null,
  api: BackendApi = backendApi,
): BackendDocumentState {
  const [snapshot, setSnapshot] = useState<BackendDocumentSnapshot>(() => (
    file ? { status: 'uploading', record: null, error: null } : IDLE_SNAPSHOT
  ))
  const recordRef = useRef<BackendDocumentRecord | null>(null)
  const uploadControllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!file) {
      recordRef.current = null
      setSnapshot(IDLE_SNAPSHOT)
      return
    }

    const controller = new AbortController()
    let active = true
    uploadControllerRef.current = controller
    recordRef.current = null
    setSnapshot({ status: 'uploading', record: null, error: null })

    void api.uploadDocument(file, controller.signal).then(
      (record) => {
        if (!active) {
          void api.deleteDocument(record.id).catch(() => undefined)
          return
        }
        recordRef.current = record
        setSnapshot({ status: 'ready', record, error: null })
      },
      (error: unknown) => {
        if (!active || isAbortError(error)) return
        const unavailable = error instanceof BackendApiError
          && (
            error.code === 'backend_unavailable'
            || error.status === 502
            || error.status === 503
            || error.status === 504
          )
        setSnapshot({
          status: unavailable ? 'unavailable' : 'failed',
          record: null,
          error: unavailable
            ? 'The local backend is unavailable. Start it and choose the PDF again.'
            : error instanceof Error
              ? error.message
              : 'The PDF could not be uploaded.',
        })
      },
    )

    return () => {
      active = false
      controller.abort()
      if (uploadControllerRef.current === controller) uploadControllerRef.current = null
    }
  }, [api, file])

  const remove = useCallback(async () => {
    uploadControllerRef.current?.abort()
    uploadControllerRef.current = null
    const record = recordRef.current
    if (!record) {
      setSnapshot(IDLE_SNAPSHOT)
      return
    }

    setSnapshot({ status: 'deleting', record, error: null })
    try {
      await api.deleteDocument(record.id)
      recordRef.current = null
      setSnapshot(IDLE_SNAPSHOT)
    } catch (error) {
      setSnapshot({
        status: 'failed',
        record,
        error: error instanceof Error ? error.message : 'The stored PDF could not be removed.',
      })
      throw error
    }
  }, [api])

  return { ...snapshot, remove }
}

function isAbortError(error: unknown): boolean {
  return typeof error === 'object'
    && error !== null
    && 'name' in error
    && error.name === 'AbortError'
}
