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
  const uploadPromiseRef = useRef<Promise<BackendDocumentRecord> | null>(null)
  const removingRef = useRef(false)
  const removalPromiseRef = useRef<Promise<void> | null>(null)

  useEffect(() => {
    if (!file) {
      recordRef.current = null
      uploadPromiseRef.current = null
      removingRef.current = false
      setSnapshot(IDLE_SNAPSHOT)
      return
    }

    const controller = new AbortController()
    let active = true
    uploadControllerRef.current = controller
    removingRef.current = false
    recordRef.current = null
    setSnapshot({ status: 'uploading', record: null, error: null })

    const uploadPromise = api.uploadDocument(file, controller.signal)
    uploadPromiseRef.current = uploadPromise
    void uploadPromise.then(
      (record) => {
        if (uploadPromiseRef.current === uploadPromise) uploadPromiseRef.current = null
        if (!active) {
          void api.deleteDocument(record.id).catch(() => undefined)
          return
        }
        recordRef.current = record
        if (!removingRef.current) {
          setSnapshot({ status: 'ready', record, error: null })
        }
      },
      (error: unknown) => {
        if (uploadPromiseRef.current === uploadPromise) uploadPromiseRef.current = null
        if (!active || removingRef.current || isAbortError(error)) return
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

  const remove = useCallback(() => {
    if (removalPromiseRef.current) return removalPromiseRef.current

    removingRef.current = true
    const pendingUpload = uploadPromiseRef.current
    let record = recordRef.current
    if (!record && !pendingUpload) {
      removingRef.current = false
      setSnapshot(IDLE_SNAPSHOT)
      return Promise.resolve()
    }
    setSnapshot({ status: 'deleting', record, error: null })

    const operation = (async () => {
      if (!record && pendingUpload) {
        try {
          record = await pendingUpload
        } catch {
          setSnapshot(IDLE_SNAPSHOT)
          return
        }
      }
      if (!record) {
        setSnapshot(IDLE_SNAPSHOT)
        return
      }

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
    })().finally(() => {
      removingRef.current = false
      removalPromiseRef.current = null
    })
    removalPromiseRef.current = operation
    return operation
  }, [api])

  return { ...snapshot, remove }
}

function isAbortError(error: unknown): boolean {
  return typeof error === 'object'
    && error !== null
    && 'name' in error
    && error.name === 'AbortError'
}
