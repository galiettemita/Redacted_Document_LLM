import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { SAMPLE_PASSAGES, SAMPLE_PREDICTIONS } from '../../data/sampleDocument'
import type { BackendDocumentState } from '../../hooks/useBackendDocument'
import { usePredictionSimulation } from '../../hooks/usePredictionSimulation'
import {
  BackendApiError,
  backendApi as defaultBackendApi,
  type AnalysisJobRecord,
  type BackendApi,
  type BackendPassageRecord,
  type PassageDetectionResult,
  type VisibleContextSummary,
} from '../../services/backendApi'
import { createMockPredictionService } from '../../services/mockPredictionService'
import type { DocumentDescriptor, Passage, PassageRect } from '../../types'
import { PredictionSidebar } from '../predictions/PredictionSidebar'
import { PdfViewer } from '../viewer/PdfViewer'
import { getEffectivePassageId } from './selectionState'
import { initialWorkspaceState, workspaceReducer } from './workspaceReducer'

interface DocumentWorkspaceProps {
  document: DocumentDescriptor
  onReplace: () => void
  onViewSample: () => void
  backendDocument?: BackendDocumentState
  backendApi?: BackendApi
}

type PassageSaveStatus = 'idle' | 'saving' | 'saved' | 'failed'
type PassageDetectionStatus = 'idle' | 'detecting' | 'ready' | 'failed'
type ContextPreparationStatus = 'idle' | 'preparing' | 'ready' | 'failed'
type AnalysisRequestStatus = 'idle' | 'submitting' | 'received' | 'failed'

interface DetectionTask {
  documentId: string
  promise: Promise<PassageDetectionResult>
}

export function DocumentWorkspace({
  document,
  onReplace,
  onViewSample,
  backendDocument,
  backendApi = defaultBackendApi,
}: DocumentWorkspaceProps) {
  const [state, dispatch] = useReducer(workspaceReducer, initialWorkspaceState)
  const [passageSaveStatus, setPassageSaveStatus] = useState<PassageSaveStatus>('idle')
  const [passageError, setPassageError] = useState<string | null>(null)
  const [detectionStatus, setDetectionStatus] = useState<PassageDetectionStatus>('idle')
  const [detectionResult, setDetectionResult] = useState<PassageDetectionResult | null>(null)
  const [detectionError, setDetectionError] = useState<string | null>(null)
  const [contextStatus, setContextStatus] = useState<ContextPreparationStatus>('idle')
  const [contextSummary, setContextSummary] = useState<VisibleContextSummary | null>(null)
  const [contextError, setContextError] = useState<string | null>(null)
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisRequestStatus>('idle')
  const [analysisJob, setAnalysisJob] = useState<AnalysisJobRecord | null>(null)
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const passageTargets = useRef(new Map<string, HTMLElement>())
  const detectionTask = useRef<DetectionTask | null>(null)
  const passageRequest = useRef<AbortController | null>(null)
  const contextRequest = useRef<AbortController | null>(null)
  const analysisRequest = useRef<AbortController | null>(null)
  const analysisIdempotencyKey = useRef<string | null>(null)
  const sampleService = useMemo(() => createMockPredictionService({
    predictionsByDocument: { [document.id]: document.mode === 'sample' ? SAMPLE_PREDICTIONS : [] },
    delayMs: 650,
  }), [document.id, document.mode])
  const simulation = usePredictionSimulation(document.mode === 'sample' ? document : null, sampleService)

  useEffect(() => {
    dispatch({ type: 'reset' })
    setPassageSaveStatus('idle')
    setPassageError(null)
    setDetectionStatus('idle')
    setDetectionResult(null)
    setDetectionError(null)
    setContextStatus('idle')
    setContextSummary(null)
    setContextError(null)
    setAnalysisStatus('idle')
    setAnalysisJob(null)
    setAnalysisError(null)
    passageRequest.current?.abort()
    passageRequest.current = null
    contextRequest.current?.abort()
    contextRequest.current = null
    analysisRequest.current?.abort()
    analysisRequest.current = null
    analysisIdempotencyKey.current = null
  }, [document.id])

  useEffect(() => () => {
    passageRequest.current?.abort()
    contextRequest.current?.abort()
    analysisRequest.current?.abort()
  }, [])

  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') dispatch({ type: 'dismiss' })
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [])

  useEffect(() => {
    if (!state.pinnedId) return
    const target = passageTargets.current.get(state.pinnedId)
    target?.scrollIntoView?.({ block: 'center', behavior: 'smooth' })
    target?.focus({ preventScroll: true })
  }, [state.page, state.pinnedId])

  const passages = document.mode === 'sample' ? SAMPLE_PASSAGES : state.passages
  const predictions = document.mode === 'sample' ? simulation.predictions : []
  const phase = document.mode === 'sample'
    ? simulation.phase
    : analysisJob?.status === 'queued' || analysisJob?.status === 'running'
      ? 'predicting' as const
      : 'idle' as const
  const activePassageId = getEffectivePassageId(state.hoveredId, state.pinnedId)
  const backendReady = document.mode === 'uploaded'
    && backendDocument?.status === 'ready'
    && Boolean(document.backendDocumentId)
  const deletionInProgress = backendDocument?.status === 'deleting'
  const markingUnavailable = !backendReady
    || detectionStatus === 'detecting'
    || passageSaveStatus === 'saving'
  const selectedPassage = state.passages.find((passage) => passage.id === state.pinnedId)
    ?? state.passages[0]
    ?? null
  const stablePassageId = selectedPassage?.id.startsWith('pending-')
    ? null
    : selectedPassage?.id ?? null
  const contextPreparationUnavailable = !backendReady
    || !stablePassageId
    || passageSaveStatus === 'saving'
    || contextStatus === 'preparing'
  const analysisUnavailable = !backendReady
    || !stablePassageId
    || contextStatus !== 'ready'
    || !contextSummary
    || analysisStatus === 'submitting'

  useEffect(() => {
    if (!backendReady || !document.backendDocumentId) return
    const documentId = document.backendDocumentId
    const task = detectionTask.current?.documentId === documentId
      ? detectionTask.current
      : {
          documentId,
          promise: backendApi.detectDocumentPassages(
            documentId,
            new AbortController().signal,
          ),
        }
    detectionTask.current = task
    let active = true
    setDetectionStatus('detecting')
    setDetectionResult(null)
    setDetectionError(null)

    void task.promise
      .then((result) => {
        if (!active) return
        if (result.document_id !== documentId) {
          throw new Error('The backend returned passages for a different document.')
        }
        dispatch({ type: 'set-passages', passages: result.passages.map(toPassage) })
        setDetectionResult(result)
        setDetectionStatus('ready')
      })
      .catch((error: unknown) => {
        if (!active || isAbortError(error)) return
        dispatch({ type: 'set-passages', passages: [] })
        setDetectionStatus('failed')
        setDetectionError(errorMessage(error, 'Automatic passage detection could not finish.'))
      })
    return () => {
      active = false
    }
  }, [backendApi, backendReady, document.backendDocumentId])

  useEffect(() => {
    contextRequest.current?.abort()
    contextRequest.current = null
    setContextStatus('idle')
    setContextSummary(null)
    setContextError(null)
    analysisRequest.current?.abort()
    analysisRequest.current = null
    analysisIdempotencyKey.current = null
    setAnalysisStatus('idle')
    setAnalysisJob(null)
    setAnalysisError(null)
  }, [stablePassageId])

  async function handleMarkPassage(page: number, rect: PassageRect) {
    if (!backendReady || !document.backendDocumentId) return
    passageRequest.current?.abort()
    const controller = new AbortController()
    passageRequest.current = controller
    const existingPassage = state.passages.find(
      (passage) => passage.id === state.markingPassageId,
    ) ?? null
    const passage: Passage = {
      id: existingPassage?.id ?? `pending-${createIdempotencyKey()}`,
      label: existingPassage?.label ?? nextPassageLabel(state.passages),
      page,
      rects: [rect],
    }
    dispatch({ type: 'upsert-passage', passage })
    setPassageSaveStatus('saving')
    setPassageError(null)
    contextRequest.current?.abort()
    contextRequest.current = null
    setContextStatus('idle')
    setContextSummary(null)
    setContextError(null)
    analysisRequest.current?.abort()
    analysisRequest.current = null
    analysisIdempotencyKey.current = null
    setAnalysisStatus('idle')
    setAnalysisJob(null)
    setAnalysisError(null)

    try {
      let saved: BackendPassageRecord
      if (existingPassage && !existingPassage.id.startsWith('pending-')) {
        saved = await backendApi.updatePassage(
          existingPassage.id,
          { page, rects: [rect] },
          { signal: controller.signal },
        )
      } else {
        saved = await backendApi.createPassage(
          document.backendDocumentId,
          { label: passage.label, page, rects: [rect] },
          { signal: controller.signal, idempotencyKey: createIdempotencyKey() },
        )
      }
      dispatch({
        type: 'replace-passage',
        previousId: passage.id,
        passage: toPassage(saved),
      })
      setPassageSaveStatus('saved')
    } catch (error) {
      if (isAbortError(error)) return
      setPassageSaveStatus('failed')
      setPassageError(
        error instanceof BackendApiError || error instanceof Error
          ? error.message
          : 'The passage could not be saved.',
      )
    } finally {
      if (passageRequest.current === controller) passageRequest.current = null
    }
  }

  async function handlePrepareVisibleContext() {
    if (contextPreparationUnavailable || !stablePassageId) return
    contextRequest.current?.abort()
    const controller = new AbortController()
    contextRequest.current = controller
    setContextStatus('preparing')
    setContextSummary(null)
    setContextError(null)
    analysisRequest.current?.abort()
    analysisRequest.current = null
    analysisIdempotencyKey.current = null
    setAnalysisStatus('idle')
    setAnalysisJob(null)
    setAnalysisError(null)

    try {
      const prepared = await backendApi.prepareVisibleContext(
        stablePassageId,
        controller.signal,
      )
      if (prepared.passage_id !== stablePassageId) {
        throw new Error('The backend returned context for a different passage.')
      }
      setContextSummary(prepared)
      setContextStatus('ready')
    } catch (error) {
      if (isAbortError(error)) return
      setContextStatus('failed')
      setContextError(errorMessage(error, 'Visible context could not be prepared.'))
    } finally {
      if (contextRequest.current === controller) contextRequest.current = null
    }
  }

  async function handlePredictPassage() {
    if (analysisUnavailable || !stablePassageId || !contextSummary) return
    analysisRequest.current?.abort()
    const controller = new AbortController()
    analysisRequest.current = controller
    const idempotencyKey = analysisIdempotencyKey.current ?? createIdempotencyKey()
    analysisIdempotencyKey.current = idempotencyKey
    setAnalysisStatus('submitting')
    setAnalysisJob(null)
    setAnalysisError(null)

    try {
      const job = await backendApi.schedulePassageAnalysis(
        stablePassageId,
        contextSummary.passage_revision,
        { signal: controller.signal, idempotencyKey },
      )
      if (job.passage_id !== stablePassageId) {
        throw new Error('The backend returned analysis for a different passage.')
      }
      setAnalysisJob(job)
      setAnalysisStatus('received')
      analysisIdempotencyKey.current = null
    } catch (error) {
      if (isAbortError(error)) return
      setAnalysisStatus('failed')
      setAnalysisError(errorMessage(error, 'Prediction readiness could not be checked.'))
    } finally {
      if (analysisRequest.current === controller) analysisRequest.current = null
    }
  }

  function handleSelectPassage(passageId: string) {
    const passage = passages.find((item) => item.id === passageId)
    if (!passage) return
    dispatch({ type: 'select-passage', passageId, page: passage.page })
  }

  return (
    <main className="workspace-shell">
      <header className="workspace-header">
        <div className="workspace-identity">
          <span className="workspace-brand">Redacted Document LLM</span>
          <span className="header-divider" aria-hidden="true" />
          <span className="filename" title={document.filename}>{document.filename}</span>
          {document.label ? <span className="fictional-badge">{document.label}</span> : null}
        </div>
        <button
          className="secondary-button"
          type="button"
          aria-label={deletionInProgress ? 'Removing document…' : 'Replace document'}
          disabled={deletionInProgress}
          onClick={onReplace}
        >
          {deletionInProgress ? 'Removing document…' : 'Replace document'}
        </button>
      </header>

      <nav className="mobile-view-switch" aria-label="Workspace view">
        <button
          type="button"
          aria-pressed={state.activeView === 'document'}
          onClick={() => dispatch({ type: 'set-view', view: 'document' })}
        >
          Document
        </button>
        <button
          type="button"
          aria-pressed={state.activeView === 'predictions'}
          onClick={() => dispatch({ type: 'set-view', view: 'predictions' })}
        >
          Predictions
          {predictions.length ? <span aria-hidden="true">{predictions.length}</span> : null}
        </button>
      </nav>

      <div className="workspace-grid">
        <div className="document-column" data-active={state.activeView === 'document'}>
          {document.mode === 'uploaded' ? (
            <div className="marking-toolbar">
              <p>
                <strong>{markingToolbarTitle(backendDocument, state.markingEnabled, passageSaveStatus, detectionStatus)}</strong>
                {markingToolbarMessage(backendDocument, state.markingEnabled, passageSaveStatus, detectionStatus, Boolean(selectedPassage))}
              </p>
              <div className="marking-actions">
                {selectedPassage ? (
                  <button
                    type="button"
                    disabled={markingUnavailable}
                    onClick={() => dispatch({
                      type: 'start-marking',
                      passageId: selectedPassage.id,
                    })}
                  >
                    Correct selected area
                  </button>
                ) : null}
                <button
                  type="button"
                  disabled={markingUnavailable}
                  onClick={() => dispatch({ type: 'start-marking', passageId: null })}
                >
                  {selectedPassage ? 'Add missed passage' : 'Mark a passage'}
                </button>
              </div>
            </div>
          ) : null}
          <PdfViewer
            source={document.source}
            filename={document.filename}
            passages={passages}
            page={state.page}
            onPageChange={(page) => dispatch({ type: 'set-page', page })}
            markingEnabled={state.markingEnabled}
            onMarkPassage={handleMarkPassage}
            hoveredId={state.hoveredId}
            pinnedId={state.pinnedId}
            onHoverPassage={(passageId) => dispatch({ type: 'hover', passageId })}
            onPinPassage={(passageId) => dispatch({ type: 'pin', passageId })}
            onDismissSelection={() => dispatch({ type: 'dismiss' })}
            registerPassageTarget={(passageId, element) => {
              if (element) passageTargets.current.set(passageId, element)
              else passageTargets.current.delete(passageId)
            }}
          />
        </div>
        <div className="predictions-column" data-active={state.activeView === 'predictions'}>
          <PredictionSidebar
            passages={passages}
            predictions={predictions}
            phase={phase}
            error={document.mode === 'sample'
              ? simulation.error
              : passageError
                ?? detectionError
                ?? contextError
                ?? analysisError
                ?? backendDocumentError(backendDocument)}
            activePassageId={activePassageId}
            onSelectPassage={handleSelectPassage}
            onDismissSelection={() => dispatch({ type: 'dismiss' })}
            missingPredictionMessage={document.mode === 'uploaded'
              ? passageSaveStatus === 'saving'
                ? 'Saving this passage to the backend…'
                : detectionStatus === 'detecting'
                  ? 'Scanning every page for hidden passages…'
                  : detectionStatus === 'failed'
                    ? 'Automatic detection needs review before predictions can begin.'
                : contextStatus === 'ready'
                  ? analysisJob?.status === 'blocked'
                    ? 'Prediction readiness was checked. The required real services are listed below.'
                    : 'Context query is ready. Use Predict passage to check every real dependency.'
                  : passages.length
                    ? 'No prediction has been requested. Prepare the selected passage’s visible-only context first.'
                    : 'No hidden passages are available for prediction.'
              : undefined}
          >
            {document.mode === 'uploaded' ? (
              <div className="uploaded-document-note">
                <h3>{backendNoteTitle(
                  backendDocument,
                  passageSaveStatus,
                  contextStatus,
                  detectionStatus,
                  passages.length,
                )}</h3>
                <p>{backendNoteMessage(
                  backendDocument,
                  passageSaveStatus,
                  contextStatus,
                  detectionStatus,
                  detectionResult,
                  passages.length,
                )}</p>
                <div className="visible-context-controls">
                  <p>
                    Preparing reads only visible PDF content, detects its visible date, and
                    builds the passage-linked query that retrieval will use.
                  </p>
                  <button
                    className="prepare-context-button"
                    type="button"
                    disabled={contextPreparationUnavailable}
                    onClick={() => void handlePrepareVisibleContext()}
                  >
                    {contextStatus === 'preparing'
                      ? 'Preparing visible context…'
                      : 'Prepare visible context'}
                  </button>
                  {contextSummary ? (
                    <section className="visible-context-summary" aria-live="polite">
                      <h4>Visible-only context ready</h4>
                      <p>
                        <span>{contextSummary.document_date}</span>
                        <span>{contextSummary.length_cue}</span>
                        <span>{contextSummary.estimated_words} estimated words</span>
                        <span>{contextSummary.visible_word_count} visible words</span>
                        <span>{contextSummary.confirmed_mask_count} confirmed masks</span>
                        <span>{contextSummary.context_query_term_count} visible query terms</span>
                      </p>
                      <p><strong>Context query ready</strong></p>
                      <p>The next action checks the real corpus, worker, and model gates.</p>
                      <button
                        className="prepare-context-button"
                        type="button"
                        disabled={analysisUnavailable}
                        onClick={() => void handlePredictPassage()}
                      >
                        {analysisStatus === 'submitting'
                          ? 'Checking prediction readiness…'
                          : 'Predict passage'}
                      </button>
                      {analysisJob?.status === 'blocked' ? (
                        <section className="analysis-readiness" aria-live="polite">
                          <h4>Prediction is not ready</h4>
                          <p>No retrieval or model call was made.</p>
                          <ul>
                            {analysisJob.blockers.map((blocker) => (
                              <li key={blocker.component}>
                                <strong>{capitalize(blocker.component)}:</strong>{' '}
                                {blocker.detail}
                              </li>
                            ))}
                          </ul>
                        </section>
                      ) : null}
                    </section>
                  ) : null}
                </div>
                <button
                  className="text-button"
                  type="button"
                  disabled={deletionInProgress}
                  onClick={onViewSample}
                >
                  View interactive sample
                </button>
              </div>
            ) : null}
          </PredictionSidebar>
        </div>
      </div>
    </main>
  )
}

function toPassage(record: BackendPassageRecord): Passage {
  return {
    id: record.id,
    label: record.label,
    page: record.page_number,
    rects: record.rects,
  }
}

function createIdempotencyKey(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') return globalThis.crypto.randomUUID()
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function isAbortError(error: unknown): boolean {
  return typeof error === 'object'
    && error !== null
    && 'name' in error
    && error.name === 'AbortError'
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof BackendApiError || error instanceof Error
    ? error.message
    : fallback
}

function capitalize(value: string): string {
  return value ? `${value[0].toUpperCase()}${value.slice(1)}` : value
}

function markingToolbarTitle(
  backendDocument: BackendDocumentState | undefined,
  markingEnabled: boolean,
  passageSaveStatus: PassageSaveStatus,
  detectionStatus: PassageDetectionStatus,
): string {
  if (backendDocument?.status === 'uploading') return 'Uploading PDF'
  if (backendDocument?.status === 'deleting') return 'Removing PDF'
  if (backendDocument?.status === 'unavailable') return 'Backend unavailable'
  if (backendDocument?.status === 'failed') return 'Upload not saved'
  if (detectionStatus === 'detecting') return 'Scanning every page'
  if (passageSaveStatus === 'saving') return 'Saving passage'
  return markingEnabled ? 'Draw the passage area' : 'Review passage detection'
}

function markingToolbarMessage(
  backendDocument: BackendDocumentState | undefined,
  markingEnabled: boolean,
  passageSaveStatus: PassageSaveStatus,
  detectionStatus: PassageDetectionStatus,
  hasSelectedPassage: boolean,
): string {
  if (backendDocument?.status === 'uploading') {
    return ' Uploading to the loopback-only local backend. Marking becomes available when persistence finishes.'
  }
  if (backendDocument?.status === 'deleting') {
    return ' Waiting for confirmed backend deletion before leaving this document.'
  }
  if (backendDocument?.status === 'unavailable') {
    return ' The local PDF remains visible, but passage persistence needs the local backend.'
  }
  if (backendDocument?.status === 'failed') {
    return ` ${backendDocument.error ?? 'The PDF could not be persisted.'}`
  }
  if (detectionStatus === 'detecting') {
    return ' The backend is finding hidden passages automatically; manual correction will be available when it finishes.'
  }
  if (passageSaveStatus === 'saving') return ' Keeping this rectangle linked to its stable backend passage.'
  return markingEnabled
    ? ' Drag one rectangle over the correct hidden area.'
    : hasSelectedPassage
      ? ' Correct the selected rectangle, or add a passage only if the automatic scan missed it.'
      : ' Use this only if the automatic scan missed a hidden passage.'
}

function backendNoteTitle(
  backendDocument: BackendDocumentState | undefined,
  passageSaveStatus: PassageSaveStatus,
  contextStatus: ContextPreparationStatus,
  detectionStatus: PassageDetectionStatus,
  passageCount: number,
): string {
  if (backendDocument?.status === 'uploading') return 'Backend upload in progress'
  if (backendDocument?.status === 'deleting') return 'Removing stored PDF'
  if (backendDocument?.status === 'unavailable') return 'Local backend unavailable'
  if (backendDocument?.status === 'failed') return 'PDF upload failed'
  if (passageSaveStatus === 'saving') return 'Saving Passage 1'
  if (contextStatus === 'ready') return 'Selected passage context is ready'
  if (passageSaveStatus === 'saved') return 'Selected passage is saved'
  if (passageSaveStatus === 'failed') return 'Passage save failed'
  if (detectionStatus === 'detecting') return 'Scanning every page'
  if (detectionStatus === 'failed') return 'Automatic detection needs review'
  if (detectionStatus === 'ready' && passageCount > 0) {
    return `${passageCount} ${passageCount === 1 ? 'passage' : 'passages'} detected`
  }
  if (detectionStatus === 'ready') return 'No redactions detected'
  return 'PDF upload complete'
}

function backendNoteMessage(
  backendDocument: BackendDocumentState | undefined,
  passageSaveStatus: PassageSaveStatus,
  contextStatus: ContextPreparationStatus,
  detectionStatus: PassageDetectionStatus,
  detectionResult: PassageDetectionResult | null,
  passageCount: number,
): string {
  if (backendDocument?.status === 'uploading') {
    return 'Uploading to the loopback-only local backend while the browser displays its local copy.'
  }
  if (backendDocument?.status === 'deleting') {
    return 'The app will return to the upload screen after the backend confirms deletion.'
  }
  if (backendDocument?.status === 'unavailable') {
    return 'The PDF is displayed locally, but the backend copy and passage coordinates have not been saved.'
  }
  if (backendDocument?.status === 'failed') {
    return backendDocument.error ?? 'The backend rejected this PDF.'
  }
  if (contextStatus === 'ready') {
    return 'The backend prepared this passage and its search query from visible PDF content and geometry only. Use Predict passage to check the real execution gates.'
  }
  if (passageSaveStatus === 'saved') {
    return 'The selected passage is saved. Prepare its visible-only context below; the backend will detect the date from the PDF.'
  }
  if (passageSaveStatus === 'failed') {
    return 'The rectangle remains visible locally, but it was not saved. Try marking the passage again.'
  }
  if (detectionStatus === 'detecting') {
    return 'The PDF is stored temporarily by the local backend. Every page is being scanned for hidden passages.'
  }
  if (detectionStatus === 'failed') {
    return 'No passage was silently guessed. Review the detection error, then use manual correction only if the PDF needs it.'
  }
  if (detectionStatus === 'ready' && detectionResult) {
    return passageCount
      ? `All ${detectionResult.pages_scanned} pages were scanned automatically. Each detected area now has a stable passage record and PDF highlight; manual marking is only for corrections or a missed-area fallback.`
      : `All ${detectionResult.pages_scanned} pages were scanned automatically, but no reliable hidden passages were found. Manual marking remains available for a missed passage.`
  }
  return 'The PDF is stored temporarily by the local backend. Automatic all-page passage detection will begin now.'
}

function backendDocumentError(backendDocument: BackendDocumentState | undefined): string | null {
  if (backendDocument?.status !== 'failed' && backendDocument?.status !== 'unavailable') return null
  return backendDocument.error ?? 'The local backend is unavailable.'
}

function nextPassageLabel(passages: Passage[]): string {
  const numbers = passages.flatMap((passage) => {
    const match = /^Passage (\d+)$/.exec(passage.label)
    return match ? [Number(match[1])] : []
  })
  return `Passage ${Math.max(0, ...numbers) + 1}`
}
