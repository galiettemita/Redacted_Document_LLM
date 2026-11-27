import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import type { BackendDocumentState } from '../../hooks/useBackendDocument'
import { usePredictionStream } from '../../hooks/usePredictionStream'
import {
  BackendApiError,
  backendApi as defaultBackendApi,
  type BackendApi,
  type BackendPassageRecord,
  type BackendReadiness,
  type DocumentAnalysisScheduleResult,
  type PassageDetectionResult,
  type VisibleContextSummary,
} from '../../services/backendApi'
import { createHttpPredictionService } from '../../services/httpPredictionService'
import type { PredictionService } from '../../services/predictionService'
import type {
  DocumentDescriptor,
  Passage,
  PassageRect,
  PredictionBasis,
} from '../../types'
import { PredictionSidebar } from '../predictions/PredictionSidebar'
import { PdfViewer } from '../viewer/PdfViewer'
import { getEffectivePassageId } from './selectionState'
import { initialWorkspaceState, workspaceReducer } from './workspaceReducer'

interface DocumentWorkspaceProps {
  document: DocumentDescriptor
  onReplace: () => void
  onViewDemo?: (opener: HTMLButtonElement) => void
  isInteractive?: boolean
  backendDocument?: BackendDocumentState
  backendApi?: BackendApi
  predictionService?: PredictionService
}

type PassageSaveStatus = 'idle' | 'saving' | 'saved' | 'failed'
type PassageDetectionStatus = 'idle' | 'detecting' | 'ready' | 'failed'
type ContextPreparationStatus = 'idle' | 'preparing' | 'ready' | 'failed'
type PassageRemovalStatus = 'idle' | 'confirming' | 'deleting'
type AnalysisScheduleStatus = 'idle' | 'scheduling' | 'running' | 'blocked'
type FullDocumentPredictionBasis = Extract<
  PredictionBasis,
  'best_available_foiarchive_v1' | 'paper_exact_v1'
>
const READINESS_REFRESH_INTERVAL_MS = 60_000

interface DetectionTask {
  documentId: string
  promise: Promise<PassageDetectionResult>
}

export function DocumentWorkspace({
  document,
  onReplace,
  onViewDemo,
  isInteractive = true,
  backendDocument,
  backendApi = defaultBackendApi,
  predictionService,
}: DocumentWorkspaceProps) {
  const [state, dispatch] = useReducer(workspaceReducer, initialWorkspaceState)
  const [passageSaveStatus, setPassageSaveStatus] = useState<PassageSaveStatus>('idle')
  const [passageError, setPassageError] = useState<string | null>(null)
  const [detectionStatus, setDetectionStatus] = useState<PassageDetectionStatus>('idle')
  const [detectionResult, setDetectionResult] = useState<PassageDetectionResult | null>(null)
  const [detectionError, setDetectionError] = useState<string | null>(null)
  const [contextStatus, setContextStatus] = useState<ContextPreparationStatus>('idle')
  const [contextSummaries, setContextSummaries] = useState<Record<string, VisibleContextSummary>>({})
  const [contextError, setContextError] = useState<string | null>(null)
  const [passageRemovalStatus, setPassageRemovalStatus] = useState<PassageRemovalStatus>('idle')
  const [passageRemovalId, setPassageRemovalId] = useState<string | null>(null)
  const [analysisScheduleStatus, setAnalysisScheduleStatus] = useState<AnalysisScheduleStatus>('idle')
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [fullDocumentPredictionBasis, setFullDocumentPredictionBasis] = useState<
    FullDocumentPredictionBasis | null
  >(null)
  const [predictionRunKey, setPredictionRunKey] = useState(0)
  const [predictionJobBases, setPredictionJobBases] = useState<
    Record<string, PredictionBasis>
  >({})
  const passageTargets = useRef(new Map<string, HTMLElement>())
  const passageRemovalTrigger = useRef<HTMLButtonElement | null>(null)
  const detectionTask = useRef<DetectionTask | null>(null)
  const passageRequest = useRef<AbortController | null>(null)
  const contextRequest = useRef<AbortController | null>(null)
  const passageRemovalRequest = useRef<AbortController | null>(null)
  const analysisRequest = useRef<AbortController | null>(null)
  const analysisRunId = useRef<string | null>(null)
  const basisRecoveryKey = useRef<string | null>(null)
  const uploadedPredictionService = useMemo(
    () => predictionService ?? createHttpPredictionService(),
    [predictionService],
  )
  const modelStream = usePredictionStream(
    predictionRunKey > 0 ? document : null,
    uploadedPredictionService,
    predictionRunKey,
    predictionJobBases,
  )
  const backendReady = backendDocument?.status === 'ready'
    && Boolean(document.backendDocumentId)

  useEffect(() => {
    dispatch({ type: 'reset' })
    setPassageSaveStatus('idle')
    setPassageError(null)
    setDetectionStatus('idle')
    setDetectionResult(null)
    setDetectionError(null)
    setContextStatus('idle')
    setContextSummaries({})
    setContextError(null)
    setPassageRemovalStatus('idle')
    setPassageRemovalId(null)
    setAnalysisScheduleStatus('idle')
    setAnalysisError(null)
    setFullDocumentPredictionBasis(null)
    setPredictionRunKey(0)
    setPredictionJobBases({})
    passageRequest.current?.abort()
    passageRequest.current = null
    contextRequest.current?.abort()
    contextRequest.current = null
    passageRemovalRequest.current?.abort()
    passageRemovalRequest.current = null
    analysisRequest.current?.abort()
    analysisRequest.current = null
    analysisRunId.current = null
    basisRecoveryKey.current = null
  }, [document.id])

  useEffect(() => () => {
    passageRequest.current?.abort()
    contextRequest.current?.abort()
    passageRemovalRequest.current?.abort()
    analysisRequest.current?.abort()
  }, [])

  useEffect(() => {
    setFullDocumentPredictionBasis(null)
    if (!backendReady) return
    let disposed = false
    let activeRequest: AbortController | null = null
    let refreshTimer: ReturnType<typeof setTimeout> | null = null

    const refreshReadiness = () => {
      if (disposed) return
      if (refreshTimer !== null) {
        clearTimeout(refreshTimer)
        refreshTimer = null
      }
      activeRequest?.abort()
      const controller = new AbortController()
      activeRequest = controller
      void backendApi.getReadiness(controller.signal)
        .then((readiness) => {
          if (!controller.signal.aborted && !disposed) {
            setFullDocumentPredictionBasis(readyFullDocumentBasis(readiness))
          }
        })
        .catch((error: unknown) => {
          if (!isAbortError(error) && !controller.signal.aborted && !disposed) {
            setFullDocumentPredictionBasis(null)
          }
        })
        .finally(() => {
          if (disposed || activeRequest !== controller) return
          activeRequest = null
          refreshTimer = setTimeout(
            refreshReadiness,
            READINESS_REFRESH_INTERVAL_MS,
          )
        })
    }

    refreshReadiness()
    window.addEventListener('focus', refreshReadiness)
    return () => {
      disposed = true
      window.removeEventListener('focus', refreshReadiness)
      activeRequest?.abort()
      if (refreshTimer !== null) clearTimeout(refreshTimer)
    }
  }, [backendApi, backendReady, document.id])

  useEffect(() => {
    if (!isInteractive) return
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') dispatch({ type: 'dismiss' })
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [isInteractive])

  useEffect(() => {
    if (!state.pinnedId) return
    const target = passageTargets.current.get(state.pinnedId)
    target?.scrollIntoView?.({ block: 'center', behavior: 'smooth' })
    target?.focus({ preventScroll: true })
  }, [state.page, state.pinnedId])

  const passages = state.passages
  const predictions = modelStream.predictions
  const phase = modelStream.phase
  const activePassageId = getEffectivePassageId(state.hoveredId, state.pinnedId)
  const deletionInProgress = backendDocument?.status === 'deleting'
  const predictionWorkActive = analysisScheduleStatus === 'scheduling'
    || analysisScheduleStatus === 'running'
  const markingUnavailable = !backendReady
    || detectionStatus === 'detecting'
    || passageSaveStatus === 'saving'
    || passageRemovalStatus !== 'idle'
    || predictionWorkActive
  const selectedPassage = state.passages.find((passage) => passage.id === state.pinnedId)
    ?? state.passages[0]
    ?? null
  const passageRemovalTarget = state.passages.find(
    (passage) => passage.id === passageRemovalId,
  ) ?? null

  const prepareAllPassageContexts = useCallback(async (
    documentId: string,
    expectedPassageIds?: string[],
  ) => {
    basisRecoveryKey.current = null
    analysisRequest.current?.abort()
    analysisRequest.current = null
    analysisRunId.current = null
    setAnalysisScheduleStatus('idle')
    setAnalysisError(null)
    setPredictionRunKey(0)
    setPredictionJobBases({})
    contextRequest.current?.abort()
    const controller = new AbortController()
    contextRequest.current = controller
    setContextStatus('preparing')
    setContextSummaries({})
    setContextError(null)
    try {
      const prepared = await backendApi.prepareDocumentContexts(
        documentId,
        controller.signal,
      )
      if (prepared.document_id !== documentId) {
        throw new Error('The backend returned contexts for a different document.')
      }
      if (prepared.passage_count !== prepared.contexts.length) {
        throw new Error('The backend returned an incomplete passage context set.')
      }
      const byPassage = Object.fromEntries(
        prepared.contexts.map((summary) => [summary.passage_id, summary]),
      )
      if (Object.keys(byPassage).length !== prepared.contexts.length) {
        throw new Error('The backend returned duplicate passage contexts.')
      }
      if (expectedPassageIds) {
        const expected = new Set(expectedPassageIds)
        if (
          expected.size !== prepared.contexts.length
          || prepared.contexts.some((summary) => !expected.has(summary.passage_id))
        ) {
          throw new Error('The backend returned contexts for the wrong passages.')
        }
      }
      const recovered = await backendApi.getDocumentAnalysis(
        documentId,
        controller.signal,
      )
      if (recovered.document_id !== documentId) {
        throw new Error('The backend returned predictions for a different document.')
      }
      for (const job of recovered.jobs) {
        const summary = byPassage[job.passage_id]
        if (!summary || summary.passage_revision !== job.passage_revision) {
          throw new Error('The backend returned a prediction for an old passage revision.')
        }
      }
      setContextSummaries(byPassage)
      setContextStatus('ready')
      if (recovered.jobs.length > 0) {
        setAnalysisScheduleStatus('running')
        setPredictionJobBases(Object.fromEntries(
          recovered.jobs.map((job) => [job.id, job.prediction_basis]),
        ))
        setPredictionRunKey((current) => current + 1)
      }
    } catch (error) {
      if (isAbortError(error)) return
      setContextStatus('failed')
      setContextError(error instanceof BackendApiError
        ? error.message
        : 'We couldn’t prepare this document for prediction. Please try again.')
    } finally {
      if (contextRequest.current === controller) contextRequest.current = null
    }
  }, [backendApi])

  useEffect(() => {
    if (
      !backendReady
      || !document.backendDocumentId
      || contextStatus !== 'ready'
      || !fullDocumentPredictionBasis
      || Object.keys(contextSummaries).length === 0
      || Object.values(predictionJobBases).includes(fullDocumentPredictionBasis)
    ) return

    const recoveryKey = [
      document.backendDocumentId,
      fullDocumentPredictionBasis,
      ...Object.values(contextSummaries)
        .map((summary) => `${summary.passage_id}:${summary.passage_revision}`)
        .sort(),
    ].join('|')
    if (basisRecoveryKey.current === recoveryKey) return
    basisRecoveryKey.current = recoveryKey

    const controller = new AbortController()
    void backendApi.getDocumentAnalysis(
      document.backendDocumentId,
      controller.signal,
      fullDocumentPredictionBasis,
    ).then((recovered) => {
      if (controller.signal.aborted) return
      const jobBases = validateRecoveredBasisRun(
        recovered,
        document.backendDocumentId!,
        contextSummaries,
        fullDocumentPredictionBasis,
      )
      if (Object.keys(jobBases).length === 0) return
      setAnalysisScheduleStatus('running')
      setPredictionJobBases(jobBases)
      setPredictionRunKey((current) => current + 1)
    }).catch((error: unknown) => {
      if (!isAbortError(error) && !controller.signal.aborted) {
        basisRecoveryKey.current = null
      }
    })
    return () => controller.abort()
  }, [
    backendApi,
    backendReady,
    contextStatus,
    contextSummaries,
    document.backendDocumentId,
    fullDocumentPredictionBasis,
    predictionJobBases,
  ])

  useEffect(() => {
    if (analysisScheduleStatus !== 'running') return
    if (modelStream.phaseRunKey !== predictionRunKey) return
    if (modelStream.phase === 'complete' || modelStream.phase === 'empty') {
      setAnalysisScheduleStatus('idle')
    } else if (modelStream.phase === 'failed') {
      setAnalysisScheduleStatus('idle')
    }
  }, [analysisScheduleStatus, modelStream.phase, modelStream.phaseRunKey, predictionRunKey])

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
        if (result.passages.length > 0) {
          void prepareAllPassageContexts(
            documentId,
            result.passages.map((passage) => passage.id),
          )
        }
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
  }, [backendApi, backendReady, document.backendDocumentId, prepareAllPassageContexts])

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
    setContextSummaries({})
    setContextError(null)

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
      void prepareAllPassageContexts(document.backendDocumentId)
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

  async function handleConfirmPassageRemoval() {
    if (
      !backendReady
      || !document.backendDocumentId
      || !passageRemovalTarget
      || passageRemovalTarget.id.startsWith('pending-')
    ) return
    passageRemovalRequest.current?.abort()
    contextRequest.current?.abort()
    contextRequest.current = null
    const controller = new AbortController()
    passageRemovalRequest.current = controller
    const removedPassageId = passageRemovalTarget.id
    const remainingPassages = state.passages.filter(
      (passage) => passage.id !== removedPassageId,
    )
    const contextStatusBeforeRemoval = contextStatus
    setPassageRemovalStatus('deleting')
    setPassageError(null)

    try {
      const deleted = await backendApi.deletePassage(
        removedPassageId,
        controller.signal,
      )
      if (deleted.id !== removedPassageId || deleted.status !== 'deleted') {
        throw new Error('The backend did not confirm the selected passage removal.')
      }
      setContextStatus('idle')
      setContextSummaries({})
      setContextError(null)
      dispatch({ type: 'remove-passage', passageId: removedPassageId })
      setDetectionResult((current) => current
        ? {
            ...current,
            passages: current.passages.filter(
              (passage) => passage.id !== removedPassageId,
            ),
          }
        : current)
      setPassageRemovalId(null)
      setPassageRemovalStatus('idle')
      if (remainingPassages.length > 0) {
        await prepareAllPassageContexts(
          document.backendDocumentId,
          remainingPassages.map((passage) => passage.id),
        )
      }
    } catch (error) {
      if (isAbortError(error)) return
      setPassageRemovalStatus('idle')
      setPassageRemovalId(null)
      setPassageError(errorMessage(error, 'The selected passage could not be removed.'))
      queueMicrotask(() => passageRemovalTrigger.current?.focus())
      if (contextStatusBeforeRemoval === 'preparing') {
        void prepareAllPassageContexts(
          document.backendDocumentId,
          state.passages.map((passage) => passage.id),
        )
      }
    } finally {
      if (passageRemovalRequest.current === controller) {
        passageRemovalRequest.current = null
      }
    }
  }

  async function handlePredictSelectedPassage() {
    if (
      !backendReady
      || contextStatus !== 'ready'
      || !document.backendDocumentId
      || !selectedPassage
    ) return
    const selectedContext = contextSummaries[selectedPassage.id]
    if (selectedPassage.id.startsWith('pending-') || !selectedContext) {
      setAnalysisError('The selected passage needs a matching prepared revision before prediction.')
      return
    }
    analysisRequest.current?.abort()
    const controller = new AbortController()
    analysisRequest.current = controller
    setAnalysisScheduleStatus('scheduling')
    setAnalysisError(null)
    const runId = analysisRunId.current ?? createIdempotencyKey()
    analysisRunId.current = runId
    try {
      const job = await backendApi.schedulePassageAnalysis(
        selectedPassage.id,
        selectedContext.passage_revision,
        { idempotencyKey: runId, signal: controller.signal },
      )
      if (
        job.document_id !== document.backendDocumentId
        || job.passage_id !== selectedPassage.id
        || job.passage_revision !== selectedContext.passage_revision
      ) {
        throw new Error('The backend returned a prediction for a different passage.')
      }
      if (job.status === 'blocked' || job.blockers.length > 0) {
        analysisRunId.current = null
        setAnalysisScheduleStatus('blocked')
        setAnalysisError('Predictions are not available yet. Please try again later.')
        return
      }
      setAnalysisScheduleStatus('running')
      setPredictionJobBases({ [job.id]: job.prediction_basis })
      setPredictionRunKey((current) => current + 1)
      analysisRunId.current = null
    } catch (error) {
      if (isAbortError(error)) return
      setAnalysisScheduleStatus('idle')
      setAnalysisError(errorMessage(error, 'Predictions could not be started.'))
    } finally {
      if (analysisRequest.current === controller) analysisRequest.current = null
    }
  }

  async function handlePredictAllPassages() {
    if (
      !backendReady
      || !fullDocumentPredictionBasis
      || contextStatus !== 'ready'
      || !document.backendDocumentId
      || passages.length === 0
    ) return
    analysisRequest.current?.abort()
    const controller = new AbortController()
    analysisRequest.current = controller
    setAnalysisScheduleStatus('scheduling')
    setAnalysisError(null)
    const runId = analysisRunId.current ?? createIdempotencyKey()
    analysisRunId.current = runId
    try {
      const result = await backendApi.scheduleDocumentAnalysis(
        document.backendDocumentId,
        {
          idempotencyKey: runId,
          signal: controller.signal,
          expectedPredictionBasis: fullDocumentPredictionBasis,
        },
      )
      let jobBases: Record<string, PredictionBasis>
      try {
        jobBases = validateFullDocumentRun(
          result,
          document.backendDocumentId,
          contextSummaries,
          fullDocumentPredictionBasis,
        )
      } catch {
        analysisRunId.current = null
        throw new Error('The backend returned an invalid prediction run.')
      }
      setAnalysisScheduleStatus('running')
      setPredictionJobBases(jobBases)
      setPredictionRunKey((current) => current + 1)
      analysisRunId.current = null
    } catch (error) {
      if (isAbortError(error)) return
      setAnalysisScheduleStatus('idle')
      setAnalysisError(errorMessage(error, 'Predictions could not be started.'))
    } finally {
      if (analysisRequest.current === controller) analysisRequest.current = null
    }
  }

  function handleSelectPassage(passageId: string) {
    const passage = passages.find((item) => item.id === passageId)
    if (!passage) return
    dispatch({ type: 'select-passage', passageId, page: passage.page })
  }

  const selectedHasPrediction = selectedPassage
    ? predictions.some((prediction) => prediction.passageId === selectedPassage.id)
    : false
  const hasFullDocumentPrediction = fullDocumentPredictionBasis !== null
    && predictions.some(
      (prediction) => prediction.predictionBasis === fullDocumentPredictionBasis,
    )
  const hasRemainingPrediction = passages.some(
    (passage) => !predictions.some(
      (prediction) => prediction.passageId === passage.id
        && (
          fullDocumentPredictionBasis === null
          || prediction.predictionBasis === fullDocumentPredictionBasis
        ),
    ),
  )
  const fullDocumentPredictionReady = fullDocumentPredictionBasis !== null

  return (
    <main className="workspace-shell">
      <header className="workspace-header">
        <div className="workspace-identity">
          <span className="workspace-brand">Redacted Document LLM</span>
          <span className="header-divider" aria-hidden="true" />
          <span className="filename" title={document.filename}>{document.filename}</span>
        </div>
        <div className="workspace-actions">
          {onViewDemo ? (
            <button
              className="secondary-button"
              type="button"
              onClick={(event) => onViewDemo(event.currentTarget)}
            >
              View fictional demo
            </button>
          ) : null}
          <button
            className="secondary-button"
            type="button"
            aria-label={deletionInProgress ? 'Removing document…' : 'Replace document'}
            disabled={deletionInProgress || predictionWorkActive}
            onClick={onReplace}
          >
            {deletionInProgress ? 'Removing document…' : 'Replace document'}
          </button>
        </div>
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
                  <>
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
                    <button
                      ref={passageRemovalTrigger}
                      className="danger-text-button"
                      type="button"
                      disabled={markingUnavailable}
                      onClick={() => {
                        setPassageRemovalId(selectedPassage.id)
                        setPassageRemovalStatus('confirming')
                        setPassageError(null)
                      }}
                    >
                      Remove selected passage
                    </button>
                  </>
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
          {passageRemovalTarget ? (
            <section
              className="passage-removal-confirmation"
              role="alertdialog"
              aria-labelledby="passage-removal-title"
              aria-describedby="passage-removal-description"
            >
              <div>
                <strong id="passage-removal-title">
                  Remove {passageRemovalTarget.label}?
                </strong>
                <p id="passage-removal-description">
                  Its highlight and saved context will be removed. The remaining passage
                  contexts will be rebuilt automatically.
                </p>
              </div>
              <div className="passage-removal-actions">
                <button
                  autoFocus
                  type="button"
                  disabled={passageRemovalStatus === 'deleting'}
                  onClick={() => {
                    setPassageRemovalId(null)
                    setPassageRemovalStatus('idle')
                    queueMicrotask(() => passageRemovalTrigger.current?.focus())
                  }}
                >
                  Keep passage
                </button>
                <button
                  className="danger-button"
                  type="button"
                  disabled={passageRemovalStatus === 'deleting'}
                  onClick={() => void handleConfirmPassageRemoval()}
                >
                  {passageRemovalStatus === 'deleting' ? 'Removing…' : 'Confirm removal'}
                </button>
              </div>
            </section>
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
            error={analysisError
              ?? modelStream.error
              ?? passageError
              ?? contextError
              ?? (detectionError
                ? 'We couldn’t prepare this document for prediction. Please try again.'
                : null)
              ?? backendDocumentError(backendDocument)}
            activePassageId={activePassageId}
            onSelectPassage={handleSelectPassage}
            onDismissSelection={() => dispatch({ type: 'dismiss' })}
            missingPredictionMessage={passageSaveStatus === 'saving'
              ? 'Preparing this passage…'
              : detectionStatus === 'detecting'
                ? 'Preparing this passage…'
                : detectionStatus === 'failed' || contextStatus === 'failed'
                  ? 'Predictions are unavailable for this passage.'
                  : contextStatus === 'preparing'
                    ? 'Preparing this passage…'
                    : contextStatus === 'ready'
                      ? analysisScheduleStatus === 'running'
                        ? 'Prediction is running…'
                        : 'Select Predict to generate this result.'
                      : passages.length
                        ? 'Preparing this passage…'
                        : 'No hidden passages are available for prediction.'}
          >
            {document.mode === 'uploaded' ? (
              <div className="uploaded-document-note">
                <h3>{predictionStatusTitle(
                  backendDocument,
                  passageSaveStatus,
                  contextStatus,
                  detectionStatus,
                  analysisScheduleStatus,
                  passages.length,
                )}</h3>
                <p>{predictionStatusMessage(
                  backendDocument,
                  passageSaveStatus,
                  contextStatus,
                  detectionStatus,
                  analysisScheduleStatus,
                  passages.length,
                )}</p>
                {contextStatus === 'failed'
                && document.backendDocumentId
                && passages.length > 0 ? (
                  <button
                    className="primary-button"
                    type="button"
                    onClick={() => void prepareAllPassageContexts(
                      document.backendDocumentId!,
                      passages.map((passage) => passage.id),
                    )}
                  >
                    Try again
                  </button>
                  ) : null}
                {contextStatus === 'ready' && passages.length > 0 ? (
                  <button
                    className="primary-button"
                    type="button"
                    disabled={
                      deletionInProgress
                      || analysisScheduleStatus === 'scheduling'
                      || analysisScheduleStatus === 'running'
                      || (fullDocumentPredictionReady
                        ? !hasRemainingPrediction
                        : !selectedPassage
                          || !contextSummaries[selectedPassage.id]
                          || selectedHasPrediction)
                    }
                    onClick={() => void (
                      fullDocumentPredictionReady
                        ? handlePredictAllPassages()
                        : handlePredictSelectedPassage()
                    )}
                  >
                    {analysisScheduleStatus === 'scheduling'
                      ? fullDocumentPredictionReady
                        ? 'Starting predictions…'
                        : 'Starting prediction…'
                      : analysisScheduleStatus === 'running'
                        ? fullDocumentPredictionReady
                          ? 'Predictions running…'
                          : 'Prediction running…'
                        : fullDocumentPredictionReady
                          ? hasFullDocumentPrediction
                            ? 'Predict remaining passages'
                            : 'Predict all passages'
                          : 'Predict passage'}
                  </button>
                ) : null}
              </div>
            ) : null}
          </PredictionSidebar>
        </div>
      </div>
    </main>
  )
}

function readyFullDocumentBasis(
  readiness: BackendReadiness,
): FullDocumentPredictionBasis | null {
  if (!readiness.prediction_ready) return null
  const basis = readiness.strongest_ready_basis
  if (
    basis !== 'best_available_foiarchive_v1'
    && basis !== 'paper_exact_v1'
  ) return null
  const status = readiness.prediction_bases?.[basis]
  return status?.operational === true
    && status.approved === true
    && status.ready === true
    ? basis
    : null
}

function validateFullDocumentRun(
  result: DocumentAnalysisScheduleResult,
  documentId: string,
  contexts: Record<string, VisibleContextSummary>,
  expectedBasis: FullDocumentPredictionBasis,
): Record<string, PredictionBasis> {
  const expectedPassages = new Set(Object.keys(contexts))
  if (
    result.document_id !== documentId
    || result.blockers.length > 0
    || result.deferred_passage_ids.length > 0
    || result.accepted_jobs.length !== expectedPassages.size
  ) throw new Error('invalid full-document schedule')

  const seenPassages = new Set<string>()
  let runBasis: PredictionBasis | null = null
  const jobBases: Record<string, PredictionBasis> = {}
  for (const job of result.accepted_jobs) {
    const context = contexts[job.passage_id]
    const fullBasis = job.prediction_basis === 'best_available_foiarchive_v1'
      || job.prediction_basis === 'paper_exact_v1'
    if (
      job.document_id !== documentId
      || !context
      || context.passage_revision !== job.passage_revision
      || seenPassages.has(job.passage_id)
      || !fullBasis
      || job.prediction_basis !== expectedBasis
      || (runBasis !== null && job.prediction_basis !== runBasis)
      || job.status === 'blocked'
      || job.status === 'failed'
      || job.status === 'cancelled'
      || job.blockers.length > 0
      || Object.prototype.hasOwnProperty.call(jobBases, job.id)
    ) throw new Error('invalid full-document schedule')
    seenPassages.add(job.passage_id)
    runBasis = job.prediction_basis
    jobBases[job.id] = job.prediction_basis
  }
  if ([...expectedPassages].some((passageId) => !seenPassages.has(passageId))) {
    throw new Error('invalid full-document schedule')
  }
  return jobBases
}

function validateRecoveredBasisRun(
  result: Awaited<ReturnType<BackendApi['getDocumentAnalysis']>>,
  documentId: string,
  contexts: Record<string, VisibleContextSummary>,
  expectedBasis: FullDocumentPredictionBasis,
): Record<string, PredictionBasis> {
  if (result.document_id !== documentId) {
    throw new Error('invalid full-document recovery')
  }

  const seenPassages = new Set<string>()
  const jobBases: Record<string, PredictionBasis> = {}
  for (const job of result.jobs) {
    const context = contexts[job.passage_id]
    if (
      job.document_id !== documentId
      || !context
      || context.passage_revision !== job.passage_revision
      || seenPassages.has(job.passage_id)
      || job.prediction_basis !== expectedBasis
      || job.status === 'blocked'
      || job.status === 'failed'
      || job.status === 'cancelled'
      || job.blockers.length > 0
      || Object.prototype.hasOwnProperty.call(jobBases, job.id)
    ) throw new Error('invalid full-document recovery')
    seenPassages.add(job.passage_id)
    jobBases[job.id] = job.prediction_basis
  }
  return jobBases
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

function predictionStatusTitle(
  backendDocument: BackendDocumentState | undefined,
  passageSaveStatus: PassageSaveStatus,
  contextStatus: ContextPreparationStatus,
  detectionStatus: PassageDetectionStatus,
  analysisScheduleStatus: AnalysisScheduleStatus,
  passageCount: number,
): string {
  if (backendDocument?.status === 'deleting') return 'Removing document…'
  if (
    backendDocument?.status === 'unavailable'
    || backendDocument?.status === 'failed'
    || passageSaveStatus === 'failed'
    || contextStatus === 'failed'
    || detectionStatus === 'failed'
    || analysisScheduleStatus === 'blocked'
  ) return 'Predictions unavailable'
  if (contextStatus === 'ready' && passageCount > 0) return 'Ready to predict'
  if (detectionStatus === 'ready') return 'No redactions detected'
  return 'Preparing document…'
}

function predictionStatusMessage(
  backendDocument: BackendDocumentState | undefined,
  passageSaveStatus: PassageSaveStatus,
  contextStatus: ContextPreparationStatus,
  detectionStatus: PassageDetectionStatus,
  analysisScheduleStatus: AnalysisScheduleStatus,
  passageCount: number,
): string {
  if (backendDocument?.status === 'deleting') {
    return 'The app will return to the upload screen when removal is complete.'
  }
  if (
    backendDocument?.status === 'unavailable'
    || backendDocument?.status === 'failed'
    || passageSaveStatus === 'failed'
    || contextStatus === 'failed'
    || detectionStatus === 'failed'
    || analysisScheduleStatus === 'blocked'
  ) return 'Predictions are not available right now. Your PDF remains open.'
  if (contextStatus === 'ready' && passageCount > 0) {
    return `${passageCount} hidden ${passageCount === 1 ? 'passage is' : 'passages are'} ready.`
  }
  if (detectionStatus === 'ready') {
    return 'No hidden passages were detected. You can mark one manually.'
  }
  return 'Finding hidden passages and preparing them for prediction.'
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
