import type { PredictionService } from './predictionService'
import type { Prediction } from '../types'

interface PredictionEventPayload {
  job_id: string
  passage_id: string
  page_number: number
  model: string
  prediction: string
  usage: {
    input_tokens: number
    output_tokens: number
    total_tokens: number
    reasoning_tokens: number
  }
  estimated_cost_usd: number
  pricing_version: string
}

interface ErrorEventPayload {
  job_id: string
  code: string
  detail: string
}

class PredictionEventError extends Error {}

export function createHttpPredictionService(
  fetchImplementation?: typeof fetch,
): PredictionService {
  const fetchRequest: typeof fetch = (...arguments_) => (
    (fetchImplementation ?? globalThis.fetch)(...arguments_)
  )
  return {
    async stream(document, signal, onPrediction, options) {
      if (!document.backendDocumentId) {
        throw new Error('A persisted backend document is required for predictions.')
      }
      const url = `/v1/documents/${encodeURIComponent(document.backendDocumentId)}/events`
      const failures = new Map<string, string>()
      let lastEventId: string | null = null
      let reconnects = 0
      while (true) {
        try {
          const headers: Record<string, string> = { Accept: 'text/event-stream' }
          if (lastEventId !== null) headers['Last-Event-ID'] = lastEventId
          const response = await fetchRequest(url, { headers, signal })
          if (!response.ok) {
            throw new PredictionEventError(
              `The prediction event stream failed (${response.status}).`,
            )
          }
          if (!response.body) {
            throw new PredictionEventError(
              'The prediction event stream returned no body.',
            )
          }
          await readEventStream(response.body, signal, (id, event, data) => {
            if (id !== null) lastEventId = id
            if (event === 'passage.prediction') {
              const payload = parsePrediction(data)
              if (!belongsToRun(payload.job_id, options?.jobIds)) return
              onPrediction({
                passageId: payload.passage_id,
                page: payload.page_number,
                model: displayModelName(payload.model),
                text: payload.prediction,
                provenance: 'model',
                usage: {
                  inputTokens: payload.usage.input_tokens,
                  outputTokens: payload.usage.output_tokens,
                  totalTokens: payload.usage.total_tokens,
                  reasoningTokens: payload.usage.reasoning_tokens,
                  estimatedCostUsd: payload.estimated_cost_usd,
                  pricingVersion: payload.pricing_version,
                },
              })
            } else if (event === 'passage.error') {
              const payload = parseError(data)
              if (!belongsToRun(payload.job_id, options?.jobIds)) return
              failures.set(`${payload.job_id}:${payload.code}`, payload.detail)
            }
          })
          if (failures.size > 0) {
            throw new PredictionEventError([...failures.values()].join('; '))
          }
          return
        } catch (error) {
          if (isAbortError(error) || error instanceof PredictionEventError) throw error
          reconnects += 1
          if (reconnects > 3) {
            throw new Error('The prediction event stream is unavailable.', { cause: error })
          }
          await abortableDelay(50, signal)
        }
      }
    },
  }
}

async function readEventStream(
  body: ReadableStream<Uint8Array>,
  signal: AbortSignal,
  onEvent: (id: string | null, event: string, data: string) => void,
): Promise<void> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      if (signal.aborted) throw new DOMException('Prediction stream aborted.', 'AbortError')
      const { done, value } = await reader.read()
      buffer += decoder.decode(value, { stream: !done }).replaceAll('\r\n', '\n')
      let boundary = buffer.indexOf('\n\n')
      while (boundary >= 0) {
        const frame = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)
        emitFrame(frame, onEvent)
        boundary = buffer.indexOf('\n\n')
      }
      if (done) break
    }
    if (buffer.trim()) emitFrame(buffer, onEvent)
  } finally {
    reader.releaseLock()
  }
}

function emitFrame(
  frame: string,
  onEvent: (id: string | null, event: string, data: string) => void,
): void {
  if (!frame || frame.startsWith(':')) return
  let id: string | null = null
  let event = 'message'
  const data: string[] = []
  for (const line of frame.split('\n')) {
    if (line.startsWith('id:')) id = line.slice(3).trimStart()
    if (line.startsWith('event:')) event = line.slice(6).trimStart()
    if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
  }
  if (data.length) onEvent(id, event, data.join('\n'))
}

function parsePrediction(data: string): PredictionEventPayload {
  let value: unknown
  try {
    value = JSON.parse(data)
  } catch (error) {
    throw new PredictionEventError('The backend returned an invalid prediction event.', {
      cause: error,
    })
  }
  if (
    typeof value !== 'object'
    || value === null
    || !('passage_id' in value)
    || !('job_id' in value)
    || typeof value.job_id !== 'string'
    || !value.job_id
    || typeof value.passage_id !== 'string'
    || !('page_number' in value)
    || typeof value.page_number !== 'number'
    || !Number.isInteger(value.page_number)
    || value.page_number < 1
    || !('model' in value)
    || typeof value.model !== 'string'
    || !('prediction' in value)
    || typeof value.prediction !== 'string'
    || !value.prediction.trim()
    || !('usage' in value)
    || !isUsage(value.usage)
    || !('estimated_cost_usd' in value)
    || typeof value.estimated_cost_usd !== 'number'
    || value.estimated_cost_usd < 0
    || !('pricing_version' in value)
    || typeof value.pricing_version !== 'string'
    || !value.pricing_version
  ) {
    throw new PredictionEventError('The backend returned an invalid prediction event.')
  }
  return value as PredictionEventPayload
}

function isUsage(value: unknown): value is PredictionEventPayload['usage'] {
  if (typeof value !== 'object' || value === null) return false
  return [
    'input_tokens',
    'output_tokens',
    'total_tokens',
    'reasoning_tokens',
  ].every((field) => (
    field in value
    && typeof value[field as keyof typeof value] === 'number'
    && Number.isInteger(value[field as keyof typeof value])
    && value[field as keyof typeof value] >= 0
  ))
}

function parseError(data: string): ErrorEventPayload {
  let value: unknown
  try {
    value = JSON.parse(data)
  } catch (error) {
    throw new PredictionEventError('The backend returned an invalid prediction error event.', {
      cause: error,
    })
  }
  if (
    typeof value !== 'object'
    || value === null
    || !('job_id' in value)
    || typeof value.job_id !== 'string'
    || !value.job_id
    || !('code' in value)
    || typeof value.code !== 'string'
    || !('detail' in value)
    || typeof value.detail !== 'string'
    || !value.detail
  ) {
    throw new PredictionEventError('The backend returned an invalid prediction error event.')
  }
  return value as ErrorEventPayload
}

function belongsToRun(jobId: string, allowed: readonly string[] | undefined): boolean {
  return !allowed?.length || allowed.includes(jobId)
}

function displayModelName(model: string): string {
  return model === 'gpt-6-astra' ? 'GPT-6 Astra' : model
}

function isAbortError(error: unknown): boolean {
  return typeof error === 'object'
    && error !== null
    && 'name' in error
    && error.name === 'AbortError'
}

function abortableDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('Prediction stream aborted.', 'AbortError'))
      return
    }
    const onAbort = () => {
      globalThis.clearTimeout(timeout)
      reject(new DOMException('Prediction stream aborted.', 'AbortError'))
    }
    const timeout = globalThis.setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, milliseconds)
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

export type { Prediction }
