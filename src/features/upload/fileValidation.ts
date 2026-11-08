import { getPdfDocument } from '../../pdfRuntime'

export type PdfValidationResult = { ok: true } | { ok: false; message: string }

const WRONG_FILE_MESSAGE = 'Choose a PDF file to continue.'
const UNREADABLE_FILE_MESSAGE = 'We couldn’t read that PDF. Try another file or export it again.'

interface PdfLoadingTask {
  promise: Promise<unknown>
  destroy: () => Promise<void>
}

type CreateLoadingTask = (options: { data: Uint8Array }) => PdfLoadingTask

const createDefaultLoadingTask: CreateLoadingTask = (options) => getPdfDocument(options)

export async function validatePdfFile(
  file: File,
  createLoadingTask: CreateLoadingTask = createDefaultLoadingTask,
): Promise<PdfValidationResult> {
  const hasPdfName = file.name.toLowerCase().endsWith('.pdf')
  const hasPdfType = file.type === 'application/pdf'
  if (!hasPdfName && !hasPdfType) {
    return { ok: false, message: WRONG_FILE_MESSAGE }
  }

  let loadingTask: PdfLoadingTask | null = null
  try {
    const bytes = new Uint8Array(await file.arrayBuffer())
    const signature = new TextDecoder().decode(bytes.slice(0, 5))
    if (signature !== '%PDF-') {
      return { ok: false, message: UNREADABLE_FILE_MESSAGE }
    }

    loadingTask = createLoadingTask({ data: bytes })
    await loadingTask.promise
    return { ok: true }
  } catch {
    return { ok: false, message: UNREADABLE_FILE_MESSAGE }
  } finally {
    if (loadingTask) {
      try {
        await loadingTask.destroy()
      } catch {
        // Validation has already produced its user-facing result; cleanup errors are non-actionable.
      }
    }
  }
}
