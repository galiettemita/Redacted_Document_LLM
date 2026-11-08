import { PDFDocument } from 'pdf-lib'
import { GlobalWorkerOptions } from 'pdfjs-dist/legacy/build/pdf.mjs'
import { configureLegacyPdfWorker } from '../../pdfRuntime'
import { validatePdfFile } from './fileValidation'

async function validPdfFile(name = 'briefing.pdf') {
  const document = await PDFDocument.create()
  document.addPage([300, 400])
  const bytes = new Uint8Array(await document.save())
  return new File([bytes.buffer], name, { type: 'application/pdf' })
}

describe('validatePdfFile', () => {
  it('configures the browser worker used by local validation', () => {
    const originalWorkerSrc = GlobalWorkerOptions.workerSrc
    try {
      configureLegacyPdfWorker(true)
      expect(GlobalWorkerOptions.workerSrc).toMatch(/pdf\.worker\.min.*\.mjs/)
    } finally {
      GlobalWorkerOptions.workerSrc = originalWorkerSrc
    }
  })

  it('rejects a non-PDF before attempting to parse it', async () => {
    const result = await validatePdfFile(new File(['notes'], 'notes.txt', { type: 'text/plain' }))
    expect(result).toEqual({ ok: false, message: 'Choose a PDF file to continue.' })
  })

  it('rejects a corrupt file even when it is named like a PDF', async () => {
    const result = await validatePdfFile(
      new File(['this is not a PDF'], 'damaged.pdf', { type: 'application/pdf' }),
    )
    expect(result).toEqual({
      ok: false,
      message: 'We couldn’t read that PDF. Try another file or export it again.',
    })
  })

  it('accepts a parseable browser-local PDF', async () => {
    await expect(validatePdfFile(await validPdfFile())).resolves.toEqual({ ok: true })
  })

  it('destroys a failed PDF loading task after a valid signature reaches the parser', async () => {
    const destroy = vi.fn().mockResolvedValue(undefined)
    const createLoadingTask = vi.fn(() => ({
      promise: Promise.reject(new Error('Malformed PDF structure')),
      destroy,
    }))
    const file = new File(['%PDF-1.7\nmalformed'], 'malformed.pdf', { type: 'application/pdf' })

    await expect(validatePdfFile(file, createLoadingTask)).resolves.toEqual({
      ok: false,
      message: 'We couldn’t read that PDF. Try another file or export it again.',
    })
    expect(destroy).toHaveBeenCalledOnce()
  })
})
