vi.mock('../../pdfRuntime', () => ({
  getPdfDocument: () => ({
    promise: new Promise(() => undefined),
    destroy: vi.fn(),
  }),
}))

import { validatePdfFile } from './fileValidation'

it('accepts a PDF signature without waiting for a PDF worker', async () => {
  const file = new File(['%PDF-1.7\n'], 'record.pdf', { type: 'application/pdf' })
  const arrayBuffer = vi.fn(() => new Promise<ArrayBuffer>(() => undefined))
  Object.defineProperty(file, 'arrayBuffer', { value: arrayBuffer })
  const result = await Promise.race([
    validatePdfFile(file),
    new Promise<'timed-out'>((resolve) => setTimeout(() => resolve('timed-out'), 100)),
  ])

  expect(result).toEqual({ ok: true })
  expect(arrayBuffer).not.toHaveBeenCalled()
})
