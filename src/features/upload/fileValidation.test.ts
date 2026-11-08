import { PDFDocument } from 'pdf-lib'
import { validatePdfFile } from './fileValidation'

async function validPdfFile(name = 'briefing.pdf') {
  const document = await PDFDocument.create()
  document.addPage([300, 400])
  const bytes = new Uint8Array(await document.save())
  return new File([bytes.buffer], name, { type: 'application/pdf' })
}

describe('validatePdfFile', () => {
  it('rejects a non-PDF before attempting to parse it', async () => {
    const result = await validatePdfFile(new File(['notes'], 'notes.txt', { type: 'text/plain' }))
    expect(result).toEqual({ ok: false, message: 'Choose a PDF file to continue.' })
  })

  it('accepts a named PDF without parsing its bytes during upload', async () => {
    const result = await validatePdfFile(
      new File(['this is not a PDF'], 'damaged.pdf', { type: 'application/pdf' }),
    )
    expect(result).toEqual({ ok: true })
  })

  it('accepts a parseable browser-local PDF', async () => {
    expect(validatePdfFile(await validPdfFile())).toEqual({ ok: true })
  })

})
