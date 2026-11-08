export type PdfValidationResult = { ok: true } | { ok: false; message: string }

const WRONG_FILE_MESSAGE = 'Choose a PDF file to continue.'

export function validatePdfFile(file: File): PdfValidationResult {
  const hasPdfName = file.name.toLowerCase().endsWith('.pdf')
  const hasPdfType = file.type === 'application/pdf'
  if (!hasPdfName && !hasPdfType) {
    return { ok: false, message: WRONG_FILE_MESSAGE }
  }

  return { ok: true }
}
