export type PdfValidationResult = { ok: true } | { ok: false; message: string }

const WRONG_FILE_MESSAGE = 'Choose a PDF file to continue.'
const UNREADABLE_FILE_MESSAGE = 'We couldn’t read that PDF. Try another file or export it again.'

export async function validatePdfFile(file: File): Promise<PdfValidationResult> {
  const hasPdfName = file.name.toLowerCase().endsWith('.pdf')
  const hasPdfType = file.type === 'application/pdf'
  if (!hasPdfName && !hasPdfType) {
    return { ok: false, message: WRONG_FILE_MESSAGE }
  }

  try {
    const bytes = new Uint8Array(await file.arrayBuffer())
    const signature = new TextDecoder().decode(bytes.slice(0, 5))
    if (signature !== '%PDF-') {
      return { ok: false, message: UNREADABLE_FILE_MESSAGE }
    }

    return { ok: true }
  } catch {
    return { ok: false, message: UNREADABLE_FILE_MESSAGE }
  }
}
