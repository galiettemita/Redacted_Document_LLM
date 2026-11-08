import { getDocument as getLegacyDocument, GlobalWorkerOptions as LegacyWorkerOptions } from 'pdfjs-dist/legacy/build/pdf.mjs'

interface PdfRuntime {
  GlobalWorkerOptions: { workerSrc: string }
}

const runtimeProcess = (globalThis as typeof globalThis & {
  process?: { versions?: { node?: string } }
}).process
const isBrowserRuntime = !runtimeProcess?.versions?.node

export function configureLegacyPdfWorker(forceBrowser = false) {
  if (forceBrowser || isBrowserRuntime) {
    LegacyWorkerOptions.workerSrc = new URL(
      'pdfjs-dist/legacy/build/pdf.worker.min.mjs',
      import.meta.url,
    ).toString()
  }
}

export function configurePdfViewerWorker(runtime: PdfRuntime) {
  runtime.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url,
  ).toString()
}

configureLegacyPdfWorker()

export { getLegacyDocument as getPdfDocument }
