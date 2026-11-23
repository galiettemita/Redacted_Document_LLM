interface PdfRuntime {
  GlobalWorkerOptions: { workerSrc: string }
}

const publicBaseUrl = import.meta.env.BASE_URL.endsWith('/')
  ? import.meta.env.BASE_URL
  : `${import.meta.env.BASE_URL}/`

export const PDF_DOCUMENT_OPTIONS = Object.freeze({
  wasmUrl: `${publicBaseUrl}pdfjs/wasm/`,
})

export function configurePdfViewerWorker(runtime: PdfRuntime) {
  runtime.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/legacy/build/pdf.worker.min.mjs',
    import.meta.url,
  ).toString()
}
