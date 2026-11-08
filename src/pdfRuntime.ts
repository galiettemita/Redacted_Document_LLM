interface PdfRuntime {
  GlobalWorkerOptions: { workerSrc: string }
}

export function configurePdfViewerWorker(runtime: PdfRuntime) {
  runtime.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/legacy/build/pdf.worker.min.mjs',
    import.meta.url,
  ).toString()
}
