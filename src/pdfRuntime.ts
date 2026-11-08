interface PdfRuntime {
  GlobalWorkerOptions: { workerSrc: string }
}

export function configurePdfViewerWorker(runtime: PdfRuntime) {
  runtime.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url,
  ).toString()
}
