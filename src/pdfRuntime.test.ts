import pdfjsPackage from 'pdfjs-dist/package.json'
import reactPdfPackage from 'react-pdf/package.json'
import { configurePdfViewerWorker } from './pdfRuntime'

describe('PDF.js runtime', () => {
  it('uses the exact worker version required by React-PDF', () => {
    expect(pdfjsPackage.version).toBe(reactPdfPackage.dependencies['pdfjs-dist'])
  })

  it('uses the compatibility worker for browsers without new Uint8Array methods', () => {
    const runtime = { GlobalWorkerOptions: { workerSrc: '' } }

    configurePdfViewerWorker(runtime)

    expect(runtime.GlobalWorkerOptions.workerSrc).toContain('/legacy/build/pdf.worker.min.mjs')
  })
})
