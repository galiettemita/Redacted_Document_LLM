import { getDocument, OPS } from 'pdfjs-dist/legacy/build/pdf.mjs'
import pdfjsPackage from 'pdfjs-dist/package.json'
import reactPdfPackage from 'react-pdf/package.json'
import { configurePdfViewerWorker, PDF_DOCUMENT_OPTIONS } from './pdfRuntime'

// Mozilla PDF.js v6.3.289 test fixture: jbig2_symbol_offset.pdf.
const JBIG2_PDF_BASE64 = 'JVBERi0xLjQNCiX/9uTnDQoxIDAgb2JqDQo8PA0KL1R5cGUgL0NhdGFsb2cNCi9PcGVuQWN0aW9uIFs0IDAgUiAvRml0XQ0KL1BhZ2VzIDMgMCBSDQo+Pg0KZW5kb2JqDQoNCjIgMCBvYmoNCjw8DQovVHlwZSAvRm9udA0KL1N1YnR5cGUgL1R5cGUxDQovQmFzZUZvbnQgL0hlbHZldGljYQ0KL0VuY29kaW5nIC9XaW5BbnNpRW5jb2RpbmcNCj4+DQplbmRvYmoNCg0KMyAwIG9iag0KPDwNCi9UeXBlIC9QYWdlcw0KL0NvdW50IDENCi9LaWRzIFsNCjQgMCBSDQpdDQo+Pg0KZW5kb2JqDQoNCjQgMCBvYmoNCjw8DQovVHlwZSAvUGFnZQ0KL1BhcmVudCAzIDAgUg0KL0NvbnRlbnRzIDUgMCBSDQovTWVkaWFCb3ggWzAgMCA1OTUuMjc1NiA4NDEuODg5OF0NCi9SZXNvdXJjZXMgPDwNCiAgICAvRm9udCA8PCAvRjEgMiAwIFIgPj4NCiAgICAvWE9iamVjdCA8PCAvSW0xIDYgMCBSID4+DQogID4+DQo+Pg0KZW5kb2JqDQoNCjUgMCBvYmoNCjw8IC9MZW5ndGggNDYgPj4NCnN0cmVhbQ0KcQ0KICA1MDAuMDAgMCAwIDUzLjAzMCA0OCA1NTAgY20NCiAgL0ltMSBEbw0KUQ0KZW5kc3RyZWFtDQplbmRvYmoNCg0KNiAwIG9iag0KPDwNCi9UeXBlIC9YT2JqZWN0DQovU3VidHlwZSAvSW1hZ2UNCi9XaWR0aCAxMzINCi9IZWlnaHQgMTQNCi9CaXRzUGVyQ29tcG9uZW50IDENCi9Db2xvclNwYWNlIC9EZXZpY2VHcmF5DQovRmlsdGVyIC9KQklHMkRlY29kZQ0KL0xlbmd0aCAxOTENCj4+DQpzdHJlYW0NCgAAAAAwAAEAAAATAAAAhAAAAA4AAAAAAAAAAAEAAAAAAAEAAQEAAABkCAAC/wAAAAgAAAAIOkg3iqy0BjDgkep75WQd/m2/8EWNvdsgkc5tcHTm85FPbi9ou+AvbOtrheIo0lRTEh6R0c5KRCGBPNKLnUP/KtAYszHJwfNPX7s3X2A3ou+KuVPcNBv/rAAAAAIHIAEBAAAAJgAAAIQAAAAOAAAAAAAAAAAAAAQAAAAJ6NCe+drwh3BhxTn6v/+sDQplbmRzdHJlYW0NCmVuZG9iag0KDQp4cmVmDQowIDcNCjAwMDAwMDAwMDAgNjU1MzUgZg0KMDAwMDAwMDAxNyAwMDAwMCBuDQowMDAwMDAwMTAwIDAwMDAwIG4NCjAwMDAwMDAyMDcgMDAwMDAgbg0KMDAwMDAwMDI3NyAwMDAwMCBuDQowMDAwMDAwNDYzIDAwMDAwIG4NCjAwMDAwMDA1NjcgMDAwMDAgbg0KDQp0cmFpbGVyDQo8PCAvU2l6ZSA3DQovUm9vdCAxIDAgUiA+Pg0Kc3RhcnR4cmVmDQo5NDQNCiUlRU9GDQo='

describe('PDF.js runtime', () => {
  it('uses the exact worker version required by React-PDF', () => {
    expect(pdfjsPackage.version).toBe(reactPdfPackage.dependencies['pdfjs-dist'])
  })

  it('uses the compatibility worker for browsers without new Uint8Array methods', () => {
    const runtime = { GlobalWorkerOptions: { workerSrc: '' } }

    configurePdfViewerWorker(runtime)

    expect(runtime.GlobalWorkerOptions.workerSrc).toContain('/legacy/build/pdf.worker.min.mjs')
  })

  it('decodes a JBIG2 page through the published PDF.js WASM assets', async () => {
    expect(PDF_DOCUMENT_OPTIONS.wasmUrl).toBe('/pdfjs/wasm/')
    const modulePath = decodeURIComponent(new URL(import.meta.url).pathname)
    const projectRoot = modulePath.slice(0, modulePath.lastIndexOf('/src/'))
    const publicWasmPath = `${projectRoot}/public/pdfjs/wasm`
    const loadingTask = getDocument({
      data: Uint8Array.from(atob(JBIG2_PDF_BASE64), (character) => character.charCodeAt(0)),
      wasmUrl: `${publicWasmPath}/`,
      useWorkerFetch: false,
    })

    try {
      const pdf = await loadingTask.promise
      const page = await pdf.getPage(1)
      const operatorList = await page.getOperatorList()
      const imageIndex = operatorList.fnArray.findIndex((operator) => (
        operator === OPS.paintImageXObject || operator === OPS.paintImageMaskXObject
      ))
      const imageId = operatorList.argsArray[imageIndex]?.[0]
      const image = await new Promise<{ data?: Uint8Array } | null>((resolveImage) => {
        page.objs.get(imageId, resolveImage)
      })

      expect(imageIndex).toBeGreaterThanOrEqual(0)
      expect(image?.data?.byteLength).toBeGreaterThan(0)
    } finally {
      await loadingTask.destroy()
    }
  })
})
