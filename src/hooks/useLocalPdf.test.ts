import { act, renderHook } from '@testing-library/react'
import { PDFDocument } from 'pdf-lib'
import { useLocalPdf } from './useLocalPdf'

async function createPdfFile() {
  const document = await PDFDocument.create()
  document.addPage([300, 400])
  const bytes = new Uint8Array(await document.save())
  return new File([bytes.buffer], 'local.pdf', { type: 'application/pdf' })
}

describe('useLocalPdf', () => {
  const createObjectURL = vi.fn(() => 'blob:local-pdf')
  const revokeObjectURL = vi.fn()

  beforeEach(() => {
    vi.stubGlobal('URL', {
      createObjectURL,
      revokeObjectURL,
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('owns and revokes the local object URL when cleared', async () => {
    const { result } = renderHook(() => useLocalPdf())

    await act(async () => result.current.openFile(await createPdfFile()))
    expect(result.current).toMatchObject({ objectUrl: 'blob:local-pdf', status: 'ready', error: null })

    act(() => result.current.clear())
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:local-pdf')
    expect(result.current.objectUrl).toBeNull()
  })

  it('revokes the current local URL when its owner unmounts', async () => {
    const { result, unmount } = renderHook(() => useLocalPdf())
    await act(async () => result.current.openFile(await createPdfFile()))

    unmount()
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:local-pdf')
  })
})
