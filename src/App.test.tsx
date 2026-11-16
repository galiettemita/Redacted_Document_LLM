import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { App } from './App'

const { validatePdfFileMock } = vi.hoisted(() => ({
  validatePdfFileMock: vi.fn().mockReturnValue({ ok: true }),
}))

vi.mock('./features/upload/fileValidation', () => ({
  validatePdfFile: validatePdfFileMock,
}))

vi.mock('./features/viewer/PdfViewer', () => ({
  PdfViewer: () => <div aria-label="Local PDF renderer" />,
}))

describe('upload screen', () => {
  it('keeps the initial experience focused on choosing a PDF', () => {
    render(<App />)

    expect(screen.getByText('Redacted Document LLM')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Explore what’s behind the redactions' })).toBeInTheDocument()
    expect(
      screen.getByText('Upload a redacted PDF to explore AI predictions for its hidden passages.'),
    ).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Choose PDF' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View fictional demo' })).toBeInTheDocument()
    expect(screen.queryByText('Simulated prediction')).not.toBeInTheDocument()
  })

  it('opens and closes the fictional demo without replacing the upload screen', () => {
    render(<App />)

    const opener = screen.getByRole('button', { name: 'View fictional demo' })
    fireEvent.click(opener)

    expect(screen.getByRole('dialog', { name: 'Fictional interactive demo' })).toBeInTheDocument()
    expect(document.querySelector('.app-surface')).toHaveAttribute('aria-hidden', 'true')
    expect(screen.getByRole('button', { name: 'Return to upload' })).toHaveFocus()

    fireEvent.click(screen.getByRole('button', { name: 'Return to upload' }))

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Explore what’s behind the redactions' })).toBeInTheDocument()
    expect(opener).toHaveFocus()
  })
})

describe('document lifecycle', () => {
  const createObjectURL = vi.fn(() => 'blob:uploaded-pdf')
  const revokeObjectURL = vi.fn()
  const fetchMock = vi.fn<typeof fetch>()

  beforeEach(() => {
    validatePdfFileMock.mockClear()
    validatePdfFileMock.mockReturnValue({ ok: true })
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjectURL })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjectURL })
    vi.stubGlobal('fetch', fetchMock)
    fetchMock.mockImplementation((_input, init) => {
      if (init?.method === 'DELETE') return Promise.resolve(new Response(null, { status: 204 }))
      return Promise.resolve(new Response(JSON.stringify({
        id: '11111111-1111-4111-8111-111111111111',
        filename: 'local-record.pdf',
        sha256: 'a'.repeat(64),
        media_type: 'application/pdf',
        page_count: 1,
        status: 'uploaded',
        storage_key: 'sha256/aa/aaaaaaaa',
        visible_date: null,
        date_provenance: null,
        visible_date_evidence: null,
        created_at: '2026-11-11T12:00:00Z',
        expires_at: '2026-11-12T12:00:00Z',
      }), { status: 201, headers: { 'Content-Type': 'application/json' } }))
    })
  })

  afterEach(() => vi.unstubAllGlobals())

  it('opens a valid local PDF and replaces it without retaining workspace state', async () => {
    const file = new File(['%PDF-1.7'], 'local-record.pdf', { type: 'application/pdf' })
    render(<App />)

    fireEvent.change(screen.getByLabelText('Choose PDF'), { target: { files: [file] } })
    expect(screen.queryByText('Checking PDF…')).not.toBeInTheDocument()
    expect(await screen.findByText('local-record.pdf')).toBeInTheDocument()
    expect(validatePdfFileMock).toHaveBeenCalledOnce()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Mark a passage' })).toBeEnabled())

    fireEvent.click(screen.getByRole('button', { name: 'Replace document' }))
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Explore what’s behind the redactions' })).toBeInTheDocument())
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:uploaded-pdf')
  })

  it('opens the local viewer while the backend upload is still pending', () => {
    fetchMock.mockImplementation(() => new Promise<Response>(() => undefined))
    const file = new File(['%PDF-1.7'], 'pending-record.pdf', { type: 'application/pdf' })
    render(<App />)

    fireEvent.change(screen.getByLabelText('Choose PDF'), { target: { files: [file] } })

    expect(screen.getByText('pending-record.pdf')).toBeInTheDocument()
    expect(screen.getByLabelText('Local PDF renderer')).toBeInTheDocument()
    expect(screen.getAllByText(/Uploading to the loopback-only local backend/)).not.toHaveLength(0)
  })

  it('keeps the uploaded document mounted and never deletes it while the demo is open', async () => {
    const file = new File(['%PDF-1.7'], 'preserved-record.pdf', { type: 'application/pdf' })
    render(<App />)
    fireEvent.change(screen.getByLabelText('Choose PDF'), { target: { files: [file] } })
    await waitFor(() => expect(screen.getByRole('button', { name: 'Mark a passage' })).toBeEnabled())

    const opener = screen.getByRole('button', { name: 'View fictional demo' })
    fireEvent.click(screen.getByRole('button', { name: 'Mark a passage' }))
    expect(screen.getByText('Draw the passage area')).toBeInTheDocument()
    fireEvent.click(opener)

    const appSurface = document.querySelector('.app-surface')
    expect(appSurface).toHaveTextContent('preserved-record.pdf')
    expect(appSurface).toHaveAttribute('aria-hidden', 'true')
    expect(screen.getByRole('button', { name: 'Return to your document' })).toBeInTheDocument()
    expect(fetchMock.mock.calls.filter(([, init]) => init?.method === 'DELETE')).toHaveLength(0)

    fireEvent.keyDown(window, { key: 'Escape' })

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByText('preserved-record.pdf')).toBeInTheDocument()
    expect(screen.getByText('Draw the passage area')).toBeInTheDocument()
    expect(opener).toHaveFocus()
    expect(fetchMock.mock.calls.filter(([, init]) => init?.method === 'DELETE')).toHaveLength(0)
  })

  it('keeps the workspace open when backend deletion fails', async () => {
    const file = new File(['%PDF-1.7'], 'retained-record.pdf', { type: 'application/pdf' })
    render(<App />)
    fireEvent.change(screen.getByLabelText('Choose PDF'), { target: { files: [file] } })
    await waitFor(() => expect(screen.getByRole('button', { name: 'Mark a passage' })).toBeEnabled())
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      detail: { code: 'delete_failed', message: 'The stored PDF could not be removed.' },
    }), { status: 500, headers: { 'Content-Type': 'application/json' } }))

    fireEvent.click(screen.getByRole('button', { name: 'Replace document' }))

    expect(
      await screen.findAllByText('The stored PDF could not be removed.'),
    ).not.toHaveLength(0)
    expect(screen.getByText('retained-record.pdf')).toBeInTheDocument()
    expect(revokeObjectURL).not.toHaveBeenCalledWith('blob:uploaded-pdf')
  })
})
