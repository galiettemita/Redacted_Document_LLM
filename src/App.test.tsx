import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { App } from './App'

const { validatePdfFileMock } = vi.hoisted(() => ({
  validatePdfFileMock: vi.fn().mockResolvedValue({ ok: true }),
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
    expect(screen.getByText('Demo — predictions are simulated')).toBeInTheDocument()
    expect(screen.queryByText('View interactive sample')).not.toBeInTheDocument()
  })
})

describe('document lifecycle', () => {
  const createObjectURL = vi.fn(() => 'blob:uploaded-pdf')
  const revokeObjectURL = vi.fn()

  beforeEach(() => {
    validatePdfFileMock.mockClear()
    validatePdfFileMock.mockResolvedValue({ ok: true })
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjectURL })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjectURL })
  })

  it('opens a valid local PDF and replaces it without retaining workspace state', async () => {
    const file = new File(['%PDF-1.7'], 'local-record.pdf', { type: 'application/pdf' })
    render(<App />)

    fireEvent.change(screen.getByLabelText('Choose PDF'), { target: { files: [file] } })
    expect(await screen.findByText('local-record.pdf')).toBeInTheDocument()
    expect(validatePdfFileMock).toHaveBeenCalledOnce()
    expect(screen.getByText(/Automatic passage detection and predictions aren’t connected yet/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Replace document' }))
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Explore what’s behind the redactions' })).toBeInTheDocument())
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:uploaded-pdf')
  })
})
