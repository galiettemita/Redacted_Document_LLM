import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { UploadScreen } from './UploadScreen'

describe('UploadScreen', () => {
  it('shows drag feedback and an understandable invalid-file error', async () => {
    render(<UploadScreen onFileAccepted={vi.fn()} />)
    const dropZone = screen.getByTestId('pdf-drop-zone')

    fireEvent.dragEnter(dropZone, { dataTransfer: { files: [] } })
    expect(dropZone).toHaveAttribute('data-drag-active', 'true')

    fireEvent.drop(dropZone, {
      dataTransfer: { files: [new File(['hello'], 'notes.txt', { type: 'text/plain' })] },
    })
    expect(await screen.findByRole('alert')).toHaveTextContent('Choose a PDF file to continue.')
    expect(dropZone).toHaveAttribute('data-drag-active', 'false')
  })

  it('passes an accepted PDF to the workspace owner', async () => {
    const file = new File(['%PDF-1.4\n'], 'record.pdf', { type: 'application/pdf' })
    const onFileAccepted = vi.fn()
    const validate = vi.fn().mockResolvedValue({ ok: true })
    render(<UploadScreen onFileAccepted={onFileAccepted} validate={validate} />)

    fireEvent.change(screen.getByLabelText('Choose PDF'), { target: { files: [file] } })

    await waitFor(() => expect(onFileAccepted).toHaveBeenCalledWith(file))
  })
})
