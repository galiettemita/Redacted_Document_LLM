import { render, screen } from '@testing-library/react'
import { PdfErrorBoundary } from './PdfErrorBoundary'

function BrokenPdfRenderer(): never {
  throw new Error('Invalid PDF structure')
}

describe('PdfErrorBoundary', () => {
  it('contains an unexpected PDF renderer exception inside the viewer', () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    render(
      <PdfErrorBoundary>
        <BrokenPdfRenderer />
      </PdfErrorBoundary>,
    )

    expect(screen.getByRole('alert')).toHaveTextContent(
      'This PDF could not be displayed. The backend will still report whether it was accepted.',
    )
    consoleError.mockRestore()
  })
})
