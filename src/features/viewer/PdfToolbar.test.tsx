import { fireEvent, render, screen } from '@testing-library/react'
import { PdfToolbar } from './PdfToolbar'

describe('PdfToolbar', () => {
  it('keeps navigation and zoom within valid bounds', () => {
    const onPageChange = vi.fn()
    const onScaleChange = vi.fn()
    render(
      <PdfToolbar
        page={1}
        pageCount={3}
        scale={1}
        onPageChange={onPageChange}
        onScaleChange={onScaleChange}
        onFitWidth={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Next page' }))
    expect(onPageChange).toHaveBeenCalledWith(2)

    for (let index = 0; index < 10; index += 1) {
      fireEvent.click(screen.getByRole('button', { name: 'Zoom out' }))
      fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }))
    }
    expect(onScaleChange.mock.calls.every(([scale]) => scale >= 0.6 && scale <= 2)).toBe(true)
  })

  it('provides a dedicated fit-to-width action', () => {
    const onFitWidth = vi.fn()
    render(
      <PdfToolbar
        page={2}
        pageCount={3}
        scale={1.25}
        onPageChange={vi.fn()}
        onScaleChange={vi.fn()}
        onFitWidth={onFitWidth}
      />,
    )

    expect(screen.getByText('Page 2 of 3')).toBeInTheDocument()
    expect(screen.getByText('125%')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Fit to width' }))
    expect(onFitWidth).toHaveBeenCalledOnce()
  })
})
