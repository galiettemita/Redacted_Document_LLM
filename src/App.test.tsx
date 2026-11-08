import { render, screen } from '@testing-library/react'
import { App } from './App'

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
