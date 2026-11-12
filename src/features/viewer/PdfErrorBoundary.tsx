import { Component, type ReactNode } from 'react'

interface PdfErrorBoundaryProps {
  children: ReactNode
}

interface PdfErrorBoundaryState {
  failed: boolean
}

export class PdfErrorBoundary extends Component<
  PdfErrorBoundaryProps,
  PdfErrorBoundaryState
> {
  state: PdfErrorBoundaryState = { failed: false }

  static getDerivedStateFromError(): PdfErrorBoundaryState {
    return { failed: true }
  }

  render() {
    if (this.state.failed) {
      return (
        <p className="viewer-state is-error" role="alert">
          This PDF could not be displayed. The backend will still report whether it was accepted.
        </p>
      )
    }
    return this.props.children
  }
}
