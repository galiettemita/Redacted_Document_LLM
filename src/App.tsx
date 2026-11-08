export function App() {
  return (
    <main className="upload-shell">
      <section className="upload-panel" aria-labelledby="upload-heading">
        <p className="brand">Redacted Document LLM</p>
        <div className="upload-copy">
          <h1 id="upload-heading">Explore what’s behind the redactions</h1>
          <p>Upload a redacted PDF to explore AI predictions for its hidden passages.</p>
        </div>
        <div className="drop-zone">
          <div className="document-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <p className="drop-title">Drop your redacted document here</p>
          <p className="drop-support">PDF files only · stays on your device</p>
          <button className="primary-button" type="button">Choose PDF</button>
        </div>
        <p className="demo-note"><span aria-hidden="true" />Demo — predictions are simulated</p>
      </section>
    </main>
  )
}
