# Project checklist

- [x] Read and reconcile the product brief.
- [x] Confirm the fictional sample strategy.
- [x] Select the React + Vite + PDF.js implementation approach.
- [x] Approve the written design.
- [x] Write the implementation plan.
- [x] Scaffold the React/TypeScript application.
- [x] Build and bundle the fictional sample PDF.
- [x] Implement upload validation and local PDF loading.
- [x] Implement the PDF viewer and controls.
- [x] Implement passage overlays, previews, and selection.
- [x] Implement progressive simulated predictions and sidebar states.
  - [x] Add deterministic progressive streaming and cancellation.
  - [x] Present processing and result states in the sidebar.
- [x] Implement uploaded-PDF rectangle marking.
- [x] Implement responsive and accessible behavior.
- [x] Add automated tests and complete available preview QA.
- [x] Commit and push the completed implementation.
- [x] Fix browser upload flow to validate each PDF only once.
- [x] Remove Safari-blocking PDF worker validation from the upload gate.
- [x] Remove Safari-blocking file byte reads from the upload gate.
- [x] Make the upload-to-workspace transition synchronous so it cannot remain in a checking state.
- [x] Pin React-PDF and its worker to the same PDF.js version.
- [x] Use the compatibility PDF.js build for Safari and older browser runtimes.
- [x] Audit upload, rendering, paging, zoom, marking, predictions, replacement, and responsive views in real browsers.
- [x] Remove on-document prediction previews and emphasize the selected sidebar prediction.
- [x] Complete an independent software-quality and production-readiness review against the written design.

## Real retrieval-assisted backend

- [x] Read Andrew Tang's paper and extract the realistic retrieval-assisted method.
- [x] Review the current frontend, passage-selection behavior, and prediction-service boundary.
- [x] Verify the existing frontend baseline (35 tests and TypeScript checks pass).
- [x] Identify the exact-method assets that are present and the corpus/code details that are still missing.
- [x] Draft and independently review the real-backend design.
- [ ] Approve the real-backend architecture and written design.
- [ ] Write the task-by-task implementation plan.
- [ ] Connect and verify one real backend feature at a time.
