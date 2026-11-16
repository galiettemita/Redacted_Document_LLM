# Non-destructive fictional demo and clean prediction UI

## Intent

Keep a polished fictional demonstration available at any time for presentations while the uploaded-document workflow continues toward real predictions using Andrew's paper. Opening or closing the demo must never delete, reset, replace, or interrupt a real document.

## Goals

- Restore the bundled fictional PDF, deterministic passage geometry, progressive simulated predictions, mock service, and cancellation behavior.
- Put a clearly labeled **View fictional demo** control on both the upload screen and the real document workspace.
- Present the demo as a full-screen layer over the current application. Closing it returns to the exact underlying upload/workspace state.
- Label the demo itself **Fictional interactive demo** and every fake result **Simulated prediction**. No fake result may enter the real HTTP/SSE path or be labeled as a model result.
- Keep the real workspace simple. Remove customer-facing implementation language and metrics such as “visible-only context,” search queries, four-list retrieval, model names, readiness gates, mask counts, query terms, and word estimates.
- Preserve detailed readiness, scientific provenance, query metadata, and Andrew-method diagnostics in backend records and automated tests rather than the ordinary product interface.

## Non-goals

- The fictional demo does not call retrieval, the worker, or a paid model.
- This change does not weaken the real prediction readiness gates or substitute demo evidence for Andrew's corpus/index requirements.
- This change does not claim that the first real end-to-end model prediction has run.
- No deployment is included.

## Product behavior

### Real application

The ordinary sidebar uses short user-facing states only:

- **Preparing document…** while detection/context preparation runs.
- **Ready to predict** when passages can be scheduled.
- **Predictions unavailable** with one concise explanation when the real path is blocked.
- Prediction cards and the primary prediction action once real results exist.

The technical panel shown in the November 15 screenshot is removed from the customer-facing UI. Backend errors may remain actionable, but normal users do not see implementation details or research terminology.

### Fictional demo

The demo opens above the current screen and owns its own PDF, passages, selection state, and progressive prediction simulation. The underlying real workspace remains mounted and hidden from assistive technology while the modal layer is open, so uploads, detection state, and event-stream state are preserved.

The demo includes a persistent **Return to your document** control when a real document exists, or **Return to upload** when it was opened from the upload screen. Escape closes the demo, and focus returns to the button that opened it.

## Architecture and safety boundary

- Restore sample assets under `public/` and `src/data/` plus the testable mock prediction service/simulation hook.
- Implement a dedicated `FictionalDemoWorkspace` presentation component rather than adding sample branches back into the real `DocumentWorkspace` data path.
- `App` owns only the `demoOpen` presentation flag and renders the existing real screen continuously beneath the demo layer.
- The real workspace continues to use `createHttpPredictionService`; the demo alone receives `createMockPredictionService`.
- Tests prove the demo never calls backend document/retrieval/model APIs, closing it preserves the real workspace, labels remain unambiguous, and the removed technical copy cannot render in the real product UI.

## Approaches considered

1. **Non-destructive full-screen demo layer — selected.** Preserves real work and gives a presentation-ready experience from anywhere with a strict service boundary.
2. Restore the original replacement behavior. Simpler, but it deletes/replaces current work and conflicts with “available at any time.”
3. Add a separate `/demo` route. Shareable, but it adds routing/deep-link complexity and does not by itself preserve the current workspace.

## Delivery order

1. Restore the fictional assets and simulation behavior behind failing tests.
2. Add the non-destructive demo layer and persistent entry/return controls.
3. Replace the real workspace's research/debug panel with the concise product states above.
4. Run the complete frontend/backend verification, production bundle scan, and local browser QA if the in-app browser connector is available.
5. Update the checklist, commit, and push. Continue the separately gated real-prediction work without changing Andrew's method.
