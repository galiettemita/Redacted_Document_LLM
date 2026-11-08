# Redacted Document LLM Demo — Design

## Intent and why

Build a polished, approachable frontend-only research demo that helps a nontechnical visitor understand how an LLM might predict text hidden by redactions in government documents. The central experience is the visual connection between a redacted passage on a PDF page and one simulated prediction in a sidebar. The demo must be explicit that no real analysis occurs.

## Goals

- Begin with a focused PDF upload screen and move immediately into a document workspace after a valid selection.
- Render PDFs locally in the browser; never transmit uploaded files.
- Provide a complete automatic experience for a bundled fictional government-style PDF with known passage locations and predictions.
- Preserve passage identity when one passage spans multiple redaction rectangles.
- Make hover, pinning, card-to-document navigation, keyboard use, touch use, responsive switching, and selection dismissal feel coherent.
- Simulate progressive processing with loading, partial, complete, empty, and failed states.
- Let an arbitrary uploaded PDF demonstrate the interaction by allowing the visitor to draw one rectangle and receive a clearly labeled sample prediction.
- Separate presentation, PDF rendering, overlay state, mock data, and the replaceable prediction service so a future backend can replace simulation without rewriting the interface.

## Non-goals

- No backend, persistence, database, authentication, cloud upload, document retrieval, OCR, redaction detection, or LLM integration.
- No model selector, dashboards, charts, chat UI, marketing page, confidence scores, correctness claims, or verification badges.
- No claim that arbitrary uploaded files have been analyzed.
- No editing or removal of PDF redactions.

## Anti-goals

- A visually busy workspace that competes with the document.
- Interactions that obscure the original black redactions.
- Sidebar reordering on transient hover.
- PDF navigation triggered merely by hovering over a prediction card.
- Fabricated precision, accuracy, or automated-analysis language.

## Constraints

- React and TypeScript, scaffolded with Vite because the repository has no existing application.
- PDF.js through `react-pdf` for browser-local rendering and page controls.
- The existing `Redaction_RAG_NAACL_v3.pdf` remains untouched and is not presented as the fictional sample.
- A new bundled fictional PDF will be generated as a static app asset and visibly labeled as fictional.
- Passage geometry is stored in normalized page-relative coordinates (`x`, `y`, `width`, `height` in the 0–1 range).
- One configurable constant supplies the displayed model name, initially `Demo LLM`.
- Object URLs and simulation timers must be released when documents change or components unmount.
- Accessibility includes semantic controls, visible focus, keyboard selection/dismissal, adequate contrast, and equivalent touch behavior.

## Approaches considered

### Chosen: React + Vite + react-pdf/PDF.js

This provides controlled local rendering, page-aware overlays, zooming, fit-to-width behavior, and programmatic navigation with a moderate implementation surface. It best supports the passage-to-prediction interaction.

### Alternative: React + direct PDF.js canvas integration

This provides lower-level control but requires more rendering, cancellation, and canvas lifecycle code. It offers little prototype value beyond the chosen wrapper.

### Alternative: native browser PDF embedding

This minimizes code but does not expose reliable page geometry, scrolling, or overlay hooks across browsers. It cannot satisfy the core linked-highlight behavior.

## Information architecture and visual system

The upload view is a centered, single-task screen containing the app name, required heading and support copy, one large drop target, a `Choose PDF` button, and a restrained `Demo — predictions are simulated` notice. The initial screen has no sample action or competing navigation.

The workspace has a compact top toolbar, a document pane using roughly two-thirds of desktop width, and a sidebar using the remainder. The document pane uses white PDF pages on a soft gray surface. Text is near-black, borders are subtle, spacing is generous, and one muted blue accent represents focus and selection. Motion is limited to small fades and position transitions for progressively appearing cards and hover previews.

On narrow screens, a segmented `Document` / `Predictions` switch replaces the side-by-side layout. Prediction selection switches back to `Document` after navigating to the passage.

## Documents and upload behavior

The upload controller accepts a single `.pdf` file with a PDF MIME type or filename extension, rejects clearly invalid files with plain-language errors, and asks PDF.js to load the selected file so unreadable PDFs can be reported. A successful choice creates a local object URL and enters the uploaded-document workspace.

The workspace toolbar always exposes `Replace document`. Replacement returns to the upload view and clears document state, overlays, predictions, errors, selection, timers, and object URLs.

For uploaded PDFs, the sidebar states that automatic detection and prediction are not active yet. It offers `View interactive sample` as a secondary workspace action. The document pane enables a `Mark a passage` mode in which the visitor drags one normalized rectangle over a rendered page. Completing the rectangle creates a local passage with one clearly labeled sample prediction. A new rectangle replaces the prior demonstration mark to keep the prototype simple and honest.

The bundled sample can be entered only from the uploaded-document workspace through `View interactive sample`; it is not a competing action on the initial upload screen. It includes a visible `Fictional demonstration document` label.

## Fictional sample

The sample is a concise, government-style multi-page PDF with fictional agency branding, declassification-style metadata, body paragraphs, and several blacked-out passages. At least one passage uses multiple rectangles across consecutive lines. The PDF content, normalized coordinates, IDs, page numbers, and matching predictions are authored together so overlays align exactly.

The sample asset is produced during development and committed; runtime PDF generation is unnecessary. The content avoids real secrets, real people, or claims about actual government activity.

## Prediction lifecycle

A replaceable frontend `PredictionService` exposes document initialization and a progressive prediction stream. The mock implementation emits passage results in document order on short staggered timers, with deterministic outcomes suitable for repeatable tests. The sidebar displays `Predicting hidden passages…` while incomplete and makes completed cards usable immediately. It can also deterministically simulate an empty or failed state through service fixtures used by component tests, even though the main sample follows the successful partial-to-complete path.

Each prediction contains a stable passage ID, page, normalized rectangles, model label, complete paragraph text, and simulated status. No confidence or correctness metadata exists in the view model.

## Selection and navigation model

State distinguishes `hoveredPassageId` from `pinnedPassageId`. Effective selection is the hovered passage while a passage is hovered; otherwise it is the pinned passage.

- Hovering or focusing any rectangle highlights every rectangle for that passage, leaves black redactions visible, shows a compact in-viewer preview placed above or below the union of rectangles to avoid clipping, and temporarily fills the sidebar's dedicated `Selected passage` area.
- Leaving the passage restores the pinned selection. If nothing is pinned, the selected area returns to its empty prompt.
- Clicking, tapping, or pressing Enter/Space on a passage pins it. Another selection replaces it. A labeled dismiss control and Escape clear the pin.
- Clicking or keyboard-activating a prediction card navigates to its page, scrolls the passage union into view, pins it, focuses the corresponding overlay, and highlights all of its rectangles.
- Hovering a sidebar card may add a quiet card-local affordance but never navigates or changes document scroll.
- The active prediction is omitted from the ordered card list only while it is shown in `Selected passage`; all other completed results remain in stable document order.

Viewer previews are clamped within the rendered page/container bounds. Preview text is truncated; the full prediction remains available in the sidebar.

## Component and module boundaries

- `UploadScreen`: drag/drop, file picker, validation messaging.
- `DocumentWorkspace`: high-level layout, document replacement, responsive view switch.
- `PdfViewer` and `PdfToolbar`: PDF.js document/page rendering, page navigation, zoom, fit width, scrolling.
- `PassageOverlayLayer` and `PassagePreview`: normalized rectangles, drawing mode, hover/focus/click behavior.
- `PredictionSidebar`, `SelectedPassage`, and `PredictionCard`: processing state, stable list, selection controls, navigation requests.
- `sampleDocument`: fictional asset metadata and passage/prediction fixture.
- `predictionService`: interface plus timer-backed mock implementation.
- `usePredictionSimulation`: lifecycle cleanup and progressive result state.
- `config`: model display name and simulation timing.

Presentation components consume typed view models and callbacks. They do not import sample fixtures or own timers.

## States and error handling

- Upload idle, drag-active, validating/loading, invalid type, and unreadable PDF.
- PDF loading, rendered, and failed.
- Sample prediction pending, partially completed, completed, empty, and failed.
- Uploaded PDF before marking, drawing in progress, and sample mark completed.
- No selected passage, temporary hover/focus, and pinned selection.

Errors use concise inline copy with a recovery action. A PDF render error never implies that the file was uploaded anywhere.

## Testing and verification

- Unit/component tests cover validation, progressive simulation, cleanup, selection precedence, active-card omission, replacement reset, and normalized drawing geometry.
- Interaction tests cover passage-to-card pinning and card-to-document navigation.
- Production build and TypeScript checks must pass.
- Browser QA checks the upload flow, fictional sample flow, multi-rectangle selection, arbitrary-upload marking, zoom/fit alignment, replacement cleanup, keyboard focus, and responsive pane switching.

## Builder calls

- Use a restrained system sans-serif stack rather than adding a font dependency.
- Generate the fictional PDF with a small development script or existing local PDF tooling, then commit only the resulting asset if the generator is not useful to the running application.
- Use deterministic short delays so the full progressive sequence is observable without slowing QA.
- Keep the application local-first; deployment is not part of this implementation unless requested separately.

