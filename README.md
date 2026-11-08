# Redacted Document LLM

A frontend-only demonstration for exploring how an LLM might predict hidden passages in redacted documents. The app renders PDFs locally in the browser and uses fictional, simulated predictions throughout.

## Run locally

1. Install Node.js 24 or a current compatible Node.js release.
2. In this repository, run `npm install`.
3. Run `npm run dev`.
4. Open the local URL printed by Vite.

To create a production bundle, run `npm run build`. The output is written to `dist/`.

## Demo flow

1. Upload a PDF. The file stays in the browser and is never sent to a service.
2. Mark one rectangle to preview the passage-to-prediction interaction on that file.
3. Choose **View interactive sample** in the prediction sidebar to open the complete fictional three-page example.
4. Hover or focus a blacked-out passage to preview it, click or press Enter/Space to pin it, or activate a prediction card to navigate back to its passage.

## Important limitations

- There is no backend, database, account system, OCR, retrieval pipeline, redaction detector, or real LLM.
- Arbitrary uploads are displayed locally but are not analyzed.
- The bundled Northstar memorandum and all of its passage predictions are fictional.
- `Demo LLM` is a configurable display label, not an active model.
- Prediction timing, empty states, and failures are simulated by a replaceable frontend service.

## Project structure

- `src/features/upload/` — file validation and upload interaction.
- `src/features/viewer/` — PDF rendering, controls, overlays, and rectangle drawing.
- `src/features/predictions/` — selected passage area and ordered prediction cards.
- `src/features/workspace/` — integrated document state and responsive navigation.
- `src/services/` — replaceable prediction service and deterministic mock.
- `src/data/` — fictional passage coordinates and prediction fixtures.
- `scripts/generate-sample-pdf.mjs` — regenerates the bundled fictional PDF from shared geometry.

## Verification

Run:

```bash
npm test -- --run
npm run typecheck
npm run build
```

The automated suite covers upload errors, object-URL cleanup, simulation cancellation, normalized drawing, grouped multi-line passage selection, sidebar ordering, replacement reset, and responsive prediction-to-document navigation.
