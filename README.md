# Redacted Document LLM

The repository is being extended with a real, visible-only retrieval-assisted backend. The current backend foundation exposes readiness only; PDF upload, document processing, corpus retrieval, and model prediction are connected incrementally and must remain visibly unavailable until their corresponding implementation task is complete.

## Backend foundation

Requirements:

- Docker Desktop with Docker Compose.
- `uv` for running Python tests directly.

Run the backend tests:

```bash
cd backend
uv run pytest
```

Validate the local service definition:

```bash
docker compose config
```

Start the readiness foundation:

```bash
docker compose up --build backend
```

Then open `http://127.0.0.1:8000/v1/readiness`. HTTP readiness is independent from database, worker, corpus, and model readiness, so prediction readiness remains false until every real dependency is connected.

Until the incremental backend stages are connected, the current user-facing flow remains a frontend demonstration for exploring how an LLM might predict hidden passages in redacted documents. It renders PDFs locally in the browser and uses fictional, simulated predictions throughout; the new readiness endpoint does not yet receive uploads or produce predictions.

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

- The readiness-only backend does not yet connect uploads, database persistence, OCR, retrieval, redaction detection, or a real LLM to the user-facing flow.
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
