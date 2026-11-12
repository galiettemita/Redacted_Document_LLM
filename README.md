# Redacted Document LLM

The repository is being extended with a real, visible-only retrieval-assisted backend. PDF upload, stable manual passage persistence, and the visible-only extraction engine are implemented. Corpus retrieval and model prediction are still disconnected and remain visibly unavailable.

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

Start the local backend:

```bash
docker compose up --build backend
```

Apply the database migrations in a second terminal:

```bash
docker compose run --rm backend alembic upgrade head
```

Then open `http://127.0.0.1:8000/v1/readiness`. HTTP readiness is independent from database, worker, corpus, and model readiness, so prediction readiness remains false until every real dependency is connected. Docker publishes the database, Redis, and API on `127.0.0.1` only.

The upload screen now sends PDFs to the local backend for validation and temporary content-addressed persistence while rendering a local object URL immediately. Manually drawn rectangles are persisted with stable passage IDs. Replacing the document asks the backend to delete its stored copy. No uploaded-document prediction is simulated or generated.

## Run locally

1. Install Node.js 24 or a current compatible Node.js release.
2. In this repository, run `npm install`.
3. Start the backend and migrations as described above.
4. Run `npm run dev`.
5. Open `http://127.0.0.1:5173`.

Vite binds to loopback and proxies only `/v1` to the loopback backend. This local development setup is not a deployed or internet-facing service.

To create a production bundle, run `npm run build`. The output is written to `dist/`.

## Demo flow

1. Upload a PDF. The browser opens it immediately and the local backend validates and stores a temporary copy.
2. Wait for **PDF upload complete**, then mark one rectangle. The normalized geometry is saved under a stable backend passage ID.
3. Choose **View interactive sample** in the prediction sidebar to open the complete fictional three-page example.
4. Hover or focus a blacked-out passage to preview it, click or press Enter/Space to pin it, or activate a prediction card to navigate back to its passage.

## Important limitations

- The visible-only embedded-text/OCR/mask extraction engine is implemented and tested in the backend, but automatic extraction is not yet triggered from the website.
- Corpus retrieval and a real LLM are not connected. Uploaded documents receive no prediction text.
- Stored documents expire after 24 hours by configuration; replacement requests immediate deletion. Automated retention cleanup is a later hardening task.
- The bundled Northstar memorandum and all of its passage predictions are fictional.
- `Demo LLM` is used only by the fictional sample and is not an active model.
- Prediction timing, empty states, and failures are simulated only for the fictional sample.

## Project structure

- `src/features/upload/` — file validation and upload interaction.
- `src/features/viewer/` — PDF rendering, controls, overlays, and rectangle drawing.
- `src/features/predictions/` — selected passage area and ordered prediction cards.
- `src/features/workspace/` — integrated document state and responsive navigation.
- `src/services/` — typed backend transport plus the fictional sample's deterministic mock.
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
