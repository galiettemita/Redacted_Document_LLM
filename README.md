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

## Corpus preparation

The backend includes a resumable corpus capture, fail-closed coverage gate, deterministic chunker, and BM25 retriever for the paper's public-document collection. The authenticated `HistoryLab/foiarchive` Hugging Face snapshot is retained as a checksummed audit artifact, but inspection proved it is not the complete paper corpus: it lacks the separate `cables` collection, and its `source` field contains origin URLs or null rather than the archive collection labels required for Andrew's exclusions. It is therefore never accepted as the finished retrieval source.

The finished source must be a frozen local capture of the public History Lab document API. The capture stores collection labels separately from origin URLs, uses crash-safe keyset pagination, verifies exact counts before and after every collection, and checksums every collection file. Interrupted work resumes after the last committed document without duplicating evidence.

Run the read-only storage preflight before acquiring any corpus files:

```bash
cd backend
uv run corpus preflight --path /absolute/path/to/external/volume
```

The minimum is 150 GiB free and 250 GiB is recommended. An insufficient result exits with status 2 and does not create corpus data.

Run this command on the host against the exact intended storage path. A Docker Desktop container's virtual filesystem capacity is not evidence that the Mac has enough physical free space.

Run the preflight again immediately before a large capture or index build. Source JSONL, checkpoints, chunks, and indexes remain outside Git. A capture that is interrupted or sees the upstream count change stays visibly incomplete and cannot pass readiness.

Freeze a downloaded copy of the Office of the Historian's official FRUS metadata:

```bash
cd backend
uv run corpus freeze-frus \
  --input /absolute/path/to/frus-metadata.xml \
  --output /absolute/path/to/frus-publication-manifest.json \
  --retrieved-at 2026-11-12T00:00:00-05:00
```

When `https://api.foiarchive.org/corpora` is healthy, start or resume the complete collection-aware capture:

```bash
cd backend
uv run corpus capture-api \
  --output-root /absolute/path/to/corpus/historylab-api
```

The public source still needs an evidence-backed mapping from Andrew's “CIA reading-room files” bucket to one exact API collection. [`backend/corpus/exclusions/v1.yaml`](backend/corpus/exclusions/v1.yaml) remains deliberately unready until that mapping and the full capture are verified. The coverage command also requires a non-empty `cables` collection, no unknown collection or FRUS identifiers, and exactly the ten 2019-2024 FRUS volumes reported by Appendix C. A different count is reported as a source-version gap, not silently accepted.

After the capture and exclusion manifest are complete, create the coverage report and deterministic chunks:

```bash
cd backend
uv run corpus validate-capture \
  --capture-root /absolute/path/to/corpus/historylab-api \
  --capture-manifest /absolute/path/to/corpus/historylab-api/capture-manifest.json \
  --exclusions corpus/exclusions/v1.yaml \
  --frus-manifest /absolute/path/to/frus-publication-manifest.json \
  --output /absolute/path/to/corpus/coverage-report.json

uv run corpus build-chunks \
  --capture-root /absolute/path/to/corpus/historylab-api \
  --capture-manifest /absolute/path/to/corpus/historylab-api/capture-manifest.json \
  --coverage-report /absolute/path/to/corpus/coverage-report.json \
  --exclusions corpus/exclusions/v1.yaml \
  --output /absolute/path/to/corpus/chunks.jsonl \
  --manifest /absolute/path/to/corpus/manifest.json
```

Only after that coverage report is ready may the pinned OpenSearch 3.9.0 profile be started and the real index built:

```bash
REDACTED_CORPUS_ROOT=/absolute/path/to/corpus docker compose --profile retrieval up -d opensearch
cd backend
uv run corpus build-bm25 \
  --chunks /absolute/path/to/corpus/chunks.jsonl \
  --corpus-manifest /absolute/path/to/corpus/manifest.json \
  --output-manifest /absolute/path/to/corpus/bm25-manifest.json
```

The index uses the standard Unicode tokenizer, a versioned lowercase stop-word list, no stemming, and frozen BM25 parameters `k1=1.2` and `b=0.75`. Search is capped at the paper's exact top 300 and records the query, corpus, and index provenance. Do not treat the service as corpus-ready unless `/v1/readiness` verifies matching corpus and BM25 manifests.

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
- Corpus retrieval and a real LLM are not connected to uploaded passages. Uploaded documents receive no prediction text.
- The deterministic corpus and BM25 tooling is implemented, but the full real index has not been built and the required CIA/FRUS provenance maps are not yet populated.
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
