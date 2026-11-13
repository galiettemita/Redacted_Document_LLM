# Redacted Document LLM

The repository is being extended with a real, visible-only retrieval-assisted backend. PDF upload, automatic all-page redaction detection, stable passage correction, automatic visible-date detection, document-wide visible-context preparation, deterministic context-query persistence, and durable prediction-readiness jobs are connected to the website. Corpus retrieval and model execution are not connected yet; the website reports that boundary instead of fabricating a result.

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

The upload screen now sends PDFs to the local backend for validation and temporary content-addressed persistence while rendering a local object URL immediately. The backend scans every page, persists detected redactions under stable passage IDs, and prepares a separate hidden-answer-free context and `context-query-v1` retrieval input for every passage in one extraction/render pass. Structural target/redaction markers and the frozen stop-word list are removed from each query; query text, SHA-256, term count, query version, and stop-word version stay bound to the exact passage revision. The browser receives only safe readiness summaries, not transcript, query, or evidence text. Users can redraw a selected passage, add a missed passage, or explicitly remove a false detection. Every mask-set change invalidates dependent contexts and automatically rebuilds the remaining set; removed detections are tombstoned so a rescan does not silently restore them. Durable revision-bound analysis jobs and resumable ordered events exist behind the readiness gate, but the website does not start retrieval while the corpus and worker are unavailable. Replacing the document asks the backend to delete its stored copy. No uploaded-document prediction is simulated or generated.

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

Andrew's removed “CIA reading-room files” bucket is mapped to History Lab's `briefing` collection: Appendix C explicitly retains CREST while removing the small 2019+ bucket, and History Lab describes `briefing` as the CIA Current/Central Intelligence Bulletin releases while its separate `cia` collection is CREST. The mapping and citations are hashed in [`backend/corpus/exclusions/v1.yaml`](backend/corpus/exclusions/v1.yaml). The policy remains unready only for full-capture FRUS verification. The coverage command requires a non-empty `cables` collection, no unknown collection or FRUS identifiers, exactly the ten 2019-2024 FRUS volumes reported by Appendix C, and a total within the rounding interval implied by the paper's 5.55M-file report. A different count is reported as a source-version gap, not silently accepted.

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

Create a newline-delimited recall sample from real visible-only queries, then build the pinned dense index:

```bash
uv run corpus build-dense \
  --chunks /absolute/path/to/corpus/chunks.jsonl \
  --corpus-manifest /absolute/path/to/corpus/manifest.json \
  --recall-queries /absolute/path/to/corpus/visible-recall-queries.txt \
  --output-directory /absolute/path/to/corpus/dense \
  --output-manifest /absolute/path/to/corpus/dense-manifest.json
```

The sparse index uses the standard Unicode tokenizer, a versioned lowercase stop-word list, no stemming, and frozen BM25 parameters `k1=1.2` and `b=0.75`. The dense index uses the immutable `BAAI/bge-base-en-v1.5` revision recorded in the scientific manifest, normalized 768-dimensional embeddings, float16 exact re-scoring, and FAISS HNSW candidates with `M=32`, `efConstruction=200`, and `efSearch=512`. Both searches are capped at the paper's exact top 300 and record query, corpus, and index provenance. Dense build and readiness fail unless measured `recall@300` is at least `0.98`. Do not treat the service as corpus-ready unless `/v1/readiness` verifies matching corpus, BM25, and dense manifests.

The BGE/FAISS/Torch packages are isolated in the locked `retrieval` dependency group. Normal development installs them through the `dev` group, while the loopback web API image deliberately excludes them; the future retrieval worker image will install `retrieval` explicitly. This keeps the upload/API service small without replacing the real embedding implementation.

The prediction input path now has the paper's four-list shape: visible context and model-produced own query are each sent independently to BM25 and dense retrieval; self-document candidates are removed before document-level reciprocal-rank fusion; and the top 100 documents retain at most two chunk-level RRF excerpts. The immutable trace boundary preserves all four raw ranked lists and provenance without comparing BM25 scores to cosine scores. It is not yet exposed as a prediction because the complete source is unavailable, traces are not yet attached to durable analysis jobs, and no paid model call has been approved.

The two Appendix F prompts are frozen files. The model gateway is implemented against OpenAI's Responses API but is not called automatically: it fixes `gpt-6-astra`, medium reasoning, one user message, no system message or tools, `store: false`, disabled truncation, and a 4,096-token output reservation. It first uses the provider's exact [input-token count endpoint](https://developers.openai.com/api/docs/guides/token-counting), rejects an oversized complete evidence prompt without dropping documents, and records provider IDs, usage, prompt/request hashes, and a versioned price estimate. The gateway follows the [Responses create contract](https://developers.openai.com/api/reference/resources/responses/methods/create) and the frozen [GPT-6 Astra limits and pricing](https://developers.openai.com/api/docs/models/gpt-6-astra). No key has been requested and no model or token-count request has been sent.

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
2. The backend scans every page automatically, groups multi-line redactions, and displays each detected passage with a stable PDF highlight.
3. It automatically detects the date from uncovered visible text in the first-page header and prepares a separate visible-only context and deterministic search query for every passage. The selected passage summary shows the date, geometry-derived length cue, estimated words, visible-word count, confirmed-mask count, and visible-query-term count. Transcript and query text are not exposed as predictions or placed over the PDF.
4. Review the highlights. Use **Correct selected area** to redraw one, **Add missed passage** for an omission, or **Remove selected passage** to confirm a false-positive removal. Remaining contexts rebuild automatically after every correction.
5. Hover, focus, or select a passage to link its PDF geometry with the selected-passage area in the sidebar. Choose **View interactive sample** only if you want to inspect the separate fictional prediction demonstration.

## Important limitations

- Visible-only context preparation runs automatically for all detected passages. Because the paper requires a date but does not prescribe a parser, `visible-date-v1` accepts only one agreeing ISO or English month-name date that occupies a complete line in the top 20% of page one. OCR date words must each have at least 0.75 confidence. It fails closed when that date is missing, ambiguous, low-confidence, inside any automatic or manually marked redaction, mask coverage is uncertain, OCR is unavailable, or nearby visible geometry cannot support a length cue. Browser-supplied dates, PDF metadata alone, corpus results, models, and hidden passages cannot supply the date.
- Context queries and durable analysis orchestration are implemented, but the worker is deliberately gated before search or generation. Corpus retrieval and a real LLM are not connected to uploaded passages, so uploaded documents receive no evidence or prediction text and the website does not expose a misleading prediction control yet.
- The deterministic corpus, BM25, and pinned BGE/FAISS tooling and the CIA/FRUS exclusion provenance are implemented, but the full History Lab capture and real indexes cannot be completed while the retired API and the replacement public MCP corpus backend return HTTP 502 (verified November 12, 2026). The incomplete Hugging Face export is not silently substituted.
- Appendix F prints `{hint}` but does not publish its substitution code or a generation token cap. The implementation records the evidence-supported rendering `It is {bucket} long.` (matching the paper's examples) and a 4,096-token provider reservation as versioned engineering choices rather than attributing either to Andrew's method.
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

The automated suite covers upload errors, object-URL cleanup, simulation cancellation, normalized drawing, grouped multi-line passage selection, deterministic visible-date evidence and conflicts, revision-bound context/query/job persistence and invalidation, ordered/resumable analysis events, uncertain-request idempotency, structural-marker removal, safe retry/backfill/conflict behavior, safe response fields, no prediction text over uploaded PDFs, sidebar ordering, replacement reset, and responsive prediction-to-document navigation.
