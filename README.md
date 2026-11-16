# Redacted Document LLM

The repository contains two deliberately separate experiences: an always-available fictional interactive demo with simulated predictions, and a production prediction path from an uploaded PDF to passage-linked model output. The real path connects PDF upload, automatic redaction detection, stable correction, visible-date detection, document-wide context preparation, deterministic queries, durable Dramatiq execution, four-list retrieval, immutable trace persistence, GPT-6 Astra generation, resumable SSE, and real sidebar cards. It fails closed until the complete Andrew-method corpus indexes, a live worker, and separately approved model access are all ready; it never fabricates an uploaded-document result.

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

Start the stateful dependencies first:

```bash
docker compose up -d postgres redis
```

Build the backend image and apply migrations before starting the API. This order
ensures the retention service never runs against an older schema:

```bash
docker compose build backend
docker compose run --rm backend alembic upgrade head
docker compose up -d backend
```

Then open `http://127.0.0.1:8000/v1/readiness`. HTTP readiness is independent from database, worker, corpus, and model readiness, so prediction readiness remains false until every real dependency is connected. Docker publishes the database, Redis, and API on `127.0.0.1` only.

After complete corpus artifacts exist and the one-passage paid-call gate has been approved, start the retrieval profile:

```bash
REDACTED_CORPUS_ROOT=/absolute/path/to/corpus \
docker compose --profile retrieval up --build backend opensearch worker
```

`REDACTED_CORPUS_ROOT` must be the directory containing `manifest.json`, `bm25-manifest.json`, `dense-manifest.json`, and the `dense/` artifact directory; Compose mounts it read-only into the API and worker while OpenSearch keeps its own writable subdirectory there. The worker image installs the locked BGE/FAISS retrieval group; the web API image remains small. The worker publishes no port, and readiness stays false unless its expiring Redis heartbeat is current.

The upload screen sends PDFs to the local backend for validation and temporary content-addressed persistence while rendering a local object URL immediately. The backend scans every page, persists detected redactions under stable passage IDs, and prepares a separate hidden-answer-free context and `context-query-v1` retrieval input for every passage in one extraction/render pass. Structural target/redaction markers and the frozen stop-word list are removed from each query; query text, SHA-256, term count, query version, and stop-word version stay bound to the exact passage revision. The browser shows only simple preparation/prediction states and final provider results, not transcript, query, evidence, or research diagnostics.

Uploaded PDFs become inaccessible at their 24-hour deadline. A background
retention sweep marks expired records with the auditable `retention_expired`
reason, removes PDF-derived date/context/retrieval/prediction payloads, and
deletes content-addressed bytes only after the final live document reference is
gone. Physical deletion is durably recorded and retried after storage failures;
the default sweep interval is five minutes. User deletion is idempotent and
records `user_deleted`. Runtime settings cannot raise the frozen
PDF size, page, decompression, render-pixel, parser-timeout, or retention ceilings.
Normal Python logs globally redact structured PDF text, prompt/evidence fields,
authorization values, and OpenAI-style secret tokens.

Users can redraw, add, or remove passages, then choose **Predict passage** or **Predict all passages**. A blocked run displays the exact corpus, model, and worker gates. A ready run executes outside the request, resumes from durable query/retrieval checkpoints, filters replayed events to the newly scheduled jobs, and renders provider output as **Model prediction** with recorded token usage and estimated cost in the sidebar—never over the PDF. The worker uses an expiring Redis heartbeat and a PostgreSQL advisory lock per job. Provider results are stored on the owning job before the next stage advances; passage changes or deletion cancel active work before another external stage is allowed.

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

The BGE/FAISS/Torch packages are isolated in the locked `retrieval` dependency group. Normal development installs them through the `dev` group, while the loopback web API image deliberately excludes them and `Dockerfile.worker` installs them explicitly. This keeps the upload/API service small without replacing the real embedding implementation.

The prediction input path has the paper's four-list shape: visible context and model-produced own query are each sent independently to BM25 and dense retrieval; self-document candidates are removed before document-level reciprocal-rank fusion; and the top 100 documents retain at most two chunk-level RRF excerpts. The worker persists all four raw ranked lists, provenance, and fused evidence transactionally on the job without comparing BM25 scores to cosine scores. The browser exposes only the final passage-linked prediction, not the query or evidence text.

The two Appendix F prompts are frozen files. The model gateway is implemented against OpenAI's Responses API but is not called automatically: it fixes `gpt-6-astra`, medium reasoning, one user message, no system message or tools, `store: false`, disabled truncation, and a 4,096-token output reservation. It first uses the provider's exact [input-token count endpoint](https://developers.openai.com/api/docs/guides/token-counting), rejects an oversized complete evidence prompt without dropping documents, and records provider IDs, usage, prompt/request hashes, and a versioned price estimate. The gateway follows the [Responses create contract](https://developers.openai.com/api/reference/resources/responses/methods/create) and the frozen [GPT-6 Astra limits and pricing](https://developers.openai.com/api/docs/models/gpt-6-astra). No key has been requested and no model or token-count request has been sent.

## Run locally

1. Install Node.js 24 or a current compatible Node.js release.
2. In this repository, run `npm install`.
3. Start the backend and migrations as described above.
4. Run `npm run dev`.
5. Open `http://127.0.0.1:5173`.

Vite binds to loopback and proxies only `/v1` to the loopback backend. This local development setup is not a deployed or internet-facing service.

To create a production bundle, run `npm run build`. The output is written to `dist/`.

## Product flow

1. Choose **View fictional demo** from either the upload screen or an open real document. The demo is explicitly labeled **Fictional interactive demo**, and every invented result is labeled **Simulated prediction**.
2. Close the demo with its return button or Escape. Any uploaded document remains mounted and unchanged underneath; opening the demo never invokes replacement or deletion.
3. For the real path, upload a PDF. The backend scans every page, highlights stable detected passages, and prepares each one automatically while the sidebar shows only concise product states.
4. Review the highlights. Use **Correct selected area** to redraw one, **Add missed passage** for an omission, or **Remove selected passage** to confirm a false-positive removal.
5. Choose **Predict passage** or **Predict all passages**. If the real services are unavailable, the UI fails closed with a concise unavailable state and makes no model call. If ready, the worker streams provider results labeled **Model prediction**.

## Important limitations

- Visible-only context preparation runs automatically for all detected passages. Because the paper requires a date but does not prescribe a parser, `visible-date-v1` accepts only one agreeing ISO or English month-name date that occupies a complete line in the top 20% of page one. OCR date words must each have at least 0.75 confidence. It fails closed when that date is missing, ambiguous, low-confidence, inside any automatic or manually marked redaction, mask coverage is uncertain, OCR is unavailable, or nearby visible geometry cannot support a length cue. Browser-supplied dates, PDF metadata alone, corpus results, models, and hidden passages cannot supply the date.
- The production worker, retrieval, model boundary, and website control are connected, but live prediction readiness remains false until the complete corpus/index manifests, worker heartbeat, model key, and explicit paid-call approval all pass. The current loopback instance therefore creates a durable blocked job and makes no retrieval, token-count, or model request.
- History Lab's replacement public MCP was healthy on November 15, 2026 and reports the expected multi-million-document collections, but its capped hosted search tools are not a bulk export and cannot replace Andrew's locally reproduced BM25/BGE indexes. The older document API still returns HTTP 502, the inspected Hugging Face snapshot remains scientifically incomplete, and this Mac has 115 GiB free versus the capture preflight's 150 GiB minimum. Full source acquisition still needs a complete export path and additional storage.
- Appendix F prints `{hint}` but does not publish its substitution code or a generation token cap. The implementation records the evidence-supported rendering `It is {bucket} long.` (matching the paper's examples) and a 4,096-token provider reservation as versioned engineering choices rather than attributing either to Andrew's method.
- Stored documents become unavailable after at most 24 hours; the production retention scheduler and replacement flow retry unreferenced-byte cleanup and remove derived sensitive artifacts while retaining an auditable tombstone.
- The fictional demo is a dedicated presentation layer. Only `FictionalDemoWorkspace` imports its invented document and deterministic mock service; the real `DocumentWorkspace` remains connected exclusively to the backend HTTP/SSE prediction service.

## Project structure

- `src/features/upload/` — file validation and upload interaction.
- `src/features/viewer/` — PDF rendering, controls, overlays, and rectangle drawing.
- `src/features/predictions/` — selected passage area and ordered prediction cards.
- `src/features/demo/` — isolated fictional demo workspace.
- `src/features/workspace/` — integrated document state and responsive navigation.
- `src/services/` — typed backend transport, the real HTTP/SSE prediction adapter, and the demo-only mock adapter.
- `src/test/` — deterministic fixtures imported only by automated tests.
- `backend/src/redacted_backend/jobs/retention.py` — scheduled expiry, sensitive-artifact cleanup, and reference-safe storage deletion.
- `backend/src/redacted_backend/observability.py` — process-wide structured log redaction.

## Verification

Run:

```bash
npm test -- --run
npm run typecheck
npm run build
```

The automated suite covers upload errors, object-URL cleanup, production-only provider labels, normalized drawing, grouped multi-line passage selection, deterministic visible-date evidence and conflicts, revision-bound context/query/job persistence and invalidation, ordered/resumable analysis events, uncertain-request idempotency, structural-marker removal, safe retry/backfill/conflict behavior, PDF safety ceilings, log redaction, reference-safe retention, safe response fields, no prediction text over uploaded PDFs, sidebar ordering, replacement reset, and responsive prediction-to-document navigation.
