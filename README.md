# Redacted Document LLM

The repository contains two deliberately separate experiences: an always-available fictional interactive demo with simulated predictions, and a production prediction path from an uploaded PDF to passage-linked model output. The real path connects PDF upload, automatic redaction detection, stable correction, visible-date detection, document-wide context preparation, durable Dramatiq execution, GPT-6 Astra generation, resumable SSE, and real sidebar cards. It can run Andrew Tang's frozen Appendix F closed-book prompt as an honestly labelled model-only prediction, or use the pinned best-available FOIArchive snapshot through Andrew-shaped BM25/BGE four-list retrieval. The future exact-paper corpus gate remains separate and fail closed. It never fabricates an uploaded-document result.

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

Build the API and real worker images and apply migrations before starting them. This order
ensures the retention service never runs against an older schema:

```bash
docker compose build backend worker
docker compose run --rm backend alembic upgrade head
docker compose up -d backend worker
```

Then open `http://127.0.0.1:8000/v1/readiness`. The real worker starts and publishes its Redis heartbeat even when corpus artifacts, OpenSearch, and model approval are absent. Its prediction runtimes are constructed lazily and cached separately by immutable prediction basis; there is no fallback or automatic basis change. HTTP, database, worker, corpus/retrieval, and model readiness remain separate. `prediction_bases` reports operational and approval state for each basis, `strongest_ready_basis` reports the basis a newly scheduled job would receive, and `prediction_ready` becomes true when at least one basis is both operational and separately approved. Docker's dependency-free liveness probe uses `/v1/health`, so an unavailable retrieval service cannot mark the HTTP process dead while `/v1/readiness` reports the closed gate. Docker publishes the database, Redis, and API on `127.0.0.1` only; the worker publishes no port.

The production model-only path is implemented but disabled by default. It requires the database, live worker heartbeat, configured model key, master paid-call approval, and separate model-only approval; it does not require corpus manifests, OpenSearch, BM25, BGE, or FAISS. Do not enable these values until the exact one-passage request has been counted, its expected and maximum cost have been shown, and one live generation has been explicitly approved:

```dotenv
REDACTED_OPENAI_API_KEY=your-real-key
REDACTED_ALLOW_PAID_MODEL_CALLS=true
REDACTED_ALLOW_MODEL_ONLY_CALLS=true
REDACTED_ALLOW_BEST_AVAILABLE_CALLS=false
REDACTED_ALLOW_PAPER_EXACT_CALLS=false
```

Keep the real key only in the untracked root `.env` file. Compose passes all five values to both the API and worker. After the paid-call gate is approved, rebuild/restart the two services with `docker compose up -d --build backend worker`, then confirm `/v1/readiness` reports `"strongest_ready_basis":"model_only_v1"` before selecting **Predict passage**. Enabling the two approval switches authorizes provider token-count and generation requests; it is not a no-spend verification step.

After complete corpus artifacts exist and the one-passage paid-call gate has been approved, start the retrieval profile:

```bash
REDACTED_CORPUS_ROOT=/absolute/path/to/corpus \
docker compose --profile retrieval up --build backend opensearch worker
```

`REDACTED_CORPUS_ROOT` is the parent artifact directory. Exact-paper artifacts retain their existing top-level paths. Best-available artifacts live under `best-available/` as `corpus-manifest.json`, `bm25-manifest.json`, `dense/manifest.json`, `dense/recall-queries.txt`, and the dense shard directories. Compose mounts the root read-only into the API and worker while OpenSearch keeps its own writable subdirectory there. The worker image installs the locked BGE/FAISS retrieval group; the web API image remains small. Each basis has independent paths, manifests, readiness, and approval. Best-available readiness verifies the frozen corpus identity, live BM25 count, all shard ranges/checksums, the 100-query certification hash, and `recall@300 >= 0.98`; it can never satisfy `paper_exact_v1`.

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

Users can redraw, add, or remove passages, then choose **Predict passage** or **Predict all passages**. A blocked run shows a concise unavailable state while the backend records the exact gates. A ready run executes outside the request, resumes from its durable basis-specific checkpoint, filters replayed events to the newly scheduled jobs, and renders provider output in the sidebar—never over the PDF. Model-only output is labelled **Model-only prediction — no archive evidence used**; best-available output is labelled **Archive-assisted prediction — available FOIArchive snapshot used**; exact-paper output is labelled **Paper-method prediction**. The worker uses an expiring Redis heartbeat and a PostgreSQL advisory lock per job. Provider results are stored on the owning job before completion; passage changes or deletion cancel active work before another external request is allowed.

## Corpus preparation

The backend includes a resumable corpus capture, fail-closed coverage gate, deterministic chunker, and BM25 retriever for the paper's public-document collection. The authenticated `HistoryLab/foiarchive` Hugging Face snapshot is retained as a checksummed audit artifact, but a full streaming audit proved it is not the complete paper corpus: it lacks the separate `cables` collection, and every parsed `source` value is an origin URL or null rather than an archive collection label required for Andrew's exclusions. It is therefore never accepted as the finished retrieval source.

The finished source must be either a frozen local capture of the public History Lab document API or an immutable official bulk export that passes the same coverage contract. The API capture stores collection labels separately from origin URLs, uses crash-safe keyset pagination, verifies exact counts before and after every collection, and checksums every collection file. Interrupted work resumes after the last committed document without duplicating evidence.

Run the read-only storage preflight before acquiring any corpus files:

```bash
cd backend
uv run corpus preflight --path /absolute/path/to/external/volume
```

The report retains the conservative 150 GiB minimum and 250 GiB recommendation, but capacity is advisory by default: an `insufficient` result still exits successfully and does not prevent source work. To make that estimate a hard gate for an unattended environment, add `--enforce-minimum`; only that explicit mode exits with status 2 below the minimum.

Run this command on the host against the exact intended storage path. A Docker Desktop container's virtual filesystem capacity is not evidence that the Mac has enough physical free space.

Run the preflight again immediately before a large capture or index build. Source JSONL, checkpoints, chunks, and indexes remain outside Git. A capture that is interrupted, encounters an actual out-of-space write, or sees the upstream count change stays visibly incomplete and cannot pass readiness; resume truncates bytes beyond the last fsynced checkpoint before continuing.

Audit the pinned official Hugging Face bundle without expanding it to a second raw file:

```bash
huggingface-cli download HistoryLab/foiarchive foiarchive-202409.json.gz \
  --repo-type dataset \
  --revision 411cf6f86b38342683e27148be167c6094d1a10c

cd backend
uv run corpus lock-snapshot \
  --repo-id HistoryLab/foiarchive \
  --revision 411cf6f86b38342683e27148be167c6094d1a10c \
  --output ~/.cache/redacted-document-llm/corpus-audit/historylab-411cf6f-snapshot-lock.json

uv run corpus audit-bulk-export \
  --source ~/.cache/huggingface/hub/datasets--HistoryLab--foiarchive/snapshots/411cf6f86b38342683e27148be167c6094d1a10c/foiarchive-202409.json.gz \
  --snapshot-lock ~/.cache/redacted-document-llm/corpus-audit/historylab-411cf6f-snapshot-lock.json \
  --catalog corpus/historylab-huggingface-2025-catalog.json \
  --exclusions corpus/exclusions/v1.yaml \
  --output ~/.cache/redacted-document-llm/corpus-audit/historylab-411cf6f-audit.json
```

The audit verifies the locked 7,976,348,471-byte SHA-256 before streaming gzip rows and writes its small report atomically outside Git. A ready source exits 0; a valid but scientifically incomplete source still writes its evidence and exits 2. The November 16, 2026 run parsed 4,889,089 rows, 60,782 fewer than the frozen card total of 4,949,871. All 4,889,089 rows had unresolved collection identity: 1,304,229 HTTP(S) origin URLs and 3,584,860 nulls, with zero blank or unknown values. Computed canonical collection counts were empty, and `cables` was absent. If a rerun fails before publication, atomic replacement intentionally preserves the preceding verified report; the command's exit status is authoritative, and each report identifies its source revision and SHA-256. The exact external requirement is recorded in [`docs/corpus/historylab-complete-export-request.md`](docs/corpus/historylab-complete-export-request.md); that request has not been sent.

### Best-available FOIArchive production path

The audited Hugging Face snapshot is now supported as an honest best-available production basis. It remains scientifically distinct from an exact reproduction; the four fixed limitations are recorded verbatim in [`4-bulletpoint-limitations.md`](4-bulletpoint-limitations.md). Every imported document uses the literal `historylab-unlabeled` corpus identity, and no missing collection label or publication date is inferred.

Keep the 7.98 GB gzip in its existing cache and link it into the ignored artifact workspace rather than copying it:

```bash
mkdir -p var/corpus/best-available
ln -s /absolute/path/to/foiarchive-202409.json.gz \
  var/corpus/best-available/foiarchive-202409.json.gz
```

The importer validates the pinned filename, byte count, SHA-256, snapshot, catalog, canonical audit, limitations hash, counts, and reason set before writing. It streams gzip rows, checkpoints fsynced output, detects duplicate document IDs on disk, and resumes without duplicating chunks:

```bash
cd backend
uv run corpus build-best-available-chunks \
  --source ../var/corpus/best-available/foiarchive-202409.json.gz \
  --snapshot-lock corpus/historylab-huggingface-2025-snapshot-lock.json \
  --catalog corpus/historylab-huggingface-2025-catalog.json \
  --audit corpus/historylab-huggingface-2025-audit.json \
  --limitations ../4-bulletpoint-limitations.md \
  --output ../var/corpus/best-available/chunks.jsonl \
  --checkpoint ../var/corpus/best-available/chunks-checkpoint.json \
  --manifest ../var/corpus/best-available/corpus-manifest.json
```

Start the pinned OpenSearch service, then build or resume the separately named BM25 index:

```bash
cd ..
REDACTED_CORPUS_ROOT=/absolute/path/to/repository/var/corpus \
  docker compose --profile retrieval up -d opensearch

cd backend
uv run corpus build-best-available-bm25 \
  --chunks ../var/corpus/best-available/chunks.jsonl \
  --corpus-manifest ../var/corpus/best-available/corpus-manifest.json \
  --output-manifest ../var/corpus/best-available/bm25-manifest.json \
  --checkpoint ../var/corpus/best-available/bm25-checkpoint.json \
  --opensearch-url http://127.0.0.1:59200
```

Build the bounded dense index. Each shard contains at most 1,000,000 consecutive chunks and publishes atomically. Certification combines the 50 frozen public application-shaped queries with 50 stable-hash title/opening-text queries from the verified corpus, scans the stored float16 embeddings exactly in bounded blocks, and refuses to publish the top manifest below `recall@300 = 0.98`:

```bash
uv run corpus build-best-available-dense \
  --chunks ../var/corpus/best-available/chunks.jsonl \
  --corpus-manifest ../var/corpus/best-available/corpus-manifest.json \
  --application-queries corpus/best_available_application_queries.txt \
  --certification-queries ../var/corpus/best-available/dense/recall-queries.txt \
  --output-directory ../var/corpus/best-available/dense \
  --output-manifest ../var/corpus/best-available/dense/manifest.json
```

With all provider approvals still false, restart the API and worker and check `/v1/readiness`. The expected safe state is `best_available_foiarchive_v1.operational=true`, `approved=false`, and `ready=false`. Do not enable best-available generation from the artifact build alone. First count the exact own-query request and the resulting evidence-bound prediction request, show their expected and maximum combined cost, and obtain a new explicit approval for that two-stage call. Only then set the master and best-available flags; keep the other basis flags false unless separately approved:

```dotenv
REDACTED_ALLOW_PAID_MODEL_CALLS=true
REDACTED_ALLOW_MODEL_ONLY_CALLS=false
REDACTED_ALLOW_BEST_AVAILABLE_CALLS=true
REDACTED_ALLOW_PAPER_EXACT_CALLS=false
```

Ordinary prediction cards show only the concise archive-assisted basis label and the provider result. Corpus audit details, queries, evidence, checkpoints, and research limitations remain in backend provenance and documentation rather than the document UI.

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

The Appendix F own-query, retrieval-assisted prediction, and closed-book prediction templates are frozen files. The model gateway is implemented against OpenAI's Responses API but is not called automatically: it fixes `gpt-6-astra`, medium reasoning, one user message, no system message or tools, `store: false`, disabled truncation, and a 4,096-token output reservation. It first uses the provider's exact [input-token count endpoint](https://developers.openai.com/api/docs/guides/token-counting), rejects an oversized request without silently removing evidence, and records provider IDs, usage, prompt/request hashes, immutable basis, and a versioned price estimate. The gateway follows the [Responses create contract](https://developers.openai.com/api/reference/resources/responses/methods/create) and the frozen [GPT-6 Astra limits and pricing](https://developers.openai.com/api/docs/models/gpt-6-astra). Exactly one separately approved model-only generation has been completed and recorded at $0.00591 actual cost; all paid-call approvals were closed again afterward. No retrieval-assisted provider call has been made.

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
5. Choose **Predict passage** or **Predict all passages**. If the real services are unavailable or unapproved, the UI fails closed with a concise unavailable state and makes no model call. A ready model-only result is labelled **Model-only prediction — no archive evidence used**. A ready best-available result runs the separate query → four-list BM25/BGE retrieval → fused evidence → prediction path and is labelled **Archive-assisted prediction — available FOIArchive snapshot used**.

## Important limitations

- Visible-only context preparation runs automatically for all detected passages. Because the paper requires a date but does not prescribe a parser, `visible-date-v1` accepts only one agreeing ISO or English month-name date that occupies a complete line in the top 20% of page one. OCR date words must each have at least 0.75 confidence. It fails closed when that date is missing, ambiguous, low-confidence, inside any automatic or manually marked redaction, mask coverage is uncertain, OCR is unavailable, or nearby visible geometry cannot support a length cue. Browser-supplied dates, PDF metadata alone, corpus results, models, and hidden passages cannot supply the date.
- The production worker is independently live in the default stack. Model-only readiness does not require corpus indexes, but it still requires the worker heartbeat, a model key, master paid-call approval, and separate model-only approval. The November 20, 2026 no-spend loopback completed upload → detection → visible context → queued `model_only_v1` job → SSE using an injected fake transport, with zero query or retrieval calls. The November 21 no-spend archive-assisted loopback completed upload → detection → visible context → `best_available_foiarchive_v1` own-query generation → four independent ranked lists → fused evidence → basis-bound prediction → SSE, using only injected fake transports. Exact-paper readiness still requires the future complete verified corpus indexes and its own approval; best-available cannot satisfy that gate.
- History Lab's replacement public MCP was healthy on November 15, 2026 and reports the expected multi-million-document collections, but its capped hosted search tools are not a bulk export and cannot replace Andrew's locally reproduced BM25/BGE indexes. The older document API still returns HTTP 502. The November 16 full audit of the immutable Hugging Face gzip verified its size and SHA-256, then found 4,889,089 unresolved rows, no canonical collection counts, and no `cables`; it remains scientifically incomplete. Full source acquisition still needs a complete enumerable export path.
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

The automated suite covers upload errors, object-URL cleanup, exact prediction-basis labels, normalized drawing, grouped multi-line passage selection, deterministic visible-date evidence and conflicts, revision-bound context/query/job persistence and invalidation, ordered/resumable analysis events, durable provider-attempt claims that prevent uncertain generations from being resent, structural-marker removal, safe retry/backfill/conflict behavior, PDF safety ceilings, log redaction, reference-safe retention, safe response fields, no prediction text over uploaded PDFs, sidebar ordering, replacement reset, responsive prediction-to-document navigation, the fake-transport model-only loopback with zero retrieval, and the fake-transport archive-assisted loopback with two provider stages and all four retrieval lists.
