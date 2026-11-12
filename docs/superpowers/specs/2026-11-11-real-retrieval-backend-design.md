# Real Retrieval-Assisted Backend Design

## Intent and why

Turn the existing frontend-only Redacted Document LLM demonstration into a real, auditable application that predicts passages hidden by visible redactions. The scientific contract is the realistic retrieval-assisted condition in Andrew Tang's supplied paper: the visible redacted document creates the search, the public-document collection supplies evidence, and that evidence plus the visible document produces one prediction for each target passage.

The application must never use a hidden answer to form a query, select evidence, or generate a prediction. It must not quietly replace the paper's disclosed method when an implementation detail is missing. For details the paper leaves open, this design makes explicit, versioned engineering choices that can later be replaced if the authors release their code.

## Goals

- Connect the current React/PDF.js frontend to a real document-processing and prediction backend.
- Preserve the current layout and linked passage behavior: hover highlights the passage and moves its prediction to the selected area at the top of the sidebar; prediction text never appears over the PDF.
- Build one visible-only example per target passage: the target is `[[TARGET]]`, other redactions are `[REDACTED]`, and the whole visible bulletin remains available.
- Reproduce the paper's realistic retrieval shape: two visible-only queries, BM25 and `bge-base-en-v1.5` for each query, four top-300 chunk lists, document-level reciprocal-rank fusion, top 100 documents, and two best chunks per document.
- Use the paper's verbatim own-query and generation prompts with GPT-6 Astra at medium reasoning effort.
- Keep stable passage IDs from page geometry through retrieval, generation, streaming, and sidebar display.
- Make every output auditable through immutable prompt, configuration, corpus, query, ranking, and model provenance.
- Deliver in independently testable vertical slices, committing and pushing after each accepted slice.

## Non-goals

- No oracle retrieval, hidden-answer lookup, later-release access, or manually selected evidence.
- No closed-book comparison, answer judge, correctness score, pass rate, cross-model comparison, or evaluation dashboard.
- No chat interface, model selector, confidence score, or prediction text placed over the document.
- No production deployment, user accounts, billing integration, or public hosting until separately requested and approved.
- No claim that arbitrary PDFs reproduce the paper's dataset preparation exactly. OCR and automatic box detection are an input adapter around the disclosed experiment.
- No fake prediction or dummy model in a user-visible finished path. Test doubles remain allowed in isolated automated tests.

## Anti-goals

- A fast demo that silently uses the public History Lab search API even though its ranking differs from the paper.
- A large rewrite of the working frontend.
- A single opaque `analyze()` function that makes provenance or leakage impossible to inspect.
- Starting paid model runs before one passage can be traced through local extraction and retrieval.
- Blocking the first real vertical slice on automatic detection of every redaction.

## Scientific invariants

1. Runtime request and domain types contain no hidden-answer field.
2. The query generator sees only the date, the whole visible transcript, the target marker, other-redaction markers, and a length cue derived from visible geometry.
3. Retrieval uses the corpus index and deterministic ranking code only. No human may add, remove, or reorder evidence for a passage.
4. The generator receives exactly the fused realistic documents for that passage, not oracle or cited documents.
5. The query and generation prompts are frozen files, each sent as one user message with no system prompt.
6. The configured model is exactly `gpt-6-astra`, reasoning effort `medium`; the application fails clearly rather than substituting another model.
7. Corpus snapshot, exclusion manifest, chunker, tokenizer, embedding model, ranking parameters, prompts, and model settings are hashed into every run.
8. A passage prediction always carries the same stable passage ID used by its PDF rectangles.

## Approaches considered

### Chosen: faithful disclosed-method reconstruction with explicit gaps

Build the pipeline from the paper and fill undisclosed implementation details with frozen, tested choices. This follows the user's instruction to use Andrew's ideas, methods, and design patterns without waiting indefinitely for private research code. Every filled gap is named in the run manifest so author-released settings can replace it later without changing API or UI contracts.

### Alternative: wait for the authors' exact repository and indexes

This would maximize reproduction fidelity but blocks useful product work on assets the supplied anonymous draft does not link. It is no longer the chosen path.

### Rejected: public History Lab search service as the final retriever

The service is useful for exploration, but its public client caps vector results at 100 and its lexical path uses PostgreSQL full-text search. That cannot supply the paper's four top-300 BM25/BGE lists, so using it as the finished retriever would change the method.

### Rejected: CALIBER or Assessment Workbench as the application foundation

Their strongest operational patterns are useful, but their FISC/CFPF claim-reconstruction and evaluation products answer different questions. Importing them would add comparison, grading, and release-review scope the user explicitly excluded.

## Patterns adopted from Andrew's work

- Dataset-specific input construction feeds a shared retrieval and prediction pipeline.
- The visible passage/example is the seam; shared code does not branch on a dataset name.
- Prompts and stage parameters are frozen, versioned inputs rather than strings hidden in application code.
- Large corpus indexes and model artifacts live outside Git with checksums and manifests.
- Dependencies point one way: frontend and dataset adapters depend on shared contracts; shared retrieval knows nothing about the frontend or dataset.
- A readiness/smoke check proves document processing, retrieval, and model access separately before a full run.
- Observed outputs are results, not proof of correctness or permission to release information.

## System shape

The repository remains one application with two top-level runtimes:

- `src/`: the existing React 19, TypeScript, Vite, and PDF.js frontend.
- `backend/`: a Python 3.12 package exposing FastAPI endpoints and background jobs.

PostgreSQL stores document, passage, job, prediction, and provenance records. Redis and Dramatiq run OCR, retrieval, and model work outside web requests. Uploaded PDFs and large intermediate artifacts use a storage interface whose first implementation is a local directory outside Git. Docker Compose supplies PostgreSQL, Redis, and the backend locally; the existing Vite process remains the frontend development server.

This is more infrastructure than an in-process demo, but it prevents long OCR/index/model work from dying with one browser request and gives cancellation and retries stable semantics. The first slice brings up only health/readiness, upload, and passage creation; later slices activate worker stages behind the same contracts.

## Domain contracts

### `DocumentRecord`

- `id`: opaque UUID
- `filename`, `sha256`, `media_type`, `page_count`
- `status`: `uploaded | extracting | ready | failed | deleted`
- `created_at`, `expires_at`
- storage key, never an arbitrary filesystem path

### `PassageRecord`

- `id`: stable UUID
- `document_id`, human label, one-based page number
- one or more normalized rectangles in the current frontend's 0-1 coordinate system
- `source`: `manual | detected`
- `status`: `marked | context_ready | retrieving | predicting | complete | failed`

### `VisibleExample`

- `passage_id`, document date when visible or extracted
- whole visible transcript split into `before_target` and `after_target`
- paragraph window used for the context query
- other redactions represented only as `[REDACTED]`
- target represented only as `[[TARGET]]`
- coarse length bucket and a record of the visible geometry calculation

There is deliberately no `answer`, `reveal`, `ground_truth`, or later-release field.

### `RetrievalTrace`

- context and model-written queries
- four ranked chunk lists, preserving source rank and score
- document aliases and merges
- self-document exclusions and their reason
- fused top 100 documents and selected two chunks
- corpus and retrieval configuration hashes

### `PredictionRecord`

- `passage_id`, prediction text, model ID, reasoning effort
- prompt version and response ID
- retrieval-trace ID
- status, timestamps, token usage, and estimated/billed cost when returned by the provider

## API and event flow

- `GET /v1/readiness`: reports database, worker, corpus, and model configuration separately. It never claims prediction readiness when only the web server is alive.
- `POST /v1/documents`: validates and stores one PDF, then returns a document record.
- `POST /v1/documents/{document_id}/passages`: creates a manual passage from existing frontend rectangles.
- `PATCH /v1/passages/{passage_id}`: corrects a label or rectangles, increments the passage revision, and invalidates every derived artifact.
- `DELETE /v1/passages/{passage_id}`: cancels work and tombstones a passage.
- `POST /v1/passages:merge`: merges selected passages; the first passage in document order keeps its UUID and the others record `merged_into`.
- `POST /v1/passages/{passage_id}:split`: keeps the original UUID for the first resulting passage and creates new UUIDs with `split_from` provenance for the rest.
- `POST /v1/documents/{document_id}/passages:detect`: starts automatic OCR/box detection when that slice is enabled.
- `POST /v1/passages/{passage_id}:analyze`: idempotently advances one passage through context, retrieval, and prediction.
- `POST /v1/documents/{document_id}:analyze`: schedules every ready passage with bounded concurrency.
- `GET /v1/documents/{document_id}/events`: server-sent events for passage status and predictions.
- `DELETE /v1/documents/{document_id}`: cancels jobs and removes transient document artifacts through the storage interface.

Idempotency keys prevent a double click or reconnect from paying for the same stage twice. Jobs checkpoint after context construction, retrieval, and generation. A failed stage may retry without discarding earlier deterministic artifacts.

## Document processing

### Upload

The browser continues to use its object URL for immediate PDF viewing, while sending the same file to the local backend for analysis. The frontend copy must stop claiming that the file never leaves the browser. Before deployment, the copy and retention behavior will be reviewed again for the actual hosting boundary.

The backend validates size, `%PDF-` signature, parser readability, page count, and decompression limits. The initial local limit is 100 MiB and 500 pages; both values are configuration, tested at the boundary, and visible in readiness metadata.

### Visible text and layout

PyMuPDF first extracts any usable embedded text and page geometry. Pages without adequate text coverage are rendered and passed through Tesseract OCR with word-level boxes. OpenCV finds high-contrast filled rectangles and blank/covered spans. The text layer and OCR words are normalized into a page-layout model while retaining their original boxes. Any embedded-text word whose box intersects a visible redaction candidate is suppressed before transcript construction, so an invisible PDF text layer cannot leak words painted over by a redaction.

The first vertical slice does not wait for fully automatic passage detection: the existing `Mark a passage` button supplies target geometry. It does include early candidate-mask discovery across the document so every other visible box can become `[REDACTED]`. Before retrieval, the user may add, delete, or correct visible mask geometry; that is document transcription, not evidence selection. The backend refuses to build a visible example while any page is marked as having uncertain redaction coverage. Automatic grouping and one-click whole-document detection arrive after one-passage retrieval and generation work end to end.

### Passage grouping

Touching, overlapping, or line-wrapped rectangles with no readable word between them become one passage, matching the paper's passage rule. Readable words between rectangles split passages. Cross-column rectangles never merge. The grouping algorithm returns stable ordering by page, column, and reading order.

### Canonical transcript serialization

The serializer is frozen and versioned. It processes pages in page-number order, removes repeated running headers/footers, marginal release stamps, page numbers, and administrative markings, then uses recursive XY-cut column segmentation. Columns read left to right; blocks, lines, and words read top to bottom within a column. Words on one line join with one space. Paragraphs join with two newlines based on block membership, indentation, and a vertical gap greater than 1.5 times the local median line gap. Page boundaries add two newlines but no invented page label. A line-final hyphen rejoins with the next line only when OCR/text geometry shows one split word; otherwise the hyphen remains.

The target's page geometry determines where `[[TARGET]]` is inserted, and all other confirmed masks insert `[REDACTED]`. The original visible words are retained alongside the serialized text in provenance so a reviewer can trace every token to a page box.

The paper's prompt requires a bulletin date. The backend extracts it only from visible document text or PDF metadata corroborated by visible text. When no reliable date is present, the user may enter a date copied from the visible document; the value and its manual provenance are stored before retrieval. Analysis does not run with an unknown or inferred date.

### Visible-only length cue

The paper's buckets remain exact: `a few words` (up to 4), `part of a sentence` (5-15), `about a sentence` (16-30), and `a paragraph` (31 or more). The application must not count revealed answer words.

Instead, it estimates capacity from visible geometry. For each redaction line, it measures median character width and median characters-per-word from nearby visible OCR/text-layer words with a similar line height, estimates the covered character capacity, converts it to estimated words, sums lines, and applies the paper's buckets. The trace stores the measurements and result. If the page lacks enough visible text to estimate a bucket, the passage stops with a plain error rather than borrowing a hidden answer.

## Corpus build and provenance

### Source reconciliation

The authenticated `HistoryLab/foiarchive` Hugging Face snapshot remains a useful immutable source artifact, but inspection proved that it is not the complete paper corpus. Its card describes about 4.95 million documents across 11 collections, while Appendix C reports 5.55 million files. The current History Lab public catalog also exposes a separate `cables` collection of about 718,000 documents that is absent from that snapshot. In addition, the snapshot's field named `source` contains an origin URL or null rather than the archive collection label required to apply Andrew's collection-level exclusions. The application therefore must not call a corpus or index made directly from that file a reproduction of the paper's public collection.

The corrected source boundary is a frozen local capture of the History Lab public document API. The capture records the exact base URL, projection, ordering, capture time, before-and-after exact counts, and one checksummed JSONL artifact per collection. Keyset pagination by `doc_id` is checkpointed so an interrupted multi-million-document transfer resumes without restarting. Every normalized record carries an explicit `corpus` label; an absent or unknown label fails closed. Because the API is a mutable service rather than a content-addressed release, the resulting local checksums and count-consistency report are the immutable input to every later build.

The existing Hugging Face snapshot is retained outside Git with its repository commit, byte size, and SHA-256 for audit and reconciliation. It is not mixed with a later API capture to manufacture a hybrid corpus: doing so would silently combine source versions. If the History Lab API is unavailable or changes while a capture is in progress, the export pauses and readiness remains false.

Official FRUS publication metadata comes from the Office of the Historian's open `frus-metadata.xml` feed and is also frozen by URL, retrieval time, and SHA-256. Every FRUS document ID must match a canonical volume identifier. The captured archive must contain exactly the ten 2019-2024 FRUS volumes described by Appendix C; a different count is reported as a source-version mismatch rather than waived. The whole History Lab collection corresponding to the paper's CIA reading-room bucket is excluded only after its mapping is verified from public collection metadata. The `cables` collection must be present and non-empty.

The local preflight requires at least 150 GiB free disk for the source capture, normalized/chunk data, OpenSearch index, float16 embeddings, and FAISS artifacts; 250 GiB is recommended to leave safe rebuild headroom. It records RAM, CPU, accelerator availability, and free bytes before each material stage. The user has supplied the minimum local headroom, so the capture may proceed while storage is monitored. A remote build would be separately costed and still requires explicit approval before incurring cost.

The paper's known cutoffs are encoded as a versioned exclusion manifest:

- exclude the CIA reading-room bucket that carries 2019+ releases;
- exclude FRUS volumes published in 2019-2024 using a generated Office of the Historian volume-publication manifest;
- retain other pre-cutoff archive collections used by the paper;
- exclude a target bulletin under every known earlier/later identifier before retrieval.

The builder produces a coverage report. Readiness remains false if a required exclusion rule cannot be applied to the captured schema, the archive changes during capture, the document total falls outside the rounding interval implied by the paper's 5.55M-file report, `cables` is absent, the CIA collection mapping is unverified, a FRUS identifier is unmatched, or the expected ten excluded FRUS volumes are not present. Unknown release provenance is surfaced in that report; it is never silently treated as pre-cutoff.

### Deterministic chunks

Because the paper specifies only chunks of up to 300 words, the reconstruction uses a frozen paragraph-aware chunker:

1. Normalize Unicode and whitespace without changing words.
2. Greedily combine paragraphs while the chunk stays at or below 300 whitespace-delimited words.
3. Split a paragraph longer than 300 words at sentence boundaries, then at word boundaries only when a sentence itself is longer than 300 words.
4. Use no overlap, because the paper reports no overlap and duplicate overlap would alter document ranking.
5. Assign `sha256(document_id, chunk_ordinal, normalized_text)` as the chunk ID.

The algorithm and its version are part of the corpus manifest.

## Retrieval implementation

### Lexical lists

OpenSearch stores the chunk text and metadata and returns BM25 top 300. The analyzer lowercases and tokenizes with the standard Unicode tokenizer; it uses a vendored, versioned English stop-word list and no stemming so historical names and transliterations remain intact. BM25 parameters are frozen at Lucene defaults `k1=1.2` and `b=0.75` because the paper does not report alternatives.

The context query is the visible paragraph window with that same frozen stop-word list removed. The own query is generated from the whole visible bulletin by the paper's Appendix F prompt and is validated as one line, at most 30 words, with the permitted punctuation removed exactly as instructed.

### Dense lists

`BAAI/bge-base-en-v1.5` is pinned by model revision. Documents use raw chunk text; queries use the model's published retrieval instruction. Embeddings are L2-normalized and searched by inner product, which is cosine similarity after normalization.

The full float16 chunk matrix is stored as a memory-mapped artifact. FAISS HNSW supplies a larger candidate set, and the worker exactly re-scores candidates against the float16 matrix before keeping the top 300. Initial HNSW parameters are `M=32`, `efConstruction=200`, and query `efSearch=512`, with 2,000 candidates before exact re-scoring. The index build must meet a retrieval quality gate of at least 0.98 recall@300 against exact flat search on a versioned stratified query sample; otherwise parameters increase before the corpus is marked ready. This is an index-quality check, not a model comparison or prediction evaluation.

### Document grouping and fusion

Each of the four lists is grouped by document using its highest-ranked chunk. Documents merge when either their normalized filename matches or the normalized first 120 word tokens of their best chunk hash identically. Filename normalization is Unicode NFKC, case-folding, path/extension removal, whitespace collapse, and removal of a versioned list of archive-copy suffixes.

Reciprocal-rank fusion uses the standard missing-paper-detail choice `k=60`. For every document, sum `1 / (60 + rank)` over the four lists where it appears. Ties break by stable document ID. Keep the top 100 documents, each with its two highest-scoring retrieved chunks across the four lists, one when only one was retrieved.

Self-document exclusion happens before fusion. Known archive identifiers are primary. When an arbitrary upload lacks identifiers, a normalized 8-word-shingle fingerprint checks for a near-duplicate corpus document. The deterministic matcher excludes every corpus document above the frozen self-match threshold. If several candidates straddle the threshold or appear to be different releases of the same document, retrieval fails with `self_document_identity_unresolved`; no person may resolve it by choosing search results after seeing them. A user may correct archive identifiers as pre-retrieval document metadata, and the correction is recorded before any search runs.

Chunk selection avoids comparing raw BM25 and cosine scores. For each chunk inside a fused document, sum `1 / (60 + rank)` over the four lists where that same chunk appears, then select the top two by this chunk-level reciprocal-rank score with chunk ID as the stable tie-breaker.

The `{documents}` prompt value has a frozen minimal serialization:

```text
[DOCUMENT 1]
[EXCERPT 1]
<retrieved chunk text>
[EXCERPT 2]
<retrieved chunk text>

[DOCUMENT 2]
...
```

Only a missing second excerpt is omitted. Before a model call, the backend counts the complete request with the target model tokenizer. The disclosed maximum of 100 documents times two 300-word chunks is expected to remain comfortably inside GPT-6 Astra's context window; if the complete request exceeds the provider's context limit, the passage fails with `prompt_too_large` rather than truncating, dropping, or reordering evidence.

## Model calls

The backend uses the OpenAI Responses API with `gpt-6-astra`, reasoning effort `medium`, no temperature override, no tools, no system prompt, and data storage disabled when the API supports it. The OpenAI API key stays outside Git and is read only by the worker.

Each passage has two calls:

1. The verbatim Appendix F own-query prompt over the visible bulletin and geometry-derived bucket.
2. The verbatim Appendix F generation-with-documents prompt over the fused top 100 documents and visible bulletin.

The response must be prediction text only. Provider refusals, empty output, invalid query shape, rate limits, and timeouts become explicit passage failures. The system never falls back to a cheaper or different model.

No paid call occurs until the user supplies or confirms model access and approves a per-smoke-test spending cap. Before any larger run, readiness reports the number of passages, estimated input tokens, service tier, and cost ceiling for separate approval. Deployment is another separate approval.

## Frontend integration

The existing view components and selection reducer remain authoritative. `PassageOverlayLayer` continues to render only transparent interactive geometry and visual highlighting. `PredictionSidebar` continues to move the effective hovered or pinned passage into its selected area while keeping other results in stable document order.

The production `PredictionService` becomes an HTTP/SSE adapter. `DocumentWorkspace` stops constructing `SAMPLE_PREDICTIONS` or the uploaded fake prediction in production. During incomplete slices, the sidebar states exactly which backend stage is not connected; it never fills the card with invented text.

The fictional sample may remain only until the real one-passage path is operational and remains labeled simulated. Before completion it is removed from production or replaced by a real, public paper example that runs through the same backend. Test fixtures may continue to use the mock service in Vitest.

Real prediction cards replace `Demo LLM` and `Simulated prediction` with the configured model label and neutral `Model prediction` copy. The layout, hover/pin behavior, PDF page navigation, keyboard behavior, responsive view switch, and lack of on-PDF prediction text do not change.

## Failure handling and honesty

- Backend unavailable: upload/viewing may still work locally, but analysis controls say the backend is not connected.
- Corpus unavailable or building: analysis is disabled with corpus progress/readiness, not fake results.
- OCR/layout failure: keep the PDF visible and identify the affected page; manual passage marking may retry after extraction improves.
- Retrieval returns no documents: still run generation only if the user later approves a documented no-evidence behavior; the initial design reports `no evidence retrieved` and does not pretend retrieval succeeded.
- Model unavailable or unpaid: preserve extraction and retrieval artifacts and show `Model access not connected`.
- Replacement/deletion: cancel queued jobs, close SSE, delete transient files, and prevent late events from entering the new workspace.

## Security and retention

- Accept PDFs only; validate signature and parser result, sanitize filenames, and never interpolate paths.
- Set page, file-size, OCR-time, decompression, request, and model-token limits.
- Run OCR and PDF parsing in resource-limited worker processes.
- Do not log PDF contents, prompts, retrieved text, or API keys in normal application logs. Detailed provenance lives in access-controlled database/artifact records.
- Local development defaults to deleting uploaded PDFs and derived page images after 24 hours; predictions and traces are deleted with the document.
- Before any hosted deployment, add authentication, encrypted object storage, TLS, explicit retention copy, and a threat-model review.

## Incremental delivery slices

### Slice 1: backend foundation and truthful readiness

Bring up FastAPI, PostgreSQL, Redis, worker, migrations, typed frontend client, and `GET /v1/readiness`. No current UI behavior changes except an honest backend-unavailable state. Verify backend unit/integration tests plus all existing frontend tests and type checking.

### Slice 2: real upload and one manual passage

Connect `Choose PDF` to backend upload while retaining immediate local viewing. Connect `Mark a passage` to a real `PassageRecord`; extract visible text/layout and build `VisibleExample`. No prediction appears. Verify the stored PDF hash, normalized rectangles, target/other markers, replacement cleanup, and absence of any answer field.

### Slice 3: corpus build and visible context retrieval

Build and checksum the corpus, exclusion manifest, chunks, and BM25 index. Run the context query through BM25 for one passage and record the top 300. This produces real retrieval artifacts without a model call. Verify deterministic rebuild samples and self-document exclusion.

### Slice 4: dense retrieval and fusion

Build pinned BGE embeddings and FAISS index, run both retrieval methods for the context query, then implement four-list fusion with a test-supplied own-query string at the service boundary. This tests retrieval mechanics without pretending the query came from a model. Nothing user-visible is called a prediction.

### Slice 5: one real query and prediction

Ask the user for model access and a small spending cap. Make one own-query call and one generation call for one manually marked passage. Stream its real result to the existing sidebar. Verify the exact prompt/config trace and passage-linked hover, pin, keyboard, and mobile behavior.

### Slice 6: automatic passage detection

Enable OCR/CV detection and grouping for all pages, keeping manual marking as correction/fallback. Verify page overlays visually and with golden public PDFs before scheduling predictions.

### Slice 7: whole-document progressive processing

Analyze all detected passages with bounded concurrency, idempotency, cancellation, retry, partial completion, and per-passage errors. The sidebar receives results progressively in document order while hover temporarily raises one to the selected area.

### Slice 8: remove finished-path simulation and harden

Remove or replace the fictional production sample, complete retention/security behavior, run the complete frontend/backend/browser suite, update documentation and checklist, and prepare—but do not perform—deployment steps for separate approval.

## Verification strategy

- Python: `pytest`, type checking, linting, migration checks, API contract tests, and worker integration tests.
- Frontend: preserve the current Vitest suite and add upload, readiness, SSE, cancellation, stable passage identity, sidebar ordering, and no-overlay-text tests.
- Retrieval: deterministic chunk and rank fixtures; four-list fusion tests; self-document exclusion tests; corpus manifest/checksum checks; dense index recall gate.
- Document processing: public non-sensitive PDFs covering embedded text, scans, multi-line boxes, columns, white/black boxes, malformed files, and boundary limits.
- Model smoke test: exactly one passage only after approval, with request/response IDs, usage, cost, and prompt/config hashes recorded.
- Browser QA after each user-visible slice: upload, viewer, marking/detection, page/zoom alignment, hover/pin, prediction-to-document navigation, replacement, keyboard, and responsive views.

Every slice ends with the relevant checks, a checklist update, a focused commit, and a push. A slice is not called complete because later stages are mocked; incomplete stages stay visibly unconnected.

## Builder calls

- Keep all secrets and large artifacts outside the repository.
- Prefer explicit versioned configuration objects over environment-variable-only behavior; environment variables choose credentials and artifact locations, not scientific parameters.
- Do not add an evaluation subsystem merely because the paper contains one.
- Do not expose retrieved documents in the current UI unless the user later asks; retain them in provenance for audit and debugging.
- When corpus metadata cannot support a disclosed exclusion, stop corpus readiness and report the exact gap rather than broadening the archive silently.
- Optimize only after one passage works end to end, but design every stage for bounded whole-document concurrency from the start.
