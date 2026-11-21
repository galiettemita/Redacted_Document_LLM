# Staged Real Predictions with Best-Available Retrieval

## Intent

Deliver an honest, genuinely model-generated prediction as soon as model access is approved, then upgrade the same production flow to retrieval over the available Hugging Face FOIArchive while preserving a separate, still-closed gate for an exact reproduction of Andrew Tang's corpus.

The product must never describe the available archive as the paper's complete corpus. The four agreed limitations live verbatim in `4-bulletpoint-limitations.md` and are part of the best-available profile's frozen provenance.

## Current state

The production upload, passage detection, visible-only context preparation, durable job queue, SSE events, OpenAI gateway, four-list retrieval implementation, trace persistence, worker heartbeat, and simple prediction cards already exist. The worker is independently live, but scheduling currently treats the exact paper corpus as mandatory for every real prediction.

The authenticated Hugging Face snapshot is genuine and checksummed, but its audit found 4,889,089 rows rather than the paper's approximately 5.55 million files. Its collection identity is unusable for the paper's collection-specific exclusions, and it cannot prove that every retrieved document was public before a redaction. It can nevertheless support honest best-available full-text retrieval if those limitations remain explicit and it is never passed through the exact-paper readiness gate.

## Goals

- Produce a real model-only prediction from the uploaded document's visible text without waiting for a retrieval corpus.
- Reuse the available FOIArchive for real retrieval once verified BM25 and BGE indexes are operational.
- Match the paper wherever the available data permits: frozen prompts, 300-word chunks, BM25, `bge-base-en-v1.5`, context and model-written queries, four top-300 lists, document-level reciprocal-rank fusion, top 100 documents, and two excerpts per document.
- Persist which prediction basis was selected before execution and expose it clearly in the result.
- Preserve the fictional demo unchanged and isolated.
- Keep paid model calls disabled until credentials, an exact token/cost estimate, and explicit approval exist.

## Non-goals

- Claiming that the Hugging Face snapshot reproduces the paper's exact corpus or results.
- Guessing collection identity from URLs, document text, document IDs, or retrieval results.
- Claiming that best-available evidence predates the redaction when release metadata cannot prove it.
- Using a hosted top-k search endpoint as a substitute for local reproducible indexes.
- Silently falling back from retrieval to model-only after a job has started.
- Showing research diagnostics, token counts, or cost details in the customer prediction cards.

## Anti-goals

- No mock output in an uploaded-document workspace.
- No unlabelled mixture of simulated, model-only, and retrieval-assisted predictions.
- No weakening of the exact-paper readiness contract to make the incomplete archive pass.
- No direct LLM scan of a compressed multi-million-document file; retrieval must use bounded, reproducible indexes.

## Approaches considered

### Chosen: staged real modes with one production action

Keep the existing **Predict passage** and **Predict all passages** actions. At scheduling time the backend selects and persists the strongest ready basis: exact-paper retrieval in the future, best-available retrieval when its indexes are ready, otherwise model-only when the worker and approved model are ready. A job never changes basis after scheduling and never silently downgrades on failure.

This produces genuine output soon, keeps the UI simple, and lets the large index build proceed without blocking all useful testing.

### Rejected: wait for every best-available index before any model call

This avoids introducing a model-only mode, but it delays the user's primary goal behind a 4.89-million-document import and a potentially long CPU dense-index build.

### Rejected: let the model call hosted search or inspect the gzip directly

Hosted search is capped and has already depended on an unreliable upstream API. A model cannot efficiently or reproducibly search the raw gzip. Either path would produce weaker provenance than the local pipeline already implements.

## Prediction bases

Every analysis job gains a required immutable `prediction_basis`:

- `model_only_v1`: a real provider prediction using the paper's closed-book prompt and visible example, with no query-generation or retrieval stage.
- `best_available_foiarchive_v1`: the existing realistic query and prediction prompts over the audited Hugging Face snapshot and its verified local indexes.
- `paper_exact_v1`: reserved for the existing exact-paper corpus contract when a suitable source eventually exists.

The scheduling priority is `paper_exact_v1`, then `best_available_foiarchive_v1`, then `model_only_v1`, considering only bases that are both operationally ready and separately approved for paid use. Selection occurs once inside the scheduling transaction. If the selected runtime later fails, the job fails or retries under that same basis.

## Separate spending approvals

The existing `allow_paid_model_calls` remains a master emergency switch. Add separate approvals for `model_only_v1`, `best_available_foiarchive_v1`, and `paper_exact_v1`. A basis is schedulable only when both the master switch and its own approval are true.

Completing best-available indexes therefore cannot silently change a user from one provider generation per passage to the two-stage query-plus-prediction path. The first live call for each basis receives its own exact token/cost review and explicit approval. Readiness reports an operational basis separately from its spending approval.

## Model-only execution

Add a frozen closed-book Appendix F prompt template and a gateway method that produces a `PredictionRecord` without a retrieval trace. The orchestrator skips query generation and retrieval entirely for `model_only_v1`; it still uses the revision-bound visible example, idempotency key, exact input-token count, disabled truncation, provider/request IDs, usage, cost estimate, cancellation checks, and durable completion behavior.

`PredictionRecord` records the basis and permits `retrieval_trace_sha256=null` only for `model_only_v1`. Model-only output remains **Model prediction**, never simulated prediction.

## Best-available corpus import

The importer consumes only the already locked `foiarchive-202409.json.gz` and its exact v2 audit. It verifies filename, byte size, SHA-256, revision, row count, row schema, and a canonical audit-report SHA-256 before publishing derived artifacts.

This profile intentionally accepts one exact failed audit, not any source that happens to be unready. The accepted contract pins revision `411cf6f86b38342683e27148be167c6094d1a10c`, source SHA-256 `dd73e590134f3a0cd1e22f81ffd3732054fd469b09c56d78cd25e1cc90b5e5e4`, 4,889,089 parsed and unresolved rows, 1,304,229 HTTP URLs, 3,584,860 nulls, zero blank/unknown labels, empty canonical counts, and the exact current reason-code set. The canonical audit hash is computed and frozen before import. Any changed hash, count, category, reason, or source property stops the build and requires a new audit plus an explicitly versioned profile; it is never accepted by widening a generic whitelist.

It streams rows and creates necessary normalized chunks without writing a second expanded raw source. Collection identity is stored as the literal non-scientific value `historylab-unlabeled`; origin `source`, `doc_id`, title, authored date, classification, and other supplied metadata remain separate. Empty-body documents are skipped with exact counts. No URL or text is converted into a paper collection label.

The new `historylab-best-available-v1` manifest is separate from `CorpusManifest`'s exact-paper coverage contract. It records:

- source snapshot, catalog, audit, and limitations hashes;
- parsed, included, empty, malformed, document, and chunk counts;
- exact chunk-file checksum and chunker/scientific configuration hashes;
- `paper_coverage_ready=false`; and
- `retrieval_profile=best_available_foiarchive_v1`.

Partial or interrupted artifacts remain unready. Import and index stages use fsynced checkpoints and resume after actual interruption or out-of-space failure.

## Retrieval and indexes

Best-available retrieval deliberately reuses the paper-shaped implementation:

1. Split documents into deterministic chunks of at most 300 words.
2. Build OpenSearch BM25 with `k1=1.2`, `b=0.75`, and the frozen tokenizer/stop words.
3. Encode chunks and queries with the pinned `BAAI/bge-base-en-v1.5` revision.
4. Preserve exact float16 re-scoring over FAISS candidates and the existing recall gate.
5. Retrieve context-BM25, context-dense, own-query-BM25, and own-query-dense lists independently, each to 300 chunks.
6. Apply best-effort self-exclusion using stable IDs, source filenames, and the existing visible-text shingle detector. Ambiguous self-matches are excluded rather than retained.
7. Fuse by the existing document-level RRF configuration, retain 100 documents, and serialize at most two excerpts per document.

The BM25 and dense builders must become resumable/partitioned before a full local build so an interruption does not discard days of work. Best-available readiness stays closed until both live indexes and all manifest hashes/counts agree. Model-only remains available during the build.

### Dense sharding and global ranking

Dense artifacts use deterministic shards of at most 1,000,000 consecutive chunks. Each shard has its own float16 embeddings, metadata, FAISS HNSW candidate index, row range, count, and checksums. A top-level manifest freezes shard order and hashes.

At query time every shard returns up to 2,000 HNSW candidates. The runtime unions those candidates, reads their float16 vectors, exact-rescores them against the normalized query, and applies one global stable top-300 ordering. Because every shard participates, sharding changes storage/execution rather than the final ranking contract. Shards load through a bounded cache or sequentially so the runtime never requires all HNSW graphs in memory. A missing, overlapping, reordered, or incompatible shard closes readiness.

### Dense recall certification

The best-available profile freezes 100 non-sensitive queries: 50 application-shaped visible-context queries authored from public fixtures without hidden text, and 50 deterministic title/opening-text queries selected from the imported corpus by stable document hash. The query file and SHA-256 are recorded in the dense manifest.

For each query, certification streams every embedding shard to compute the exact global top 300, then compares it with the sharded HNSW candidate-union result after exact rescoring. Mean recall@300 must remain at least 0.98. This gate certifies approximate-index coverage, not the scientific completeness of the corpus; the manifest names that distinction. Later real visible-only queries may be audited separately but cannot retroactively change the frozen build result.

## Readiness and scheduling

Readiness becomes additive rather than redefining the exact corpus:

- `model_only`: database, worker heartbeat, model credentials, master paid-call approval, and model-only approval.
- `best_available_retrieval`: database, worker, model credentials, both master and best-available approvals, plus verified best-available chunk/BM25/dense artifacts and live index probes.
- `paper_exact`: the existing exact corpus and index checks plus its own paid approval.

`prediction_ready` means at least one real basis can be scheduled. The response also names the strongest ready basis and reports each basis independently. The scheduling endpoint persists that basis with the job and returns blockers only when no real basis is available.

## API, persistence, and UI

A database migration adds `prediction_basis` to analysis jobs and result events. Existing incomplete jobs are conservatively marked with their known legacy basis or cancelled during migration; no legacy record is guessed into a completed mode.

The browser retains the existing prediction buttons and cards. Each completed card displays one concise basis label:

- **Model-only prediction** — no archive evidence used.
- **Archive-assisted prediction** — available FOIArchive snapshot used.
- **Paper-method prediction** — reserved for a future exact source.

The four limitations remain in the requested Markdown artifact and provenance documentation. The uploaded-document UI does not show corpus counts, implementation diagnostics, token metrics, or the full research disclaimer. Prediction text remains off the PDF.

## Paid-call activation

Implementation and all local tests make no provider request. Before the first real model-only passage:

1. configure an approved provider key without committing or logging it;
2. enable paid calls explicitly;
3. count the exact closed-book request;
4. show the estimated maximum and expected cost to the user; and
5. obtain explicit approval for one passage.

Retrieval-assisted mode uses two provider stages per passage—own query and prediction—and receives a separate cost check before its first live run.

## Failure and safety behavior

- Provider, worker, or selected-runtime failure is durable and never replaced by a fake or weaker result.
- A passage revision or document deletion cancels all remaining stages as it does today.
- Secrets, prompts, document text, queries, evidence, and provider payloads retain existing log-redaction and retention controls.
- The selected basis, prompt hashes, source/index hashes, retrieval trace where applicable, provider IDs, usage, and cost remain auditable.
- Best-available results never assert that evidence was public before the redaction; they assert only that the available snapshot supplied it.

## Verification

Tests must prove:

- model-only jobs schedule with corpus unavailable but model/worker ready;
- model-only performs exactly one prediction generation and no query/retrieval call;
- paid-call approval remains mandatory;
- basis selection is transactional, deterministic, persisted, and visible in SSE/results;
- no runtime failure silently changes basis;
- retrieval traces are absent only for model-only records;
- the best-available importer preserves exact source/audit counts and never invents collection labels;
- interrupted imports/index builds do not publish ready manifests;
- best-available readiness requires both verified indexes and live probes;
- four-list retrieval settings remain identical to the frozen paper-shaped implementation;
- the real workspace labels each basis correctly while the fictional demo remains unchanged; and
- backend, frontend, typecheck, production build, and one approved live one-passage trace pass.

## Delivery order

1. Add persisted prediction bases and the model-only prompt/runtime.
2. Expose model-only readiness and concise UI provenance.
3. After credentials and one-call approval, run a single real model-only passage.
4. Build the streaming best-available importer and resumable chunks.
5. Build and verify BM25, then the partitioned/resumable dense artifacts.
6. Enable best-available retrieval only after all operational gates pass.
7. Keep `paper_exact_v1` closed until a paper-compatible source exists.

## Success criteria

The first milestone is a real, paid, model-only prediction displayed in the existing frontend with durable provenance and no corpus dependency. The second is a real archive-assisted prediction using the audited Hugging Face snapshot and the paper-shaped four-list retrieval pipeline, clearly distinguished from exact paper reproduction. At no point may the system fabricate output, invent missing metadata, or silently weaken a job's selected basis.
