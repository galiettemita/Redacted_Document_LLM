# Best-Available FOIArchive Predictions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the pinned, audited History Lab Hugging Face snapshot into verified local retrieval artifacts and use Andrew Tang's four-list method to produce honestly labelled archive-assisted predictions.

**Architecture:** Add a separate `historylab-best-available-v1` source and manifest family that can never satisfy the exact-paper corpus gate. Stream the locked gzip into resumable deterministic chunks, build a resumable profile-specific BM25 index and bounded sharded BGE/FAISS index, then wire those artifacts into the existing basis-specific readiness, worker, two-stage model flow, SSE, and clean frontend labels.

**Tech Stack:** Python 3.12, Pydantic, gzip JSONL streaming, OpenSearch 3.9, BAAI/bge-base-en-v1.5, NumPy float16, FAISS HNSW, FastAPI, Dramatiq, PostgreSQL, Redis, React/TypeScript.

**Spec:** `docs/superpowers/specs/2026-11-20-staged-real-predictions-design.md`

## Global Constraints

- Accept only revision `411cf6f86b38342683e27148be167c6094d1a10c`, source SHA-256 `dd73e590134f3a0cd1e22f81ffd3732054fd469b09c56d78cd25e1cc90b5e5e4`, canonical audit manifest SHA-256 `901a51d5b468e7dad5d59a3614b73d9b8b46f14bd2a22eb23b797a770a9e2817`, and limitations SHA-256 `e1148411dbd8e91378d8e6e926d88f68908327ab7e1f1c9f92b389cf5c5db2e5`.
- Preserve the audited 4,889,089 parsed/unresolved rows, 1,304,229 HTTP URLs, 3,584,860 null sources, empty canonical collection counts, and exact reason-code set; never infer collection labels.
- Store every included row under the literal non-scientific corpus label `historylab-unlabeled`; preserve `source` only as `source_url`.
- Keep `paper_coverage_ready=false`, and keep best-available manifests, index names, paths, readiness, approvals, and runtime distinct from `paper_exact_v1`.
- Keep Andrew-shaped choices unchanged: 300-word chunks, BM25 `k1=1.2`/`b=0.75`, pinned BGE revision, four independent top-300 lists, RRF `k=60`, top 100 documents, and at most two excerpts per document.
- Dense shards contain at most 1,000,000 consecutive chunks and participate in one globally exact-rescored top-300 result.
- Do not make a provider generation during implementation or index construction; the first retrieval-assisted call needs a separate exact two-stage cost review and explicit approval.
- Keep the fictional demo unchanged and never display internal corpus diagnostics in ordinary prediction cards.

## Review Focus

- A valid gzip paired with a changed audit, snapshot, limitations file, row count, or reason set must fail before publishing any ready artifact; cover in Task 1.
- Interruption after a committed chunk or BM25 batch must resume without duplicating or skipping chunks and without publishing a final manifest early; cover in Tasks 2 and 3.
- Empty bodies, malformed rows, duplicate document IDs, and filesystem exhaustion must produce explicit counts/errors while retaining the last valid checkpoint; cover in Task 2.
- Missing, overlapping, reordered, corrupt, or scientifically incompatible dense shards must close readiness and runtime construction; cover in Task 4.
- A best-available job must retain `best_available_foiarchive_v1` through query, retrieval, prediction, retry, events, and frontend display and must never downgrade to model-only; cover in Tasks 5 and 6.

---

### Task 1: Freeze the best-available source contract

**Files:**
- Create: `backend/src/redacted_backend/corpus/best_available.py`
- Test: `backend/tests/corpus/test_best_available.py`
- Modify: `backend/src/redacted_backend/corpus/cli.py`

**Interfaces:**
- Consumes: `SnapshotLock`, `BulkExportAudit`, `BulkExportCatalog`, `ScientificConfig`, and the four-bullet limitations file.
- Produces: `BestAvailableSourceProfile`, `BestAvailableCorpusManifest`, `load_best_available_source(...)`, and CLI command `build-best-available-chunks`.

- [x] **Step 1: Write failing profile tests** for every frozen hash/count/reason, literal label, audit mismatch, changed limitations, and `paper_coverage_ready=false`.
- [x] **Step 2: Run `uv run pytest tests/corpus/test_best_available.py -q` and verify the new imports fail.**
- [x] **Step 3: Implement the frozen Pydantic profile and provenance verifier** with exact source, audit, snapshot, catalog, and limitations checks.
- [x] **Step 4: Run the focused tests and verify they pass.**
- [x] **Step 5: Commit and push with `feat: freeze best-available corpus profile`.**

### Task 2: Stream and resume deterministic chunk publication

**Files:**
- Modify: `backend/src/redacted_backend/corpus/best_available.py`
- Modify: `backend/src/redacted_backend/corpus/cli.py`
- Test: `backend/tests/corpus/test_best_available.py`

**Interfaces:**
- Consumes: `load_best_available_source(...)`, `_BulkExportRow`-compatible nine-field rows, and `chunk_document(...)`.
- Produces: `build_best_available_chunks(source_path, output_path, manifest_path, checkpoint_path, *, ...) -> BestAvailableCorpusManifest`.

- [ ] **Step 1: Write failing streaming tests** for deterministic chunks, literal corpus identity, metadata preservation, empty-body counts, malformed/duplicate failure, atomic publication, simulated interruption, checkpoint truncation/verification, and exact resume equivalence.
- [ ] **Step 2: Run the focused tests and verify failure.**
- [ ] **Step 3: Implement bounded-memory gzip streaming** with fsynced output/checkpoints, committed source-row and byte offsets, prefix checksum verification, re-stream-and-skip resume, and atomic final manifest publication.
- [ ] **Step 4: Add the CLI arguments** for source, snapshot lock, catalog, audit, limitations, output chunks, checkpoint, and manifest.
- [ ] **Step 5: Run focused corpus and CLI tests, then commit/push with `feat: import best-available corpus chunks`.**

### Task 3: Build resumable profile-specific BM25

**Files:**
- Modify: `backend/src/redacted_backend/retrieval/bm25.py`
- Modify: `backend/src/redacted_backend/corpus/build_bm25.py`
- Modify: `backend/src/redacted_backend/corpus/cli.py`
- Modify: `backend/src/redacted_backend/retrieval/types.py`
- Test: `backend/tests/corpus/test_build_bm25.py`
- Test: `backend/tests/retrieval/test_bm25.py`

**Interfaces:**
- Consumes: `BestAvailableCorpusManifest` and chunk JSONL.
- Produces: `build_best_available_bm25(...) -> Bm25BuildReport` and index `foiarchive-best-available-bm25-v1` with a resumable checkpoint tied to all manifest hashes.

- [ ] **Step 1: Write failing tests** for separate index identity, idempotent batch resume, live-count reconciliation, incompatible checkpoint rejection, no premature manifest, and unchanged exact-paper behavior.
- [ ] **Step 2: Run focused tests and verify failure.**
- [ ] **Step 3: Generalize BM25 manifest validation** to an explicit retrieval-corpus protocol without allowing best-available to satisfy exact coverage.
- [ ] **Step 4: Implement checkpointed idempotent bulk indexing** and atomic final manifest publication.
- [ ] **Step 5: Run focused tests and commit/push with `feat: build resumable best-available bm25`.**

### Task 4: Build and query bounded sharded dense artifacts

**Files:**
- Create: `backend/src/redacted_backend/corpus/build_dense_sharded.py`
- Create: `backend/src/redacted_backend/retrieval/dense_sharded.py`
- Modify: `backend/src/redacted_backend/retrieval/types.py`
- Modify: `backend/src/redacted_backend/corpus/cli.py`
- Create: `backend/corpus/best_available_application_queries.txt`
- Test: `backend/tests/corpus/test_build_dense_sharded.py`
- Test: `backend/tests/retrieval/test_dense_sharded.py`
- Test: `backend/tests/retrieval/test_dense_recall.py`

**Interfaces:**
- Consumes: best-available chunks/manifest, `BgeEncoder`, and 50 frozen public application-shaped queries.
- Produces: `ShardedDenseIndexManifest`, a generated 100-query certification artifact containing 50 application-shaped and 50 stable-hash corpus-derived queries, atomic per-shard embeddings/metadata/offsets/FAISS artifacts, `build_sharded_dense_from_files(...)`, and `ShardedDenseIndex.search(...)`.

- [ ] **Step 1: Write failing builder tests** for one-million-row boundaries, consecutive non-overlapping ranges, per-file checksums, resume, corruption rejection, and no top-level manifest before all shards certify.
- [ ] **Step 2: Write failing runtime tests** proving every shard participates, only candidate metadata is read by offset, candidate vectors are exact-rescored globally, ties use chunk ID, and output matches an unsharded exact fixture.
- [ ] **Step 3: Write failing certification tests** for batched exact streaming top-300, sharded candidate-union recall, query-file hash, and the 0.98 minimum.
- [ ] **Step 4: Implement atomic resumable shard construction** with float16 memmaps, uint64 metadata offsets, FAISS HNSW, and bounded batches.
- [ ] **Step 5: Implement sequential/bounded shard querying and certification** without loading all graphs or metadata into memory.
- [ ] **Step 6: Freeze 50 public application-shaped queries and implement stable-hash selection of 50 title/opening-text queries from the verified corpus; atomically publish the combined query artifact and record its SHA-256.**
- [ ] **Step 7: Run dense suites and commit/push with `feat: add sharded best-available dense retrieval`.**

### Task 5: Wire separate readiness and worker runtime

**Files:**
- Modify: `backend/src/redacted_backend/config.py`
- Modify: `backend/src/redacted_backend/api/readiness.py`
- Modify: `backend/src/redacted_backend/domain/readiness.py`
- Modify: `backend/src/redacted_backend/jobs/runtime.py`
- Modify: `backend/src/redacted_backend/jobs/orchestrator.py`
- Modify: `backend/src/redacted_backend/models/openai_gateway.py`
- Modify: `compose.yaml`
- Test: `backend/tests/test_readiness.py`
- Test: `backend/tests/jobs/test_runtime_provider.py`
- Test: `backend/tests/jobs/test_orchestrator.py`
- Test: `backend/tests/models/test_openai_gateway.py`

**Interfaces:**
- Consumes: verified best-available corpus/BM25/sharded-dense manifests and artifacts.
- Produces: operational `best_available_foiarchive_v1` readiness and an orchestrator that runs query generation, four-list retrieval, and a basis-correct prediction.

- [ ] **Step 1: Write failing readiness tests** for separate paths, hashes, live index count, dense shard completeness, approval independence, and exact-paper non-contamination.
- [ ] **Step 2: Write failing runtime/orchestrator tests** for two provider stages, immutable best-available basis, durable resume boundaries, retrieval-trace persistence, and no downgrade.
- [ ] **Step 3: Make retrieval-assisted generation basis-aware** so best-available records/events never claim `paper_exact_v1`.
- [ ] **Step 4: Load and probe best-available BM25/sharded dense only for that basis** and expose its separate Compose paths.
- [ ] **Step 5: Run focused tests and commit/push with `feat: enable archive-assisted prediction runtime`.**

### Task 6: Verify the product flow and preserve the simple UI/demo

**Files:**
- Modify: `README.md`
- Modify: `CHECKLIST.md`
- Modify only if tests reveal a gap: `src/features/predictions/*`, `src/features/workspace/DocumentWorkspace.tsx`, and their tests.
- Create: `backend/tests/integration/test_best_available_loopback.py`

**Interfaces:**
- Consumes: complete best-available runtime and existing prediction-basis UI.
- Produces: upload-to-SSE archive-assisted loopback, operational runbook, and completed checklist evidence.

- [ ] **Step 1: Add a fake-transport integration test** covering upload, detection, visible context, own-query generation, four ranked lists, fused evidence, prediction, durable best-available provenance, and SSE.
- [ ] **Step 2: Re-run UI tests** proving **Archive-assisted prediction**, **Model-only prediction**, and **Simulated prediction** remain distinct and no prediction text appears over the PDF.
- [ ] **Step 3: Document source linking, chunk/index commands, readiness, limitations, and separate cost approval without exposing research diagnostics in the UI.**
- [ ] **Step 4: Run `uv run pytest -q`, `npm test -- --run`, `npm run typecheck`, `npm run build`, migration checks, and `git diff --check`.**
- [ ] **Step 5: Update `CHECKLIST.md`, commit/push with `feat: ship best-available archive prediction path`, and keep all paid approvals disabled.**

### Task 7: Build the real local artifacts without paid generation

**Files:**
- Runtime artifacts only under ignored `var/corpus/`; no source data or index binaries enter Git.
- Modify: `CHECKLIST.md` only after each verified stage.

**Interfaces:**
- Consumes: the pinned local source, audit, snapshot, limitations, production builders, OpenSearch profile, and frozen recall queries.
- Produces: verified chunks, BM25, dense shards, live best-available operational readiness, and an exact two-stage cost estimate for later approval.

- [ ] **Step 1: Link the pinned source into the ignored corpus workspace and run the production chunk importer to completion/resume.**
- [ ] **Step 2: Start OpenSearch and run the resumable BM25 build; verify live count equals the manifest.**
- [ ] **Step 3: Run the resumable sharded dense build and 100-query recall certification; verify every checksum and range.**
- [ ] **Step 4: Recreate backend/worker with best-available artifact paths but all paid approvals false; verify operational=true, approved=false, ready=false.**
- [ ] **Step 5: Count the exact own-query request and the resulting evidence-bound prediction request without generation, show expected/maximum combined cost, and request separate approval for one archive-assisted passage.**
- [ ] **Step 6: Update the checklist and push the verified artifact-build state; do not commit runtime artifacts.**
