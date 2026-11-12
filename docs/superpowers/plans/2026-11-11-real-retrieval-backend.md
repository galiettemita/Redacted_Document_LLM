# Real Retrieval-Assisted Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the frontend's simulated uploaded-document path with a real, visible-only PDF processing, hybrid retrieval, and GPT-6 Astra prediction backend while preserving passage-linked interactions.

**Architecture:** A Python 3.12 FastAPI service persists documents, passages, jobs, and provenance in PostgreSQL; Dramatiq/Redis run document, retrieval, and model stages outside requests. The existing React/Vite viewer keeps local PDF rendering and consumes typed HTTP/SSE contracts. A frozen History Lab corpus build supplies four BM25/BGE lists that deterministic code fuses before one prediction is generated per stable passage ID.

**Tech Stack:** Python 3.12, uv, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, PostgreSQL, Redis, Dramatiq, PyMuPDF, Tesseract, OpenCV, OpenSearch, `BAAI/bge-base-en-v1.5`, FAISS, OpenAI Responses API, React 19, TypeScript 7, Vite 8, Vitest, Pytest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-11-real-retrieval-backend-design.md`

## Global Constraints

- Runtime request/domain types must not contain `answer`, `reveal`, `ground_truth`, later-release text, oracle queries, or manually selected evidence.
- The only generator is exactly `gpt-6-astra` with reasoning effort `medium`, one user message, no system prompt, and no silent model fallback.
- Retrieval is exactly two visible-only queries through BM25 and pinned `BAAI/bge-base-en-v1.5`, four top-300 chunk lists, document-level RRF with `k=60`, top 100 documents, and two rank-selected chunks per document.
- Chunks contain at most 300 whitespace-delimited words, preserve paragraphs, use no overlap, and are identified by a deterministic SHA-256 value.
- Passage rectangles remain normalized page-relative `x`, `y`, `width`, and `height` values in the 0-1 range.
- Hover highlights every rectangle for one passage and raises its prediction to the sidebar's selected area; prediction text never appears over the PDF.
- No fake model result may appear in a finished user path. Test doubles are confined to automated tests.
- Corpus data, indexes, uploads, credentials, and secrets stay outside Git. Scientific parameters are explicit versioned configuration, not environment-only values.
- Ask before downloading/building the gated full corpus on paid remote infrastructure, before the first paid model call, and before deployment.
- Every completed task updates `CHECKLIST.md`, passes its focused tests plus applicable regression checks, commits, and pushes without staging `Redaction_RAG_NAACL_v3.pdf`.

## Review Focus

- A PDF text layer may contain words underneath a visible black box; Task 4 tests that any intersecting word is removed before transcript serialization.
- A double click, SSE reconnect, worker retry, or document replacement must not create duplicate jobs, duplicate charges, or stale predictions; Tasks 2, 10, and 11 test idempotency and cancellation.
- Multi-column pages, line-wrapped boxes, white boxes, and repeated headers must serialize in visible reading order without joining columns; Tasks 4 and 12 test these inputs.
- A large fused prompt must fail with `prompt_too_large` rather than dropping or reordering retrieved evidence; Tasks 9 and 10 test the token-budget boundary.
- Oversized, malformed, encrypted, or decompression-heavy PDFs must remain viewable only when safe and must never exhaust the backend; Task 2 tests parser and resource limits.

## File Structure

### Backend foundation and document seam

- `backend/pyproject.toml`, `backend/uv.lock`: locked Python application and development dependencies.
- `backend/src/redacted_backend/config.py`: credential locations, service URLs, artifact paths, and operational limits only.
- `backend/src/redacted_backend/scientific_config.py`: frozen chunking, retrieval, prompt, and model parameters plus manifest hashing.
- `backend/src/redacted_backend/main.py`: FastAPI application factory and route composition.
- `backend/src/redacted_backend/api/`: readiness, document, passage, analysis, and SSE HTTP adapters.
- `backend/src/redacted_backend/domain/`: Pydantic domain contracts with no persistence or provider imports.
- `backend/src/redacted_backend/db/`: SQLAlchemy models, sessions, repositories, and Alembic migrations.
- `backend/src/redacted_backend/storage/`: content-addressed local document/artifact storage interface.
- `backend/src/redacted_backend/documents/`: PDF validation, visible text/layout, mask detection, serialization, and length cue construction.
- `backend/src/redacted_backend/jobs/`: Dramatiq messages, idempotent stage orchestration, cancellation, and Redis event publication.

### Corpus and retrieval

- `backend/src/redacted_backend/corpus/`: Hugging Face snapshot lock, exclusion manifest, deterministic chunks, preflight, and build CLI.
- `backend/src/redacted_backend/retrieval/`: OpenSearch BM25, BGE/FAISS dense search, self-document exclusion, RRF, and trace serialization.
- `backend/corpus/exclusions/`: versioned non-secret exclusion rules and generated FRUS publication manifest metadata.

### Model boundary

- `backend/prompts/realistic_query_v1.txt`: verbatim Appendix F own-query prompt.
- `backend/prompts/realistic_prediction_v1.txt`: verbatim Appendix F generation-with-documents prompt.
- `backend/src/redacted_backend/models/`: OpenAI gateway, prompt rendering, validation, token budgeting, and cost capture.

### Frontend boundary

- `src/services/backendApi.ts`: typed fetch/SSE transport only.
- `src/services/httpPredictionService.ts`: implements `PredictionService` from backend events.
- `src/hooks/useBackendDocument.ts`: upload, deletion, readiness, and passage lifecycle.
- Existing `App`, `UploadScreen`, `DocumentWorkspace`, sidebar, and overlay modules retain presentation and selection responsibilities.

---

### Task 1: Backend application, frozen configuration, and readiness

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/src/redacted_backend/__init__.py`
- Create: `backend/src/redacted_backend/config.py`
- Create: `backend/src/redacted_backend/scientific_config.py`
- Create: `backend/src/redacted_backend/domain/readiness.py`
- Create: `backend/src/redacted_backend/api/readiness.py`
- Create: `backend/src/redacted_backend/main.py`
- Create: `backend/tests/test_readiness.py`
- Create: `backend/tests/test_scientific_config.py`
- Create: `backend/Dockerfile`, `compose.yaml`, `.env.example`
- Modify: `.gitignore`, `README.md`, `CHECKLIST.md`

**Interfaces:**
- Produces: `create_app(settings: Settings | None = None) -> FastAPI`.
- Produces: `ScientificConfig.manifest_hash() -> str` with `chunk_words=300`, `chunk_overlap=0`, `bm25_k1=1.2`, `bm25_b=0.75`, `top_chunks_per_list=300`, `rrf_k=60`, `top_documents=100`, `chunks_per_document=2`, `embedding_model='BAAI/bge-base-en-v1.5'`, `model='gpt-6-astra'`, `reasoning_effort='medium'`.
- Produces: `GET /v1/readiness -> ReadinessResponse` with independent `database`, `worker`, `corpus`, and `model` components.

- [x] **Step 1: Write failing readiness and scientific-manifest tests**

```python
def test_readiness_never_equates_http_health_with_prediction_readiness(client):
    body = client.get('/v1/readiness').json()
    assert body['http']['ready'] is True
    assert body['prediction_ready'] is False
    assert set(body) >= {'http', 'database', 'worker', 'corpus', 'model', 'prediction_ready'}

def test_scientific_manifest_is_stable_and_contains_disclosed_parameters():
    config = ScientificConfig()
    assert config.top_chunks_per_list == 300
    assert config.rrf_k == 60
    assert config.model == 'gpt-6-astra'
    assert config.manifest_hash() == ScientificConfig().manifest_hash()
```

- [x] **Step 2: Run `cd backend && uv run pytest tests/test_readiness.py tests/test_scientific_config.py -v` and verify collection/import failure.**
- [x] **Step 3: Create the Python 3.12 package and implement the exact interfaces above; generate and commit `backend/uv.lock`.**
- [x] **Step 4: Add PostgreSQL, Redis, backend services, persistent named volumes, health checks, and non-secret defaults to `compose.yaml`; do not start OpenSearch yet.**
- [x] **Step 5: Run `docker compose config`, `cd backend && uv run pytest -v`, and `npm test -- --run`; expect all to pass.**
- [x] **Step 6: Update the checklist, commit with `feat: add backend readiness foundation`, and push.**

### Task 2: Safe PDF persistence and upload API

**Files:**
- Create: `backend/src/redacted_backend/domain/documents.py`
- Create: `backend/src/redacted_backend/db/base.py`, `backend/src/redacted_backend/db/models.py`, `backend/src/redacted_backend/db/repositories/documents.py`
- Create: `backend/src/redacted_backend/storage/base.py`, `backend/src/redacted_backend/storage/local.py`
- Create: `backend/src/redacted_backend/documents/validation.py`
- Create: `backend/src/redacted_backend/api/documents.py`
- Create: `backend/alembic.ini`, `backend/alembic/env.py`, `backend/alembic/versions/0001_documents.py`
- Create: `backend/tests/api/test_documents.py`, `backend/tests/integration/test_document_persistence.py`
- Modify: `backend/src/redacted_backend/main.py`, `CHECKLIST.md`

**Interfaces:**
- Produces: `DocumentRecord`, `DocumentStatus`, `DocumentRepository.create/get/mark_deleted`.
- Produces: `DocumentStore.put(stream, expected_sha256) -> StorageObject`, `open(storage_key)`, `delete(storage_key)`.
- Produces: `validate_pdf(path: Path, limits: PdfLimits) -> PdfMetadata` with 100 MiB and 500-page defaults.
- Produces: `POST /v1/documents`, `PATCH /v1/documents/{document_id}/metadata` for a visible document date plus its provenance, and `DELETE /v1/documents/{document_id}`.

- [x] **Step 1: Write failing upload tests for a valid PDF, incorrect `%PDF-` signature, malformed/encrypted/decompression-heavy PDF, parser timeout, 100 MiB boundary, 500-page boundary, sanitized filename, SHA-256 storage key, visible-date metadata, and deletion.**
- [x] **Step 2: Add an integration test proving two identical uploads share content bytes but receive separate document records, and deletion never removes bytes still referenced by another live record.**
- [x] **Step 3: Run `cd backend && uv run pytest tests/api/test_documents.py tests/integration/test_document_persistence.py -v`; expect missing interfaces.**
- [x] **Step 4: Implement models, migration, repository, content-addressed storage, PyMuPDF validation, routes, and dependency injection. Never trust `UploadFile.filename` as a path.**
- [x] **Step 5: Run the focused tests, `uv run alembic upgrade head`, and a Compose-backed upload/delete smoke test; expect PASS and no files inside the Git checkout's tracked tree.**
- [x] **Step 6: Update the checklist, commit with `feat: persist validated PDF uploads`, and push.**

### Task 3: Stable passage records and correction operations

**Files:**
- Create: `backend/src/redacted_backend/domain/passages.py`
- Create: `backend/src/redacted_backend/db/repositories/passages.py`
- Create: `backend/src/redacted_backend/api/passages.py`
- Create: `backend/alembic/versions/0002_passages.py`
- Create: `backend/tests/api/test_passages.py`
- Modify: `backend/src/redacted_backend/db/models.py`, `backend/src/redacted_backend/main.py`, `CHECKLIST.md`

**Interfaces:**
- Produces: `PassageRect`, `PassageRecord`, `PassageSource`, `PassageStatus`, and monotonic `revision`.
- Produces: create, patch, delete, merge, and split endpoints from the spec.
- Invalidation contract: any geometry/date change increments `revision`, cancels active jobs, and tombstones derived visible examples, retrieval traces, and predictions.

- [x] **Step 1: Write failing API tests asserting rectangles reject values outside 0-1, a manual passage keeps its UUID after patch, merge keeps the first document-order UUID, split keeps the original UUID for its first result, and each mutation increments `revision`.**
- [x] **Step 2: Write a failing idempotency test sending the same `Idempotency-Key` twice and asserting one passage record and one response UUID.**
- [x] **Step 3: Run `cd backend && uv run pytest tests/api/test_passages.py -v`; expect missing passage routes.**
- [x] **Step 4: Implement the migration, repository transaction rules, route schemas, and idempotency record without adding prediction fields.**
- [x] **Step 5: Run focused tests and `uv run alembic upgrade head`; expect PASS.**
- [x] **Step 6: Update the checklist, commit with `feat: add stable passage lifecycle`, and push.**

### Task 4: Visible-only extraction, mask safety, and example construction

**Files:**
- Create: `backend/src/redacted_backend/documents/extract.py`
- Create: `backend/src/redacted_backend/documents/masks.py`
- Create: `backend/src/redacted_backend/documents/layout.py`
- Create: `backend/src/redacted_backend/documents/length_cue.py`
- Create: `backend/src/redacted_backend/documents/visible_example.py`
- Create: `backend/src/redacted_backend/domain/visible_examples.py`
- Create: `backend/tests/documents/test_masks.py`, `test_layout.py`, `test_length_cue.py`, `test_visible_example.py`, `test_ocr_integration.py`
- Create: `backend/tests/fixtures/pdfs/README.md` and public/non-sensitive generated fixtures
- Modify: `backend/Dockerfile`, `backend/src/redacted_backend/scientific_config.py`, `CHECKLIST.md`

**Interfaces:**
- Produces: `extract_visible_words(pdf_path: Path) -> list[PageWords]`.
- Produces: `discover_candidate_masks(page_image, words) -> MaskDiscovery` with `coverage_confident`.
- Produces: `serialize_visible_document(pages, masks, target_id) -> SerializedTranscript`.
- Produces: `estimate_length_cue(target_rects, nearby_words) -> LengthEstimate` using the four paper buckets.
- Produces: `build_visible_example(document, passages, target_id) -> VisibleExample`; its Pydantic schema forbids unknown fields and has no hidden-answer names.

- [x] **Step 1: Write failing tests for embedded words intersecting a black box, white boxes, multi-line targets, columns, repeated headers, line-break hyphenation, uncertain mask coverage, missing visible date, and all four length buckets.**
- [x] **Step 2: Add a schema test that recursively inspects `VisibleExample.model_json_schema()` and asserts none of `answer`, `reveal`, `ground_truth`, or `later_release` appears.**
- [x] **Step 3: Run `cd backend && uv run pytest tests/documents -v`; expect missing extraction modules. Unit tests inject OCR output and do not require a host Tesseract install.**
- [x] **Step 4: Add Tesseract and OpenCV to the backend image and implement embedded-text-first extraction, OCR fallback, early mask discovery, XY-cut ordering, target/other markers, visible date handling, and geometry-derived cue.**
- [x] **Step 5: Render every PDF fixture to PNG and visually verify mask alignment, reading order, and no covered text in the serialized transcript.**
- [x] **Step 6: Run `cd backend && uv run pytest tests/documents -v` for unit tests and `docker compose run --rm backend uv run pytest tests/documents/test_ocr_integration.py -v` for real Tesseract; expect PASS.**
- [x] **Step 7: Update the checklist, commit with `feat: build visible-only passage examples`, and push.**

### Task 5: Connect real upload and manual passage marking to the frontend

**Files:**
- Create: `src/services/backendApi.ts`, `src/services/backendApi.test.ts`
- Create: `src/hooks/useBackendDocument.ts`, `src/hooks/useBackendDocument.test.tsx`
- Modify: `src/types.ts`, `src/App.tsx`, `src/App.test.tsx`
- Modify: `src/features/upload/UploadScreen.tsx`, `src/features/upload/UploadScreen.test.tsx`
- Modify: `src/features/workspace/DocumentWorkspace.tsx`, `src/features/workspace/DocumentWorkspace.test.tsx`
- Modify: `vite.config.ts`, `README.md`, `CHECKLIST.md`

**Interfaces:**
- Produces: `BackendApi.getReadiness`, `uploadDocument`, `deleteDocument`, `createPassage`, and `updatePassage` with `AbortSignal` support.
- Produces: `useBackendDocument(file: File | null) -> BackendDocumentState` with `uploading | ready | unavailable | failed | deleting`.
- `DocumentDescriptor` gains optional `backendDocumentId`; local object URL remains `source`.

- [x] **Step 1: Write failing transport tests asserting multipart upload shape, normalized passage rectangles, typed error mapping, abort behavior, and deletion request.**
- [x] **Step 2: Write failing component tests asserting the PDF workspace opens immediately while backend upload is pending, uploaded copy no longer says `stays on your device`, a mark calls the backend, and no fake prediction appears after marking.**
- [x] **Step 3: Run the focused Vitest files; expect failures against the current synchronous/local-only flow.**
- [x] **Step 4: Implement the API client, hook, Vite `/v1` proxy, immediate local viewer plus background backend upload, real passage creation, honest readiness/error copy, and replacement cleanup.**
- [x] **Step 5: Keep the fictional sample explicitly simulated for now, but remove `UPLOADED_SAMPLE_TEXT` and every uploaded-document production use of `MODEL_NAME`.**
- [x] **Step 6: Run `npm test -- --run`, `npm run typecheck`, and `npm run build`; expect all existing and new tests PASS.**
- [x] **Step 7: Run browser QA for choose/drop, immediate rendering, backend pending/failed/ready, marking, hover/pin, replacement, and mobile view switching.**
- [x] **Step 8: Update the checklist, commit with `feat: connect PDF uploads to backend`, and push.**

### Task 5A: Persist and show the real visible-context checkpoint

**Files:**
- Create: `backend/src/redacted_backend/documents/preparation.py`
- Create: `backend/src/redacted_backend/db/repositories/visible_examples.py`
- Create: `backend/src/redacted_backend/api/context.py`
- Create: `backend/alembic/versions/0003_visible_examples.py`
- Create: `backend/tests/documents/test_preparation.py`, `backend/tests/api/test_context.py`
- Modify: `backend/src/redacted_backend/documents/visible_example.py`, `db/models.py`, `db/repositories/passages.py`, `main.py`
- Modify: `src/services/backendApi.ts`, `backendApi.test.ts`
- Modify: `src/features/workspace/DocumentWorkspace.tsx`, `DocumentWorkspace.test.tsx`, `src/styles/global.css`
- Modify: `README.md`, `CHECKLIST.md`

**Interfaces:**
- Produces: `prepare_visible_context(pdf_path, document, passages, target_id) -> PreparedVisibleContext` using the stored PDF, whole-document visible extraction, page-mask discovery, and the existing hidden-answer-free `build_visible_example`.
- Produces: `POST /v1/passages/{passage_id}:prepare-context -> VisibleContextSummary` with the same passage ID and revision, document date, length-cue bucket, estimated words, visible-word count, confirmed-mask count, and `context_ready`; no transcript or prediction text is returned.
- Persists one active `VisibleExample` payload per passage revision. Repeating the request for the same revision returns it without re-extraction; passage/date mutation invalidates it.
- Frontend adds `BackendApi.updateDocumentMetadata` and `prepareVisibleContext`. The existing uploaded-document sidebar owns the date form, preparation button, status, and summary while hover/pin remains owned by the current passage selection code.

- [x] **Step 1: Write failing pure tests proving page images feed mask discovery, an auto mask overlapping the manual target is deduplicated, covered embedded text never reaches the example, and OCR/mask/length failures return stable safe codes.**
- [x] **Step 2: Write failing API/persistence tests for missing date, same-revision reuse, revision-bound summary, persisted payload with no hidden-answer fields, correction/date invalidation, deleted passage/document, and no retrieval/model call boundary.**
- [x] **Step 3: Run the focused backend tests and verify the new preparation interfaces are missing.**
- [x] **Step 4: Implement rendering/preparation, migration, repository invalidation, route, and safe response contract; update Task 11's future analysis migration number to `0004_analysis_jobs.py`.**
- [x] **Step 5: Write and run failing frontend transport/component tests for the manual-visible-date request, Prepare visible context button, in-progress state, passage-linked ready summary, stable error copy, cancellation on replacement, and absence of prediction text over `.pdf-page-shell`.**
- [x] **Step 6: Implement the typed client and existing-design sidebar controls without adding a transcript, fake prediction, retrieval claim, or PDF overlay text.**
- [x] **Step 7: Run focused and full backend/frontend tests, lint/format, typecheck, build, both Compose configs, migrations, and browser QA using a real generated PDF.**
- [x] **Step 8: Update checklist/docs, commit with `feat: expose visible context checkpoint`, and push.**

### Task 5B: Automatically prove the visible document date

**Files:**
- Create: `backend/src/redacted_backend/documents/dates.py`, `backend/tests/documents/test_dates.py`
- Create: `backend/alembic/versions/0004_visible_date_evidence.py`
- Modify: `backend/src/redacted_backend/documents/preparation.py`, `domain/documents.py`, `db/models.py`, `db/repositories/documents.py`, `api/context.py`
- Modify: backend preparation/context/document tests and public PDF fixtures
- Modify: `src/services/backendApi.ts`, associated transport/hook tests
- Modify: `src/features/workspace/DocumentWorkspace.tsx`, its tests, and `src/styles/global.css`
- Modify: `README.md`, `CHECKLIST.md`, and Task 11's future migration number

**Interfaces:**
- Produces: `detect_visible_document_date(pages, discoveries) -> VisibleDateEvidence` from confirmed visible words in the first-page header, with a frozen parser version and exact page/line/word-box evidence.
- `prepare_visible_context(...) -> PreparedVisibleContext` always derives the prompt date from the same visible extraction used for the transcript; it never accepts user-entered, corpus-derived, model-inferred, hidden, or uncorroborated PDF-metadata dates.
- `DocumentRepository.set_detected_visible_date(...)` persists `visible_text` provenance and evidence without changing passage revision; conflicting later extraction fails closed.
- The website exposes no date input or date-mutation request. **Prepare visible context** becomes available after a real passage is saved and displays the detected date only after successful processing.

- [x] **Step 1: Write failing pure tests for ISO, month-name day-first/day-last, OCR words, repeated identical dates, ambiguous multiple dates, missing dates, body-only dates, and dates hidden by confirmed masks.**
- [x] **Step 2: Run focused date tests and confirm the detector is missing.**
- [x] **Step 3: Implement deterministic visible-header date detection and evidence without adding a model or third-party date service.**
- [ ] **Step 4: Write failing preparation/API tests proving automatic persistence, evidence provenance, same-revision reuse, conflict refusal, and removal of the manual metadata dependency.**
- [ ] **Step 5: Implement preparation persistence and migration; remove the manual metadata product route and update Task 11 to `0005_analysis_jobs.py`.**
- [ ] **Step 6: Write failing frontend tests proving there is no date input/request and one saved passage can directly prepare and display its automatically detected date.**
- [ ] **Step 7: Remove the manual date client/UI/styles and implement permanent processing copy and states.**
- [ ] **Step 8: Run full backend/frontend/container/migration/browser verification with the real generated PDF; update docs/checklist, commit with `feat: detect visible document dates`, and push.**

### Task 6: Frozen corpus snapshot, exclusions, preflight, and deterministic chunks

**Files:**
- Create: `backend/src/redacted_backend/corpus/snapshot.py`, `preflight.py`, `exclusions.py`, `chunker.py`, `manifest.py`, `cli.py`
- Create: `backend/corpus/exclusions/v1.yaml`, `backend/corpus/exclusions/README.md`
- Create: `backend/tests/corpus/test_snapshot.py`, `test_preflight.py`, `test_exclusions.py`, `test_chunker.py`, `test_manifest.py`
- Modify: `backend/pyproject.toml`, `backend/src/redacted_backend/scientific_config.py`, `README.md`, `CHECKLIST.md`

**Interfaces:**
- Produces: `resolve_snapshot(repo_id='HistoryLab/foiarchive', revision: str) -> SnapshotLock`; mutable revisions are rejected after resolution.
- Produces: `run_preflight(paths, minimum_free_gib=150, recommended_free_gib=250) -> PreflightReport`.
- Produces: `should_include(document, exclusion_manifest) -> InclusionDecision` with fail-closed unknown provenance.
- Produces: `chunk_document(document, max_words=300) -> list[CorpusChunk]` using paragraph/sentence/word splitting and no overlap.
- Produces CLI commands `corpus preflight`, `corpus lock-snapshot`, `corpus build-chunks`.

- [x] **Step 1: Write failing tests for immutable revision locking, checksum mismatch, 149/150/250 GiB preflight boundaries, CIA reading-room exclusion, 2019-2024 FRUS exclusion, unknown provenance, 300/301-word paragraphs, and deterministic chunk IDs.**
- [x] **Step 2: Run the focused corpus tests; expect missing modules.**
- [x] **Step 3: Implement the pure corpus pipeline and CLI without downloading the gated dataset in tests. Use small JSONL fixtures with the same nine public fields.**
- [x] **Step 4: Run `corpus preflight` against the current machine and record the result without writing corpus data.**
- [x] **Step 5: Run all backend/frontend tests; expect PASS.**
- [x] **Step 6: Update the checklist, commit with `feat: add reproducible corpus builder`, and push.**

**Execution gate after Task 6:** Satisfied. The user accepted the `HistoryLab/foiarchive` terms and supplied at least 150 GiB of local headroom. The immutable snapshot was downloaded and verified, then rejected as the complete paper corpus after schema and collection-count inspection.

### Task 6A: Reconcile and freeze the complete History Lab source

**Files:**
- Create: `backend/src/redacted_backend/corpus/historylab_export.py`, `frus_metadata.py`
- Create: `backend/tests/corpus/test_historylab_export.py`, `test_frus_metadata.py`
- Modify: `backend/src/redacted_backend/corpus/chunker.py`, `exclusions.py`, `manifest.py`, `cli.py`
- Modify: `backend/corpus/exclusions/v1.yaml`, `README.md`, `CHECKLIST.md`

**Interfaces:**
- Produces: `capture_historylab_corpus(base_url, output_root, resume=True) -> ApiCaptureManifest` with explicit collection labels, keyset pagination, atomic checkpoints, exact count checks before and after each collection, and per-file SHA-256.
- Produces: `load_frus_metadata(xml_path) -> FrusPublicationManifest` from the Office of the Historian's canonical volume IDs and publication years.
- Produces: `validate_capture_for_paper(capture, exclusions, frus) -> CorpusCoverageReport` that requires a non-empty `cables` collection, a verified whole-collection CIA reading-room exclusion, no unknown collections or FRUS IDs, and exactly ten present 2019-2024 FRUS volumes.
- Normalized records distinguish `corpus` (the archive collection used for filtering) from `source_url` (origin provenance). Downstream chunk and retrieval metadata use `corpus`.

- [x] **Step 1: Write failing tests for stable keyset pagination, interrupted-page recovery, duplicate/missing IDs, API count drift, safe paths, checksums, absent `cables`, unknown collections, FRUS longest-prefix matching, unmatched IDs, and the exact ten-volume invariant.**
- [x] **Step 2: Run the focused tests and verify they fail against the missing capture/FRUS interfaces.**
- [x] **Step 3: Implement the resumable API capture, normalized schema, official FRUS parser, coverage report, and CLI commands. Keep every large artifact outside Git and make partial output visibly incomplete.**
- [ ] **Step 4: Freeze the already downloaded official FRUS feed and record its checksum. When the public History Lab API is healthy, capture all collections and verify before/after exact counts and file checksums. Do not mix the API capture with the older Hugging Face export.**
- [ ] **Step 5: Populate the exclusion manifest from verified public metadata and run the full coverage gate. If the API remains unavailable or the archive contains a different number of post-cutoff FRUS volumes, stop with an explicit source gap.**
- [ ] **Step 6: Run focused and full backend/frontend verification, update the checklist, commit with `feat: freeze complete History Lab corpus source`, and push.**

### Task 7: BM25 index and context-query retrieval

**Files:**
- Create: `backend/src/redacted_backend/retrieval/types.py`, `bm25.py`, `context_query.py`
- Create: `backend/src/redacted_backend/corpus/build_bm25.py`
- Create: `backend/tests/retrieval/test_context_query.py`, `test_bm25.py`
- Modify: `compose.yaml`, `backend/src/redacted_backend/corpus/cli.py`, `backend/src/redacted_backend/domain/readiness.py`, `CHECKLIST.md`

**Interfaces:**
- Produces: `build_context_query(paragraph_window: str) -> str` with the vendored stop-word list.
- Produces: `Bm25Index.build(chunks, manifest)` and `Bm25Index.search(query, limit=300) -> list[RankedChunk]`.
- Readiness requires an index manifest matching the frozen corpus/scientific hashes.

- [x] **Step 1: Write failing tests for stop-word removal without stemming names, exact 300-result limit, deterministic tie order, incompatible manifest rejection, and context-query result provenance.**
- [x] **Step 2: Run focused tests; expect missing retrieval code.**
- [x] **Step 3: Add a pinned OpenSearch service/profile and implement index settings with standard Unicode tokenizer, vendored lowercase English stop words, `k1=1.2`, and `b=0.75`.**
- [ ] **Step 4: Build the real index only from a Task 6A capture that passes every coverage invariant, record counts/checksums/timing, and run one visible context query without a model call.**
- [ ] **Step 5: Run focused tests, readiness, and deterministic sample rebuild checks; expect PASS.**
- [ ] **Step 6: Update the checklist, commit with `feat: add corpus BM25 retrieval`, and push.**

### Task 8: Pinned BGE embeddings and exact-rescored dense retrieval

**Files:**
- Create: `backend/src/redacted_backend/retrieval/dense.py`
- Create: `backend/src/redacted_backend/corpus/build_dense.py`
- Create: `backend/tests/retrieval/test_dense.py`, `test_dense_recall.py`
- Modify: `backend/src/redacted_backend/corpus/cli.py`, `backend/src/redacted_backend/domain/readiness.py`, `CHECKLIST.md`

**Interfaces:**
- Produces: `BgeEncoder.encode_documents(texts)` and `encode_query(text)` pinned to an immutable model revision, L2-normalized.
- Produces: `DenseIndex.search(query, candidates=2000, limit=300) -> list[RankedChunk]` using HNSW candidate generation and float16 exact re-score.
- Produces: `measure_recall_at_300(queries, approximate, exact) -> float` with required result `>=0.98`.

- [ ] **Step 1: Write failing tests for query/document encoding distinction, normalization, stable rank tie-breaks, exact re-scoring, incompatible model revision, and `recall@300 < 0.98` readiness failure.**
- [ ] **Step 2: Run focused tests; expect missing dense interfaces.**
- [ ] **Step 3: Implement the pinned BGE encoder, memory-mapped float16 matrix, FAISS HNSW (`M=32`, `efConstruction=200`, `efSearch=512`), candidate map, and exact inner-product re-score.**
- [ ] **Step 4: Build real dense artifacts on the approved machine/volume and record revision, checksums, dimensions, timings, and recall sample.**
- [ ] **Step 5: Run focused tests and the quality gate; expect PASS with recall at least 0.98.**
- [ ] **Step 6: Update the checklist, commit with `feat: add BGE dense retrieval`, and push.**

### Task 9: Four-list retrieval, self-exclusion, RRF, and prompt documents

**Files:**
- Create: `backend/src/redacted_backend/retrieval/self_exclusion.py`, `fusion.py`, `service.py`, `serialization.py`
- Create: `backend/src/redacted_backend/domain/retrieval.py`
- Create: `backend/tests/retrieval/test_self_exclusion.py`, `test_fusion.py`, `test_service.py`, `test_serialization.py`
- Modify: `backend/src/redacted_backend/scientific_config.py`, `CHECKLIST.md`

**Interfaces:**
- Produces: `retrieve_realistic(example: VisibleExample, own_query: str) -> RetrievalTrace`.
- Produces: `fuse_documents(lists: FourRankedLists, rrf_k=60, limit=100) -> list[FusedDocument]`.
- Produces: `serialize_prompt_documents(documents, chunks_per_document=2) -> str` with the spec's exact markers.
- Produces: `SelfDocumentMatcher.match(upload, corpus_candidates) -> ExclusionDecision` and fail-closed ambiguity.

- [ ] **Step 1: Write failing tests proving exactly four 300-item calls, document RRF from each list's best chunk, chunk-level RRF selection of two excerpts, duplicate filename/opening merge, stable ties, top-100 limit, and `no_evidence_retrieved` when fusion is empty.**
- [ ] **Step 2: Add failing self-exclusion tests for known aliases, same-release near duplicate, multiple ambiguous releases, and a clearly unrelated document.**
- [ ] **Step 3: Add a failing serialization/token-budget test asserting the exact `[DOCUMENT n]`/`[EXCERPT n]` format and `prompt_too_large` without truncation.**
- [ ] **Step 4: Run focused tests; expect missing fusion service.**
- [ ] **Step 5: Implement the pure retrieval service and immutable trace persistence. Raw BM25/cosine scores may be stored but never compared across methods.**
- [ ] **Step 6: Run focused and full backend tests; expect PASS.**
- [ ] **Step 7: Update the checklist, commit with `feat: fuse realistic retrieval evidence`, and push.**

### Task 10: Frozen prompts and OpenAI model gateway

**Files:**
- Create: `backend/prompts/realistic_query_v1.txt`, `backend/prompts/realistic_prediction_v1.txt`
- Create: `backend/src/redacted_backend/models/prompts.py`, `token_budget.py`, `openai_gateway.py`
- Create: `backend/src/redacted_backend/domain/predictions.py`
- Create: `backend/tests/models/test_prompts.py`, `test_token_budget.py`, `test_openai_gateway.py`
- Modify: `backend/src/redacted_backend/scientific_config.py`, `CHECKLIST.md`

**Interfaces:**
- Produces: `render_query_prompt(example: VisibleExample) -> RenderedPrompt` and `render_prediction_prompt(example, retrieval) -> RenderedPrompt`.
- Produces: `OpenAIGateway.generate_query(example, idempotency_key) -> QueryResult`.
- Produces: `OpenAIGateway.generate_prediction(example, retrieval, idempotency_key) -> PredictionRecord`.
- Every request fixes `model='gpt-6-astra'`, reasoning `medium`, no system message/tools, and one user message.

- [ ] **Step 1: Copy the two Appendix F prompts verbatim into frozen files and write snapshot/hash tests against those files.**
- [ ] **Step 2: Write failing gateway tests using an injected HTTP/SDK transport: exact model/settings, one user message, no system/tools, query validation (one line, at most 30 words), refusal/empty/rate-limit handling, response ID/usage/cost capture, and no fallback.**
- [ ] **Step 3: Add failing idempotency tests asserting a retry returns the persisted result without a second provider request.**
- [ ] **Step 4: Run focused tests; expect missing model modules.**
- [ ] **Step 5: Implement prompt loading/rendering, target-model token counting, the Responses API gateway, and persistence boundary without making a real API call.**
- [ ] **Step 6: Run focused and full backend tests; expect PASS.**
- [ ] **Step 7: Update the checklist, commit with `feat: add frozen realistic model prompts`, and push.**

**Execution gate after Task 10:** Show the exact one-passage token estimate and maximum cost, then ask the user for model access and explicit approval before any real query/prediction request.

### Task 11: Idempotent analysis jobs, SSE, and the first real linked prediction

**Files:**
- Create: `backend/src/redacted_backend/jobs/messages.py`, `orchestrator.py`, `tasks.py`, `events.py`
- Create: `backend/src/redacted_backend/api/analysis.py`, `events.py`
- Create: `backend/alembic/versions/0005_analysis_jobs.py`
- Create: `backend/tests/jobs/test_orchestrator.py`, `test_events.py`, `backend/tests/api/test_analysis.py`
- Create: `src/services/httpPredictionService.ts`, `src/services/httpPredictionService.test.ts`
- Create: `src/hooks/usePredictionStream.ts`, `src/hooks/usePredictionStream.test.tsx`
- Modify: `src/services/predictionService.ts`, `src/features/workspace/DocumentWorkspace.tsx`
- Delete or test-confine: `src/hooks/usePredictionSimulation.ts`, `src/hooks/usePredictionSimulation.test.tsx`
- Modify: associated frontend tests, `README.md`, `CHECKLIST.md`

**Interfaces:**
- Produces: `schedule_passage_analysis(passage_id, expected_revision, idempotency_key) -> AnalysisJob`.
- Produces: `GET /v1/documents/{id}/events` SSE events `passage.status`, `passage.prediction`, `passage.error`, and heartbeat.
- Produces: `createHttpPredictionService(api: BackendApi) -> PredictionService` preserving `AbortSignal` cancellation.

- [ ] **Step 1: Write failing backend tests for stage checkpoints, same-key deduplication, worker retry after retrieval, revision mismatch cancellation, deletion cancellation, ordered events, and SSE reconnect from `Last-Event-ID`.**
- [ ] **Step 2: Write failing frontend tests asserting a real event becomes a `Prediction` with the same passage ID, hover moves it to `Selected passage`, it disappears from the ordinary list while selected, and no prediction text renders inside `.pdf-page-shell`.**
- [ ] **Step 3: Run focused backend/frontend tests; expect missing orchestration and HTTP service.**
- [ ] **Step 4: Implement Dramatiq orchestration, transactionally persisted stage/idempotency state, Redis event stream, SSE adapter, and frontend service injection.**
- [ ] **Step 5: After the paid-call gate is approved, run exactly one manually marked passage through query generation and prediction; record request IDs, prompt/config hashes, retrieval trace ID, usage, and cost.**
- [ ] **Step 6: Run all backend/frontend tests, typecheck, build, and browser QA for the first real passage; expect PASS and no simulated copy on that path.**
- [ ] **Step 7: Update the checklist, commit with `feat: stream real passage predictions`, and push.**

### Task 12: Automatic detection, correction, and whole-document processing

**Files:**
- Create: `backend/src/redacted_backend/documents/detection.py`
- Create: `backend/tests/documents/test_detection.py`, `backend/tests/jobs/test_document_analysis.py`
- Modify: `backend/src/redacted_backend/api/passages.py`, `analysis.py`
- Modify: `src/hooks/useBackendDocument.ts`, `src/features/workspace/workspaceReducer.ts`, `DocumentWorkspace.tsx`, `PassageOverlayLayer.tsx`
- Modify: associated frontend tests, `CHECKLIST.md`

**Interfaces:**
- Produces: `detect_passages(document_id) -> DetectionResult` with page-ordered grouped passages and coverage status.
- Produces: `schedule_document_analysis(document_id, concurrency) -> list[AnalysisJob]` where concurrency is bounded configuration.
- Manual correction uses Task 3 APIs and invalidates affected derived artifacts only.

- [ ] **Step 1: Write failing detector tests for black/white boxes, touching and line-wrapped rectangles, readable-word separation, cross-column separation, page order, and uncertain coverage.**
- [ ] **Step 2: Write failing whole-document tests for bounded concurrency, partial completion, one-passage failure, correction invalidation, stable IDs/order, cancellation, and no stale events after replacement.**
- [ ] **Step 3: Run focused tests; expect missing automatic detection flow.**
- [ ] **Step 4: Implement detection/grouping, correction UI using existing overlays, and whole-document scheduling without changing hover/pin/sidebar semantics.**
- [ ] **Step 5: Render golden PDFs and visually verify every overlay before running any paid whole-document analysis.**
- [ ] **Step 6: Run full backend/frontend tests, typecheck, build, and browser QA; expect PASS.**
- [ ] **Step 7: Update the checklist, commit with `feat: process all detected passages`, and push.**

### Task 13: Remove finished-path simulation and production hardening

**Files:**
- Delete or test-confine: `src/services/mockPredictionService.ts`, `src/data/sampleDocument.ts`, `src/data/sampleGeometry.json`, `public/fictional-briefing.pdf`, `scripts/generate-sample-pdf.mjs`
- Modify: `src/App.tsx`, `DocumentWorkspace.tsx`, prediction components and tests
- Create: `backend/tests/security/test_pdf_limits.py`, `test_log_redaction.py`, `test_retention.py`
- Modify: `backend/src/redacted_backend/config.py`, storage/jobs modules, `README.md`, `CHECKLIST.md`

**Interfaces:**
- Production build has no route that emits a fixture prediction.
- Uploaded PDFs and page images expire according to configured retention; deletion is idempotent and auditable.
- Normal logs exclude PDF text, prompt bodies, retrieved excerpts, and credentials.

- [ ] **Step 1: Write failing tests proving production imports cannot reach fixture predictions, UI labels real output `Model prediction`, security limits hold, logs redact sensitive bodies, and retention deletes expired artifacts without touching active references.**
- [ ] **Step 2: Run focused tests; expect failures while the simulated sample remains production-accessible.**
- [ ] **Step 3: Remove or test-confine simulation assets, finish retention/resource/logging behavior, and update all user copy and documentation truthfully. Preserve the existing visual layout; if `View interactive sample` remains, it must load a cited public PDF and execute the same real backend rather than returning stored fixture text.**
- [ ] **Step 4: Run `cd backend && uv run pytest`, backend type/lint checks, `npm test -- --run`, `npm run typecheck`, and `npm run build`; expect all PASS.**
- [ ] **Step 5: Run complete browser QA: upload, detection, correction, one/all prediction, hover/pin, card navigation, zoom, replacement/cancellation, keyboard, and responsive behavior.**
- [ ] **Step 6: Confirm no deployment occurred, update the checklist, commit with `feat: complete real retrieval workflow`, and push.**

## Final Verification Gate

- [ ] Confirm every design requirement maps to a completed task and every filled paper gap appears in the scientific manifest.
- [ ] Confirm the production runtime contains no hidden-answer/oracle/evaluation field or endpoint.
- [ ] Confirm every visible prediction maps to one stable passage ID and no prediction text is rendered over the PDF.
- [ ] Confirm the real smoke trace contains four 300-chunk lists, fused top 100 documents, up to two chunks each, exact prompt/model settings, provider IDs, and cost.
- [ ] Confirm all tests/builds and the latest browser QA pass from a clean checkout with documented external artifacts.
- [ ] Ask separately before any deployment or paid whole-document run.
