# Real Model-Only Predictions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display a genuine GPT-6 Astra prediction for an uploaded passage without requiring a retrieval corpus, using Andrew Tang's frozen Appendix F closed-book prompt and durable production provenance.

**Architecture:** Add an immutable prediction basis to every analysis job, then make readiness select the strongest separately approved operational basis. The worker builds a basis-specific runtime: `model_only_v1` invokes exactly one closed-book generation and never initializes query generation, OpenSearch, BM25, BGE, FAISS, or retrieval; the existing `paper_exact_v1` path remains closed and unchanged until its corpus is ready.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, SQLAlchemy/Alembic, Dramatiq/Redis, OpenAI Responses API, React 19, TypeScript, Vite, Vitest.

**Spec:** `docs/superpowers/specs/2026-11-20-staged-real-predictions-design.md`

## Global Constraints

- Preserve the fictional interactive demo unchanged and keep every invented result labelled **Simulated prediction**.
- Use the verbatim Appendix F closed-book prompt on PDF page 26 of `/Users/galiettemita/Redacted_Document_LLM/Redaction_RAG_NAACL_v3.pdf`.
- Keep `allow_paid_model_calls` as the master switch and require a separate `allow_model_only_calls` approval; both default to `false`.
- Make no provider request during implementation or automated verification.
- Never silently change a job's basis after scheduling and never downgrade a failed retrieval job to model-only.
- Keep `paper_exact_v1` and `best_available_foiarchive_v1` separately gated; this plan does not build the best-available corpus indexes.
- Keep customer cards free of prompts, evidence, token counts, costs, and research diagnostics.
- Before the first live call, count the exact request, present expected and maximum cost, and obtain explicit approval for one passage.

## Review Focus

- Replaying the same idempotency key after readiness changes must return the original job and basis, not schedule a stronger or cheaper basis.
- A `model_only_v1` job must complete when every corpus path is absent and must make zero query/retrieval calls.
- Legacy jobs must migrate to the known `paper_exact_v1` path without losing events or acquiring a fabricated model-only result.
- Passage revision, deletion, or expiry must still cancel before another external request, and retry must reuse the durable prediction payload.
- Unknown or mismatched SSE basis values must fail closed in the browser while the fictional demo remains fully usable.

## File Structure

- `backend/src/redacted_backend/domain/analysis.py`: owns `PredictionBasis`, job records, and basis-aware stages.
- `backend/alembic/versions/0010_prediction_basis.py`: backfills legacy jobs and makes their basis durable.
- `backend/src/redacted_backend/db/models.py` and `db/repositories/analysis.py`: persist the immutable basis and include it in status/prediction events.
- `backend/prompts/closed_book_prediction_v1.txt` and `models/prompts.py`: freeze and render Appendix F's closed-book request.
- `backend/src/redacted_backend/domain/predictions.py` and `models/openai_gateway.py`: validate and execute retrieval-free predictions.
- `backend/src/redacted_backend/domain/readiness.py`, `api/readiness.py`, and `api/analysis.py`: report basis readiness and choose one basis transactionally.
- `backend/src/redacted_backend/jobs/`: construct, cache, and execute basis-specific runtimes.
- `src/types.ts`, `src/services/backendApi.ts`, and `src/services/httpPredictionService.ts`: carry the basis through typed HTTP/SSE boundaries.
- `src/features/predictions/PredictionCard.tsx` and `SelectedPassage.tsx`: show concise basis provenance.

---

### Task 1: Persist an immutable prediction basis

**Files:**
- Create: `backend/alembic/versions/0010_prediction_basis.py`
- Modify: `backend/src/redacted_backend/domain/analysis.py`
- Modify: `backend/src/redacted_backend/db/models.py`
- Modify: `backend/src/redacted_backend/db/repositories/analysis.py`
- Test: `backend/tests/security/test_prediction_basis_migration.py`
- Test: `backend/tests/api/test_analysis.py`

**Interfaces:**
- Produces: `PredictionBasis` values `model_only_v1`, `best_available_foiarchive_v1`, and `paper_exact_v1`.
- Produces: required `AnalysisJobRecord.prediction_basis: PredictionBasis` and matching database column/event field.
- Consumes: existing `AnalysisJobRepository.schedule(...)`; add required `prediction_basis: PredictionBasis` without changing same-key replay semantics.

- [x] **Step 1: Write failing migration and repository tests** proving upgrade backfills every legacy job, stored prediction payload, and existing analysis-event payload as `paper_exact_v1`; downgrade removes only the new job column; new jobs persist the supplied basis; new status events include it; and a same-key replay returns the original basis even when a different basis is offered later.
- [x] **Step 2: Run `cd backend && uv run pytest tests/security/test_prediction_basis_migration.py tests/api/test_analysis.py -v`; expect failures for the missing enum, migration, and field.**
- [x] **Step 3: Implement `PredictionBasis`, migration `0010_prediction_basis`, the non-null model column, repository persistence, and event serialization.** Backfill is `paper_exact_v1` because every pre-migration executable job belongs to the existing retrieval-only runtime; do not infer from result content.
- [x] **Step 4: Run the focused tests and `cd backend && uv run alembic upgrade head`; expect PASS and schema head `0010_prediction_basis`.**
- [x] **Step 5: Commit with `feat: persist prediction basis` and push.**

### Task 2: Freeze Appendix F and add retrieval-free model generation

**Files:**
- Create: `backend/prompts/closed_book_prediction_v1.txt`
- Modify: `backend/src/redacted_backend/models/prompts.py`
- Modify: `backend/src/redacted_backend/domain/predictions.py`
- Modify: `backend/src/redacted_backend/models/openai_gateway.py`
- Test: `backend/tests/models/test_prompts.py`
- Test: `backend/tests/models/test_openai_gateway.py`

**Interfaces:**
- Produces: `render_model_only_prediction_prompt(example: VisibleExample) -> RenderedPrompt` with version `closed-book-prediction-v1`.
- Produces: `OpenAIGateway.generate_model_only_prediction(example, idempotency_key) -> PredictionRecord`.
- Changes: `PredictionRecord.prediction_basis: PredictionBasis`; `retrieval_trace_sha256` is `None` only for `model_only_v1` and is a 64-character SHA-256 for both retrieval bases.

- [x] **Step 1: Write a failing prompt snapshot test** against the exact Appendix F page-26 text beginning `Below is the declassified CIA intelligence bulletin of {date}.` and ending `{before}[[TARGET]]{after}`; assert the same visible date, coarse length hint, and target serialization used by the paper.
- [x] **Step 2: Write failing gateway/domain tests** proving model-only creates the existing fixed Responses request, performs one token count and one generation, persists idempotently, records `model_only_v1`, has no retrieval hash, rejects an empty/refused/incomplete response, and rejects a null retrieval hash for either retrieval basis.
- [x] **Step 3: Run `cd backend && uv run pytest tests/models/test_prompts.py tests/models/test_openai_gateway.py -v`; expect the closed-book interfaces to be missing.**
- [x] **Step 4: Add the frozen template, renderer, conditional `PredictionRecord` validation, and `generate_model_only_prediction`; reuse `_request`, `_execute`, cost, usage, provider-ID, and generation-store behavior.**
- [x] **Step 5: Run the focused tests; expect PASS and no network request.**
- [x] **Step 6: Commit with `feat: add closed-book prediction gateway` and push.**

### Task 3: Select and report separately approved prediction bases

**Files:**
- Modify: `.env.example`
- Modify: `backend/src/redacted_backend/config.py`
- Modify: `backend/src/redacted_backend/domain/readiness.py`
- Modify: `backend/src/redacted_backend/api/readiness.py`
- Modify: `backend/src/redacted_backend/api/analysis.py`
- Modify: `backend/src/redacted_backend/db/repositories/analysis.py`
- Test: `backend/tests/test_readiness.py`
- Test: `backend/tests/api/test_analysis.py`

**Interfaces:**
- Produces: settings `allow_model_only_calls`, `allow_best_available_calls`, and `allow_paper_exact_calls`, each defaulting to `False`.
- Produces: `BasisReadiness(operational, approved, ready, detail)`, `ReadinessResponse.prediction_bases`, and `strongest_ready_basis`.
- Produces: `resolve_prediction_basis_readiness(...)` and `select_prediction_basis(...)` with priority exact, best-available, then model-only.
- Produces: when no basis is ready, a terminal blocked job records `model_only_v1` plus its blockers; it is a candidate for the next intentional run, never an automatically upgradable job.

- [x] **Step 1: Write failing readiness tests** proving model-only is operational without corpus artifacts, requires database/worker/key/master approval/model-only approval, reports operational separately from approval, leaves unimplemented best-available unready, and does not make `paper_exact_v1` ready without the existing exact corpus gate.
- [x] **Step 2: Write failing scheduling tests** proving selection and persistence occur in the scheduling transaction, model-only starts at `prediction`, retrieval bases start at `query`, no ready basis creates a durable blocked job, and corpus completion cannot switch an existing or unapproved job to a two-call basis.
- [x] **Step 3: Run `cd backend && uv run pytest tests/test_readiness.py tests/api/test_analysis.py -v`; expect failures for the new readiness contract.**
- [x] **Step 4: Implement settings, additive readiness response, deterministic selector, basis-aware blockers, transactional persistence, and initial-stage selection while preserving current component fields for compatibility.**
- [x] **Step 5: Run the focused tests; expect PASS.**
- [x] **Step 6: Commit with `feat: select approved prediction basis` and push.**

### Task 4: Execute model-only jobs without constructing retrieval

**Files:**
- Modify: `backend/src/redacted_backend/jobs/orchestrator.py`
- Modify: `backend/src/redacted_backend/jobs/runtime.py`
- Modify: `backend/src/redacted_backend/jobs/runtime_provider.py`
- Modify: `backend/src/redacted_backend/jobs/execution.py`
- Modify: `backend/src/redacted_backend/jobs/tasks.py`
- Test: `backend/tests/jobs/test_orchestrator.py`
- Test: `backend/tests/jobs/test_runtime_provider.py`
- Test: `backend/tests/jobs/test_execution.py`
- Test: `backend/tests/jobs/test_worker_import.py`

**Interfaces:**
- Changes: `AnalysisRuntimeProvider(builder: Callable[[PredictionBasis], AnalysisOrchestrator]).get(basis)` caches successful runtimes independently by basis and retries only the failed basis.
- Produces: `build_analysis_orchestrator(settings, prediction_basis, database=...)`; model-only requires no corpus/OpenSearch/dense configuration, while retrieval bases retain their gates.
- Changes: `AnalysisOrchestrator` dispatches from the persisted job basis and accepts `retrieval: RetrievalService | None`.

- [ ] **Step 1: Write failing orchestrator tests** proving a model-only job calls `generate_model_only_prediction` exactly once, never calls query/retrieval, persists no query or trace, emits basis in prediction/status events, resumes a stored prediction without a second call, and still cancels on revision/deletion/expiry before generation.
- [ ] **Step 2: Write failing runtime/provider tests** proving model-only constructs with database/key/two approvals alone, exact retrieval retains every existing artifact probe, best-available fails closed until implemented, per-basis caches do not contaminate each other, and sanitized retry payloads contain no secrets.
- [ ] **Step 3: Run `cd backend && uv run pytest tests/jobs -v`; expect basis-aware execution failures.**
- [ ] **Step 4: Implement basis-specific orchestration, runtime construction, provider caching, and task lookup; keep the selected basis immutable on every retry and raise instead of falling back.**
- [ ] **Step 5: Run `cd backend && uv run pytest tests/jobs -v`; expect PASS and no network request.**
- [ ] **Step 6: Commit with `feat: run model-only prediction jobs` and push.**

### Task 5: Show honest basis provenance in the existing frontend

**Files:**
- Modify: `src/types.ts`
- Modify: `src/services/backendApi.ts`
- Modify: `src/services/httpPredictionService.ts`
- Modify: `src/features/predictions/PredictionCard.tsx`
- Modify: `src/features/predictions/SelectedPassage.tsx`
- Test: `src/services/backendApi.test.ts`
- Test: `src/services/httpPredictionService.test.ts`
- Test: `src/features/predictions/PredictionSidebar.test.tsx`
- Test: `src/features/workspace/DocumentWorkspace.test.tsx`
- Test: `src/features/demo/FictionalDemoWorkspace.test.tsx`

**Interfaces:**
- Produces: frontend `PredictionBasis` matching the backend strings and `Prediction.predictionBasis` for model output.
- Displays: `Model-only prediction — no archive evidence used`, `Archive-assisted prediction — available FOIArchive snapshot used`, or `Paper-method prediction`.

- [ ] **Step 1: Write failing service tests** requiring `prediction_basis` in model SSE events, rejecting unknown/missing values, and mapping it to `Prediction.predictionBasis` without exposing usage/cost in rendered cards.
- [ ] **Step 2: Write failing component/workspace tests** proving model-only provenance appears in both card and selected views, prediction text stays off the PDF, all-passage scheduling still works, and the fictional modal still shows only **Simulated prediction**.
- [ ] **Step 3: Run `npm test -- --run src/services/httpPredictionService.test.ts src/features/predictions/PredictionSidebar.test.tsx src/features/workspace/DocumentWorkspace.test.tsx src/features/demo/FictionalDemoWorkspace.test.tsx`; expect provenance failures.**
- [ ] **Step 4: Implement the typed basis boundary and one shared `predictionProvenanceLabel(...)` formatter used by both real prediction views.**
- [ ] **Step 5: Run the focused tests, `npm run typecheck`, and `npm run build`; expect PASS.**
- [ ] **Step 6: Commit with `feat: label real prediction basis` and push.**

### Task 6: Verify, document, and stop at the paid-call gate

**Files:**
- Modify: `README.md`
- Modify: `CHECKLIST.md`
- Modify: `docs/superpowers/plans/2026-11-20-real-model-only-predictions.md`

**Interfaces:**
- Produces: a no-spend production path ready for one explicitly approved live passage.
- Defers: the best-available FOIArchive importer/index plan and every paid provider request.

- [ ] **Step 1: Run `cd backend && uv run pytest -q`; expect all backend tests PASS.**
- [ ] **Step 2: Run `npm test -- --run && npm run typecheck && npm run build`; expect all frontend checks PASS.**
- [ ] **Step 3: Run fresh upgrade/downgrade/re-upgrade migration checks and the loopback upload → detection → context → model-only queued-job → SSE path with an injected fake transport; assert zero query/retrieval calls and no real provider traffic.**
- [ ] **Step 4: Update README and CHECKLIST with the completed no-spend milestone, exact activation environment variables, and the still-unchecked live-call gate.**
- [ ] **Step 5: Mark completed plan boxes, run `git diff --check`, commit with `feat: ship no-spend model-only prediction path`, and push.**
- [ ] **Step 6: With a real key configured but generation still disabled, count the exact one-passage Appendix F request, calculate expected/maximum price from the frozen pricing table, show both numbers to the user, and stop for explicit approval before enabling or sending one live generation.**

## Deferred Follow-up Plan

After this milestone is production-ready, write a separate plan for `best_available_foiarchive_v1`: exact v2 audit pinning, streaming import, `historylab-unlabeled` manifest, resumable BM25, deterministic dense shards, global exact rescoring, and the frozen 100-query recall@300 certification. That subsystem is intentionally excluded here so the first real on-screen model prediction is not delayed by a multi-million-document index build.
