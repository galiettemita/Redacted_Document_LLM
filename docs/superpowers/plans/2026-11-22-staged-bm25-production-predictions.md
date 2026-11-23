# Staged BM25 Production Predictions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ordinary uploaded-PDF Predict control produce one real GPT prediction using two independent full-corpus BM25 searches while enforcing a cumulative $10.00 provider ceiling.

**Architecture:** Add a permanent `best_available_foiarchive_bm25_v1` basis with its own two-list trace and BM25-only retrieval service, leaving the existing four-list bases unchanged. Gate scheduling and worker construction on verified corpus/BM25 readiness, a server-side one-prediction document limit, paid-call flags, and a durable provider-spend ledger; keep the customer-facing card generic.

**Tech Stack:** Python 3.12, Pydantic, SQLAlchemy/Alembic, PostgreSQL/SQLite tests, FastAPI, Dramatiq, OpenSearch, OpenAI Responses API, React/TypeScript, Vitest.

**Spec:** `docs/superpowers/specs/2026-11-22-staged-bm25-prediction-and-durable-dense-build-design.md`

## Global Constraints

- Use basis `best_available_foiarchive_bm25_v1`; never represent it as `best_available_foiarchive_v1` or `paper_exact_v1`.
- Search all 14,549,269 indexed passages through two independent BM25 lists: `context_bm25` and `own_bm25`.
- Preserve self-document exclusion, document-level RRF `k=60`, top 100 documents, at most two excerpts per document, and immutable backend provenance.
- Enforce `REDACTED_BM25_STAGE_MAX_PREDICTIONS_PER_DOCUMENT=1` on the server.
- Enforce `REDACTED_STAGED_BM25_SPEND_CAP_USD=10.00` cumulatively and durably; missing, invalid, exhausted, or insufficient budgets fail closed before provider dispatch.
- Keep all paid-call flags false during implementation, automated tests, retrieval probes, and no-spend deployment verification.
- Show no staged basis, BM25, dense-build, method, implementation, cost, query, evidence, or limitation label in the customer-facing prediction card.
- Preserve the fictional demo and every existing prediction basis.

## Review Focus

- Two concurrent requests for the first and second passage must not bypass the one-prediction document limit; cover in Task 4.
- A process crash or timeout after provider dispatch must retain the maximum reservation and must never resend the uncertain request; cover in Task 2.
- Lowering the configured cap below already committed spend must block all new attempts without corrupting prior claims; cover in Task 2.
- A valid BM25 manifest with a wrong live OpenSearch count or failed search probe must close only staged readiness; cover in Task 3.
- A staged prediction received from SSE must render without a provenance or implementation label in both selected and list cards; cover in Task 5.

---

### Task 1: Add an honest two-list retrieval contract

**Files:**
- Modify: `backend/src/redacted_backend/domain/analysis.py`
- Modify: `backend/src/redacted_backend/domain/retrieval.py`
- Modify: `backend/src/redacted_backend/retrieval/fusion.py`
- Modify: `backend/src/redacted_backend/retrieval/service.py`
- Modify: `backend/src/redacted_backend/models/prompts.py`
- Modify: `backend/src/redacted_backend/models/openai_gateway.py`
- Modify: `backend/src/redacted_backend/jobs/orchestrator.py`
- Test: `backend/tests/retrieval/test_fusion.py`
- Test: `backend/tests/retrieval/test_service.py`
- Test: `backend/tests/models/test_openai_gateway.py`
- Test: `backend/tests/jobs/test_orchestrator.py`

**Interfaces:**
- Consumes: `Bm25Index.search(...)`, prepared visible context, generated own query, existing self-exclusion and fusion.
- Produces: `PredictionBasis.BEST_AVAILABLE_FOIARCHIVE_BM25_V1`, `TwoRankedLists`, `Bm25RetrievalTrace`, `RetrievalTraceLike`, and `Bm25RetrievalService.retrieve_realistic(...) -> Bm25RetrievalTrace`.

- [x] **Step 1: Write failing retrieval tests** proving exactly two independent non-empty BM25 searches, no dense fields, self-exclusion, deterministic RRF, and rejection of an empty own query or empty post-filter evidence.
- [x] **Step 2: Run the focused retrieval tests and verify they fail because the new types and service do not exist.**
- [x] **Step 3: Implement the two-list models and BM25 service, and generalize fusion/candidate collection over the shared `named()` list contract without changing four-list behavior.**
- [x] **Step 4: Write failing gateway/orchestrator tests** proving staged traces render through the existing evidence-bound prompt, persist and resume as the two-list type, and retain the staged basis in the final payload.
- [x] **Step 5: Run those tests red, implement the minimum trace union parsing and basis-aware generation, then run all four focused files green.**
- [x] **Step 6: Commit and push with `feat: add staged bm25 retrieval contract`.**

### Task 2: Enforce the durable $10 provider ceiling

**Files:**
- Create: `backend/alembic/versions/0011_staged_provider_spend.py`
- Create: `backend/src/redacted_backend/jobs/spend_budget.py`
- Create: `backend/tests/jobs/test_spend_budget.py`
- Modify: `backend/src/redacted_backend/db/models.py`
- Modify: `backend/src/redacted_backend/config.py`
- Modify: `backend/src/redacted_backend/models/openai_gateway.py`
- Modify: `backend/src/redacted_backend/jobs/runtime.py`
- Test: `backend/tests/models/test_openai_gateway.py`
- Test: `backend/tests/security/test_prediction_basis_migration.py`

**Interfaces:**
- Consumes: exact pre-dispatch input-token count, `estimate_maximum_cost(...)`, actual `TokenUsage`, and configured `Decimal('10.00')` ceiling.
- Produces: `StagedSpendBudget.reserve(scope, key, request_sha256, maximum_cost_usd)`, `settle(..., actual_cost_usd)`, immutable per-attempt claims, and atomic cumulative committed spend.

- [x] **Step 1: Write failing migration and budget tests** for fresh/upgrade/downgrade schema, atomic first claim, idempotent replay, cap exhaustion, cap reduction, completed settlement, and uncertain maximum retention.
- [x] **Step 2: Run the focused tests and verify missing schema/budget failures.**
- [x] **Step 3: Implement integer-microdollar ledger rows and a locked singleton aggregate so reserve happens before dispatch and settle can release only the unused maximum after a completed response.**
- [x] **Step 4: Write failing gateway tests** proving maximum cost is reserved after token counting but before transport, actual cost settles on success, and transport uncertainty leaves the maximum committed.
- [x] **Step 5: Run those tests red, inject the budget only into the staged runtime gateway, and run the focused budget/gateway/migration tests green.**
- [x] **Step 6: Commit and push with `feat: enforce staged provider spend ceiling`.**

### Task 3: Add staged readiness and worker construction

**Files:**
- Modify: `backend/src/redacted_backend/config.py`
- Modify: `backend/src/redacted_backend/api/readiness.py`
- Modify: `backend/src/redacted_backend/domain/readiness.py`
- Modify: `backend/src/redacted_backend/jobs/runtime.py`
- Modify: `backend/src/redacted_backend/jobs/runtime_provider.py`
- Modify: `compose.yaml`
- Test: `backend/tests/test_readiness.py`
- Test: `backend/tests/jobs/test_runtime_provider.py`

**Interfaces:**
- Consumes: verified best-available corpus and BM25 manifests, exact live count, real BM25 probe, API key, worker/database readiness, staged flags, and positive spend cap.
- Produces: `resolve_staged_bm25_corpus_readiness(...)`, staged basis priority, and a BM25-only `AnalysisOrchestrator`.

- [x] **Step 1: Write failing readiness tests** for manifest identity, missing path, live-count mismatch, probe failure, independent approval, positive spend cap, and priority below full four-list but above model-only.
- [x] **Step 2: Run readiness tests red, implement staged readiness/configuration, and run them green.**
- [x] **Step 3: Write failing runtime-provider tests** proving only the staged basis loads/probes BM25 without requiring dense artifacts and refuses construction without flags or budget.
- [x] **Step 4: Run runtime tests red, implement staged construction and Compose variables with safe false/empty defaults, then run focused tests green.**
- [x] **Step 5: Commit and push with `feat: wire staged bm25 runtime readiness`.**

### Task 4: Enforce one staged prediction per document

**Files:**
- Modify: `backend/src/redacted_backend/db/repositories/analysis.py`
- Modify: `backend/src/redacted_backend/api/analysis.py`
- Test: `backend/tests/api/test_analysis.py`
- Test: `backend/tests/jobs/test_orchestrator.py`

**Interfaces:**
- Consumes: persisted passage order, active/completed staged jobs, and `bm25_stage_max_predictions_per_document`.
- Produces: a durable `staged_prediction_limit` blocker before dispatch/retrieval/provider work; failed/cancelled jobs do not consume the limit.

- [x] **Step 1: Write failing API tests** for first eligible passage, blocked later passage, idempotent replay, failed/cancelled release, raised limit, and concurrent scheduling serialized by a document lock.
- [x] **Step 2: Run the focused API tests and verify the limit can currently be bypassed.**
- [x] **Step 3: Implement the repository count/order query and transaction-scoped document lock, add the blocker before scheduling, and keep dispatch limited to queued jobs.**
- [x] **Step 4: Run focused API/orchestrator tests green.**
- [x] **Step 5: Commit and push with `feat: limit staged predictions per document`.**

### Task 5: Ship the generic UI and full no-spend loopback

**Files:**
- Create: `backend/tests/integration/test_staged_bm25_loopback.py`
- Modify: `src/types.ts`
- Modify: `src/services/httpPredictionService.ts`
- Modify: `src/features/predictions/predictionProvenance.ts`
- Modify: `src/features/predictions/PredictionCard.tsx`
- Modify: `src/features/predictions/SelectedPassage.tsx`
- Modify: `src/features/predictions/PredictionSidebar.test.tsx`
- Modify: `src/services/httpPredictionService.test.ts`
- Modify: `README.md`
- Modify: `CHECKLIST.md`

**Interfaces:**
- Consumes: staged analysis/SSE payload and the existing upload, prepare, Predict, and fictional-demo flows.
- Produces: a generic staged prediction card, a fake-transport two-stage upload-to-SSE proof, and deployment instructions with all paid defaults false.

- [x] **Step 1: Write failing frontend tests** that accept the staged basis and show its prediction text in list/selected cards with no provenance element or implementation wording.
- [x] **Step 2: Run frontend tests red, implement the basis decoder and nullable label rendering, then run focused frontend tests green.**
- [x] **Step 3: Write a failing integration test** covering PDF upload, detection, preparation, schedule, fake GPT query, two real BM25 searches, fused evidence, fake GPT prediction, durable spend reserve/settle, SSE, and second-passage blocking.
- [x] **Step 4: Run the integration test red, complete the narrow wiring gaps, and run it green without network model calls.**
- [x] **Step 5: Run the complete backend/frontend suites, typecheck, production build, Compose validation, migration upgrade/downgrade/upgrade, and `git diff --check` with paid flags false.**
- [x] **Step 6: Configure the untracked $10.00 ceiling, verify local readiness with flags still false, update the checklist, commit, and push with `feat: ship staged bm25 predictions`.**

### Task 6: Enable the authorized local prediction path

**Files:**
- Runtime configuration only: untracked `.env`; no API key or secret enters Git.
- Modify: `CHECKLIST.md` after verified deployment.

**Interfaces:**
- Consumes: verified no-spend deployment, existing user API key, `10.00` cap, staged flags, and a user-uploaded PDF.
- Produces: a ready local Predict control for the first prepared redaction and durable cost/provenance records.

- [ ] **Step 1: Recreate backend and worker with the master and staged flags enabled and the hard ceiling fixed at `10.00`; leave all other basis flags unchanged.**
- [ ] **Step 2: Verify readiness selects the staged basis, the BM25 live count is 14,549,269, and no provider call occurs during readiness.**
- [ ] **Step 3: Leave the site ready for the user's uploaded document; on Predict, allow the authorized two-stage call and verify the prediction event, ledger settlement, and generic UI.**
- [ ] **Step 4: Update the checklist with actual readiness/call evidence and commit/push any documentation-only state change.**
