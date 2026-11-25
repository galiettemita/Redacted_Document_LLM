# Full-document Real Predictions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one production action durably schedule every unpredicted passage on the full four-list basis while enforcing one cumulative provider-spend ceiling across all paid bases.

**Architecture:** Reuse the current locked document scheduler, revision-bound jobs, Dramatiq queue, and SSE recovery. Add a general provider-cap setting backed by the existing durable claim ledger, expose basis readiness to the workspace, and choose selected-passage versus all-passage scheduling solely from backend readiness.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy, PostgreSQL, Dramatiq, Pydantic, React 19, TypeScript, Vitest.

**Spec:** `docs/superpowers/specs/2026-11-25-full-document-predictions-design.md`

## Global Constraints

- Do not make a provider request, raise the current $10.00 ceiling, or enable a new basis.
- Bulk mode is allowed only for `best_available_foiarchive_v1` or `paper_exact_v1` readiness.
- Keep staged prediction selected-passage-only and preserve its existing scientific provenance.
- Reuse current jobs/results and never reschedule a completed current revision.
- Keep the fictional demo and customer-facing prediction cards unchanged.

## Review Focus

- A forged or stale scheduling response must not make the browser stream another passage or revision.
- A readiness transition must not convert an already scheduled job's immutable basis.
- An uncertain provider response must retain its maximum reservation for every paid basis.
- Reopening a document with some completed and some active jobs must recover all current jobs without duplicating them.
- A full-document click while another request is pending must schedule only once.

---

### Task 1: Generalize the durable provider spending ceiling

**Files:**
- Modify: `backend/src/redacted_backend/config.py`
- Modify: `backend/src/redacted_backend/jobs/spend_budget.py`
- Modify: `backend/src/redacted_backend/jobs/runtime.py`
- Modify: `backend/src/redacted_backend/api/readiness.py`
- Modify: `compose.yaml`
- Test: `backend/tests/jobs/test_spend_budget.py`
- Test: `backend/tests/jobs/test_runtime_provider.py`
- Test: `backend/tests/test_readiness.py`

**Interfaces:**
- Consumes: existing `staged_spend_claims` ledger and `StagedSpendBudget.reserve/settle` semantics.
- Produces: `Settings.provider_spend_cap_usd`, `Settings.effective_provider_spend_cap_usd`, and a budget injected into every `OpenAIGateway` runtime.

- [ ] **Step 1:** Add failing tests proving full best-available construction requires a cap, uses the shared ledger, and readiness closes when no effective cap exists.
- [ ] **Step 2:** Run the focused tests and confirm failures are caused by the missing general cap behavior.
- [ ] **Step 3:** Implement the general setting with staged-setting fallback, inject the existing ledger into every provider runtime, and pass the setting through Compose with a safe empty default.
- [ ] **Step 4:** Run focused backend tests green, then commit `feat: enforce provider cap for full predictions`.

### Task 2: Expose full-basis readiness to the production workspace

**Files:**
- Modify: `src/services/backendApi.ts`
- Modify: `src/services/backendApi.test.ts`
- Modify: `src/features/workspace/DocumentWorkspace.tsx`
- Modify: `src/features/workspace/DocumentWorkspace.test.tsx`

**Interfaces:**
- Consumes: `/v1/readiness` fields `prediction_bases` and `strongest_ready_basis`.
- Produces: a fail-closed `fullDocumentPredictionReady` UI decision.

- [ ] **Step 1:** Add failing transport/component tests for complete readiness parsing, staged selected-passage mode, and full-basis all-passage mode.
- [ ] **Step 2:** Run the focused tests red.
- [ ] **Step 3:** Extend the typed readiness contract and fetch it in the workspace without displaying implementation diagnostics.
- [ ] **Step 4:** Run focused frontend tests green, then commit `feat: gate full-document prediction on readiness`.

### Task 3: Schedule and stream every remaining passage

**Files:**
- Modify: `src/features/workspace/DocumentWorkspace.tsx`
- Modify: `src/features/workspace/DocumentWorkspace.test.tsx`
- Test: `backend/tests/api/test_analysis.py`

**Interfaces:**
- Consumes: `BackendApi.scheduleDocumentAnalysis(documentId, options)` and current prepared revisions.
- Produces: validated job-basis allow-list for every recoverable/new full-document job and progressive retained prediction cards.

- [ ] **Step 1:** Add failing UI tests proving one full-mode click schedules the document, validates every returned passage/revision/basis, streams multiple jobs, retains earlier results, and suppresses duplicate clicks.
- [ ] **Step 2:** Add/verify backend tests proving document scheduling reuses completed jobs and creates queued jobs for every remaining passage under a full basis.
- [ ] **Step 3:** Run focused tests red for the missing frontend bulk behavior.
- [ ] **Step 4:** Implement the bulk handler, **Predict all passages**/**Predict remaining passages** copy, and fail-closed response validation.
- [ ] **Step 5:** Run focused tests green, then commit `feat: predict every remaining passage`.

### Task 4: Verify, deploy safely, and record progress

**Files:**
- Modify: `CHECKLIST.md`

**Interfaces:**
- Consumes: Tasks 1-3 and the live no-spend local stack.
- Produces: tested/deployed code with unchanged provider claims/spend and documented dense-build progress.

- [ ] **Step 1:** Run the full backend/frontend suites, typecheck, production build, Compose rendering, and `git diff --check`.
- [ ] **Step 2:** Record provider claim/spend counts, rebuild API/worker, verify health/readiness/frontend delivery, and prove the counts did not change.
- [ ] **Step 3:** Record the 23-checkpoint/1,577,156-row dense progress and full-document feature in `CHECKLIST.md`.
- [ ] **Step 4:** Commit/push documentation and perform the final whole-change review.

