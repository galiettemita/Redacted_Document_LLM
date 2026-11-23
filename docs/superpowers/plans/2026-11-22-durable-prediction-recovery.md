# Durable Prediction Recovery Implementation Plan

> Execute test-first in the isolated `fix/durable-prediction-recovery` worktree. Do not make a real provider call.

**Goal:** Make the real **Predict passage** action schedule atomically, display every accepted result, and recover stored current-revision predictions without duplicate spend.

**Architecture:** Add document-level schedule/state contracts over the existing analysis tables, locks, dispatcher, and SSE event log. Move the uploaded-document browser from passage fan-out to the document contract, recover jobs after context preparation, and lock conflicting customer mutations while accepted work is active.

**Tech stack:** FastAPI, SQLAlchemy, PostgreSQL/SQLite tests, Pydantic, React, TypeScript, Vitest, Testing Library.

---

### Task 1: Add atomic document scheduling and durable recovery

**Files:**
- Modify: `backend/src/redacted_backend/domain/analysis.py`
- Modify: `backend/src/redacted_backend/db/repositories/analysis.py`
- Modify: `backend/src/redacted_backend/api/analysis.py`
- Test: `backend/tests/api/test_analysis.py`

- [x] Add failing API tests proving a two-passage staged document returns one accepted job and one deferred passage without persisting a blocked job.
- [x] Add failing tests proving same-key replay returns the same accepted job, a completed current-revision job is returned by state and scheduling without dispatch, and global unavailability returns no accepted jobs.
- [x] Add immutable document schedule/state response models and a repository query for recoverable current-revision jobs in document order.
- [x] Implement locked document scheduling by reusing existing readiness, staged-limit, idempotency, and dispatcher behavior.
- [x] Run focused analysis tests (17 passed), update this checklist, commit, and push.

### Task 2: Protect active paid work from passage mutation

**Files:**
- Modify: `backend/src/redacted_backend/db/repositories/analysis.py`
- Modify: `backend/src/redacted_backend/db/repositories/passages.py`
- Modify: `backend/src/redacted_backend/api/passages.py`
- Test: `backend/tests/api/test_passages.py`

- [x] Add failing tests that patch, delete, and create are rejected while a document has queued/running analysis and allowed after terminal completion/failure/cancellation.
- [x] Add an analysis-repository active-work guard and map its passage conflict to a stable HTTP 409 response.
- [x] Preserve existing invalidation/cancellation behavior for non-active and internal retention paths.
- [x] Run focused passage and analysis tests (38 passed), update this checklist, commit, and push.

### Task 3: Stream accepted jobs and recover stored predictions in the browser

**Files:**
- Modify: `src/services/backendApi.ts`
- Modify: `src/services/backendApi.test.ts`
- Modify: `src/features/workspace/DocumentWorkspace.tsx`
- Modify: `src/features/workspace/DocumentWorkspace.test.tsx`

- [x] Add failing transport tests for document scheduling and state recovery.
- [x] Replace passage fan-out with one document scheduling request and prove accepted jobs stream even when passages are deferred.
- [x] Add failing workspace tests for completed-job recovery without scheduling, all-deferred messaging, duplicate-click suppression, singular **Predict passage** copy, and mutation/replacement locking while active.
- [x] Implement revision-bound recovery during context preparation and feed recovered/accepted job bases into the existing SSE service.
- [x] Keep customer copy generic and leave the fictional demo unchanged.
- [x] Run focused and full frontend tests (99 passed), typecheck, production build, update this checklist, commit, and push.

### Task 4: Full verification and no-spend deployment

**Files:**
- Modify: `CHECKLIST.md`
- Modify: this plan

- [ ] Run the full backend and frontend suites, typecheck, production build, migration round-trip, and Compose validation.
- [ ] Record provider-claim count and spend ledger before deployment.
- [ ] Merge the reviewed branch to `main`, restart only the local API/worker/frontend services required for the fix, and verify readiness.
- [ ] Recover the existing completed prediction through document state plus SSE and prove provider claims/spend are unchanged.
- [ ] Update the master checklist and this checklist with measured evidence, commit, and push.
