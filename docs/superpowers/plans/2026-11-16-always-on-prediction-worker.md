# Always-On Prediction Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the real Dramatiq worker as a normal service while keeping retrieval and paid-model execution independently fail-closed until every production dependency is operational.

**Architecture:** Replace eager orchestrator construction with a thread-safe provider that caches only successful real-runtime initialization. The task entrypoint boots with Redis and PostgreSQL, delegates error semantics to a small execution boundary, and records failures through a database handle that exists even when runtime construction fails. API corpus readiness additionally checks dense artifact presence and the live OpenSearch index count before dispatch.

**Tech Stack:** Python 3.12, Dramatiq, Redis, SQLAlchemy/PostgreSQL, FastAPI, OpenSearch HTTP API, pytest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-15-always-on-worker-and-corpus-unblock-design.md`

## Global Constraints

- Production code must never substitute a mock runtime, simulated retrieval, or fictional prediction in the uploaded-document path.
- Worker liveness means only a current expiring Redis heartbeat; it does not claim corpus or model readiness.
- No model call may occur unless the API key is configured and `REDACTED_ALLOW_PAID_MODEL_CALLS=true`; this plan never enables either.
- Runtime initialization failures are sanitized as `analysis_runtime_unavailable`; typed retrieval/model codes apply only after initialization succeeds.
- A failed runtime build is not cached; one successful runtime is cached per worker process.
- The fictional demo remains isolated and continuously available.
- No production deployment or branch merge is part of this plan.

## Review Focus

- Concurrent first jobs must not construct duplicate embedding/index/model runtimes; Task 1 tests one builder call under contention.
- A transient first initialization failure must not poison the worker process; Task 1 tests a later successful retry.
- A secret-bearing initialization exception must not be persisted; Task 2 tests that only its exception class is recorded.
- Manifest-ready files with a missing/wrong live OpenSearch index must not dispatch; Task 3 tests probe failure and count mismatch.
- The default stack must start the worker without starting or depending on OpenSearch; Task 4 verifies rendered Compose services and dependencies.

---

### Task 1: Thread-safe lazy real-runtime provider

**Files:**
- Create: `backend/src/redacted_backend/jobs/runtime_provider.py`
- Create: `backend/tests/jobs/test_runtime_provider.py`

**Interfaces:**
- Consumes: a zero-argument `Callable[[], AnalysisOrchestrator]`.
- Produces: `RuntimeInitializationError(cause_name: str)` and `AnalysisRuntimeProvider(builder).get() -> AnalysisOrchestrator`.

- [ ] **Step 1: Write failing provider behavior tests**

Add tests named:

- `test_provider_construction_does_not_build_the_runtime`
- `test_provider_caches_one_successful_runtime`
- `test_provider_retries_after_a_failed_initialization`
- `test_provider_serializes_concurrent_first_initialization`

Assert literal builder-call counts, object identity for the cached runtime, a `RuntimeInitializationError` whose public `cause_name` is only `RuntimeError`, and one call under a `ThreadPoolExecutor` contention fixture.

- [ ] **Step 2: Run the focused tests and verify RED**

Run from `backend/`: `uv run pytest tests/jobs/test_runtime_provider.py -q`

Expected: FAIL during collection because `redacted_backend.jobs.runtime_provider` does not exist.

- [ ] **Step 3: Implement the provider**

Implement:

```python
class RuntimeInitializationError(RuntimeError):
    cause_name: str

class AnalysisRuntimeProvider:
    def __init__(self, builder: Callable[[], AnalysisOrchestrator]) -> None: ...
    def get(self) -> AnalysisOrchestrator: ...
```

Use one `threading.Lock`, double-check the cached value inside the lock, cache only after a successful builder return, and raise `RuntimeInitializationError(type(error).__name__) from error` without exposing the original message.

- [ ] **Step 4: Run focused and full backend tests**

Run from `backend/`:

- `uv run pytest tests/jobs/test_runtime_provider.py -q`
- `uv run pytest -q`

Expected: focused tests PASS; full suite PASS with only the existing optional Tesseract skip allowed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/redacted_backend/jobs/runtime_provider.py backend/tests/jobs/test_runtime_provider.py
git commit -m "feat: lazily initialize prediction runtime"
```

### Task 2: Worker execution and durable initialization failure

**Files:**
- Create: `backend/src/redacted_backend/jobs/execution.py`
- Create: `backend/tests/jobs/test_execution.py`
- Create: `backend/tests/jobs/test_worker_import.py`
- Modify: `backend/src/redacted_backend/jobs/runtime.py`
- Modify: `backend/src/redacted_backend/jobs/tasks.py`

**Interfaces:**
- Consumes: `AnalysisRuntimeProvider.get()` and the existing `AnalysisJobRepository.fail()` contract.
- Produces: `execute_analysis_job(job_id: str, provider: AnalysisRuntimeProvider, *, last_attempt: bool, fail: FailureCallback) -> None`; `build_analysis_orchestrator(settings, *, database: Database | None = None) -> AnalysisOrchestrator`.

- [ ] **Step 1: Write failing execution-boundary tests**

Add tests proving:

- a non-final `RuntimeInitializationError` is re-raised and no failure is recorded;
- a final initialization failure records `analysis_runtime_unavailable` with detail `RuntimeError`, not the original secret-bearing message;
- a typed `RetrievalError` after successful initialization retains its existing code;
- a transient `ModelGatewayError("provider_unavailable", ...)` retries before the last attempt and records its typed code on the last attempt.

The fake provider/runtime may raise controlled exceptions, but assertions are on `execute_analysis_job`'s returned/raised behavior and recorded failure tuple, not on mock call presence.

- [ ] **Step 2: Write the failing import-safety subprocess test**

`test_importing_worker_tasks_does_not_require_corpus_or_model_configuration` must launch `sys.executable -c "import redacted_backend.jobs.tasks"` with only loopback Redis and SQLite `REDACTED_*` URLs configured and corpus/model variables removed.

Expected production break caught: the current eager builder raises `Analysis worker model access is not approved` during import.

- [ ] **Step 3: Run the focused tests and verify RED**

Run from `backend/`: `uv run pytest tests/jobs/test_execution.py tests/jobs/test_worker_import.py -q`

Expected: FAIL because `execution.py` does not exist and the task import still constructs the runtime eagerly.

- [ ] **Step 4: Implement execution semantics and lazy task boot**

Move the existing exception/retry mapping into `execute_analysis_job`. In `tasks.py`, require database and Redis configuration, create one `Database` handle, create an `AnalysisRuntimeProvider` around the real builder, and have the actor call `execute_analysis_job` with `_last_attempt()` and a database-backed failure callback.

Update the runtime builder signature to accept the already-created database handle and use it for the generation store/orchestrator instead of creating a second engine.

- [ ] **Step 5: Run focused and full backend tests**

Run from `backend/`:

- `uv run pytest tests/jobs/test_execution.py tests/jobs/test_worker_import.py tests/jobs/test_orchestrator.py -q`
- `uv run pytest -q`

Expected: focused tests PASS; full suite PASS with only the existing optional Tesseract skip allowed.

- [ ] **Step 6: Commit**

```bash
git add backend/src/redacted_backend/jobs backend/tests/jobs
git commit -m "feat: boot real worker before prediction dependencies"
```

### Task 3: Operational corpus and retrieval readiness

**Files:**
- Modify: `backend/src/redacted_backend/api/readiness.py`
- Modify: `backend/src/redacted_backend/retrieval/bm25.py`
- Modify: `backend/tests/test_readiness.py`

**Interfaces:**
- Consumes: compatible `CorpusManifest`, `Bm25IndexManifest`, `DenseIndexManifest`, configured dense directory, and OpenSearch URL.
- Produces: `OpenSearchHttpTransport.count(index_name: str) -> int`; `resolve_corpus_readiness(settings, *, index_counter: Callable[[str, str], int] | None = None) -> ComponentReadiness`.

- [ ] **Step 1: Write failing operational-readiness tests**

Extend readiness tests with real temporary dense artifact filenames (`embeddings.f16`, `metadata.jsonl`, `candidate.faiss`) and a fake `index_counter`. Prove these user-visible outcomes:

- missing dense directory/files returns `dense artifacts not available`;
- absent OpenSearch configuration returns `retrieval service not configured`;
- counter transport failure returns `retrieval index probe failed`;
- a live count different from the manifest returns `BM25 live index count mismatch`;
- matching files/manifests/live count returns `corpus and retrieval indexes operational`.

- [ ] **Step 2: Run readiness tests and verify RED**

Run from `backend/`: `uv run pytest tests/test_readiness.py -q`

Expected: FAIL because current readiness treats matching manifests alone as ready and does not accept `index_counter`.

- [ ] **Step 3: Implement the count probe and fail-closed readiness**

Add `OpenSearchHttpTransport.count()` using `GET /{quoted-index}/_count` and an integer `count` response. In `resolve_corpus_readiness`, keep manifest validation first, then check the dense directory and three required files, then configuration and live count. Catch network/HTTP/schema errors and return only the sanitized detail from Step 1.

- [ ] **Step 4: Run API, retrieval, and full backend tests**

Run from `backend/`:

- `uv run pytest tests/test_readiness.py tests/api/test_analysis.py tests/retrieval/test_bm25.py -q`
- `uv run pytest -q`

Expected: focused tests PASS; full suite PASS with only the existing optional Tesseract skip allowed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/redacted_backend/api/readiness.py backend/src/redacted_backend/retrieval/bm25.py backend/tests/test_readiness.py
git commit -m "feat: gate jobs on operational retrieval indexes"
```

### Task 4: Default Compose worker and live no-paid-call proof

**Files:**
- Modify: `compose.yaml`
- Modify: `README.md`
- Modify: `CHECKLIST.md`

**Interfaces:**
- Consumes: the lazy worker from Tasks 1–2 and operational readiness from Task 3.
- Produces: normal Compose startup includes `worker`; `opensearch` remains opt-in under profile `retrieval`.

- [ ] **Step 1: Record the current rendered Compose failure**

Run: `docker compose config --services`

Expected before change: output contains `postgres`, `redis`, and `backend` but omits `worker` because it is profile-gated.

- [ ] **Step 2: Update Compose**

Remove `profiles: ["retrieval"]` from `worker`. Remove only the `opensearch` dependency from `worker.depends_on`; retain healthy PostgreSQL and Redis dependencies. Keep OpenSearch under the retrieval profile and keep all real worker environment/artifact mounts.

- [ ] **Step 3: Verify rendered Compose behavior**

Run:

- `docker compose config --services`
- `docker compose --profile retrieval config --services`
- `docker compose config --format json`

Expected: default services include `worker` but not `opensearch`; retrieval-profile services include both; rendered worker dependencies name only `postgres` and `redis`.

- [ ] **Step 4: Rebuild/start the worker and prove independent readiness**

Run:

- `docker compose up -d --build postgres redis backend worker`
- `curl --fail --silent http://127.0.0.1:8000/v1/readiness`
- `docker compose logs --no-color --tail=100 worker backend`

Expected JSON: `worker.ready=true`, `corpus.ready=false`, `model.ready=false`, `prediction_ready=false`. Expected logs: worker boot/heartbeat, no model request, and no prediction-runtime fallback.

- [ ] **Step 5: Run frontend regression verification**

Run from repository root:

- `npm test -- --run`
- `npm run typecheck`
- `npm run build`

Expected: all frontend tests PASS; typecheck and build exit 0.

- [ ] **Step 6: Update operations documentation and checklist**

Document default worker startup, retrieval-profile startup, independent readiness meanings, and the fact that no paid call was made. Mark the Task 14 worker implementation line complete only after Steps 3–5 have fresh evidence.

- [ ] **Step 7: Commit and push**

```bash
git add compose.yaml README.md CHECKLIST.md
git commit -m "feat: keep real prediction worker online"
git push origin feat/real-retrieval-backend
```
