# Durable Local Dense Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the complete 14,549,269-row int8 dense benchmark and resumable production build outside the Codex process while staged BM25 predictions remain available.

**Architecture:** Wrap the existing compressed dense builder in a single-instance macOS runner that records an ignored state directory, benchmarks storage/checkpoint sizing, and launches/resumes only after storage gates pass. Publication remains controlled by the existing checksum, recall, latency, and readiness validators.

**Tech Stack:** zsh, macOS `nohup`/`caffeinate`, Python 3.12 corpus CLI, int8 memmaps, FAISS IVF-PQ, JSON state files.

**Spec:** `docs/superpowers/specs/2026-11-22-staged-bm25-prediction-and-durable-dense-build-design.md`

## Global Constraints

- Do not delete corpus data, BM25 data, Docker cache, user files, or completed checkpoints.
- Require projected artifacts `<= 16 GiB` and at least 4 GiB projected free storage.
- Use one live PID; a second start must report the live runner or resume verified checkpoints.
- Publish no final dense manifest until checksum, recall@300 `>= 0.98`, total-size, and 100-query same-worker latency `<= 5.0` seconds all pass.
- Do not enable full four-list spending when the dense build finishes.

## Review Focus

- Stale versus live PID files must be distinguished without killing unrelated processes; cover in Task 1.
- Interrupted production embedding checkpoints must resume without duplicate or skipped rows; cover through the existing builder tests and Task 1 integration test.
- Benchmark interruption must report restart-required while production checkpoint interruption remains resumable; cover in Task 1.
- Storage projections just above 16 GiB or below 4 GiB free must fail before production build; cover in Task 1.
- Dense completion must not change any paid-call flag or staged cap; cover in Task 2.

---

### Task 1: Add and verify the durable single-instance runner

**Files:**
- Create: `scripts/run-local-dense-build.zsh`
- Create: `backend/tests/corpus/test_local_dense_runner.py`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: existing compressed dense benchmark/build CLI, verified chunks and BM25 artifacts, and host disk/process state.
- Produces: ignored `var/corpus/best-available/dense-build-state/{state.json,runner.log,runner.pid}` and a durable benchmark-to-build lifecycle.

- [x] **Step 1: Write failing subprocess tests** for live/stale PID handling, no destructive commands, benchmark failure, storage rejection, resume arguments, and atomic state transitions.
- [x] **Step 2: Run the focused runner test red.**
- [x] **Step 3: Implement status/start/internal-run commands using explicit paths, `nohup`, and `caffeinate`, delegating all artifact writes to the existing builder.**
- [x] **Step 4: Run focused runner plus existing dense builder/storage suites green.**
- [x] **Step 5: Commit and push with `feat: add durable local dense runner`.**

### Task 2: Launch and monitor the real build

**Files:**
- Runtime artifacts only under ignored `var/corpus/best-available/dense-build-state/`.
- Modify: `CHECKLIST.md` after each verified phase.

**Interfaces:**
- Consumes: production source artifacts and the committed runner.
- Produces: one durable live process, benchmark report, production checkpoints, final certification reports, and eventually full four-list operational readiness with spending unchanged.

- [x] **Step 1: Run full no-spend regression verification and invoke the runner start command.**
- [x] **Step 2: Confirm the PID is live under `caffeinate`, the state advances, and no competing runner exists.**
- [x] **Step 3: Record benchmark size/rate/checkpoint decision and confirm the production build starts only if both storage gates pass.** The measured report projected 13,022,045,511 bytes; the conservative 68,572-row checkpoint shape projected 15,669,312,441 bytes with 58,277,482,496 bytes available, and the build started only afterward.
- [x] **Step 4: Update and push the checklist with measured progress while leaving runtime artifacts ignored and all full four-list approvals unchanged.** Recorded 54,080 contiguous rows (54.1%) at 10:35 PM EST while the single runner remained CPU-active and the live staged stack remained healthy; the final benchmark report and storage decision remain correctly pending Step 3.

### Task 3: Make post-build certification durable

**Files:**
- Create: `scripts/finalize-local-dense-build.zsh`
- Create: `backend/tests/corpus/test_local_dense_finalizer.py`
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `CHECKLIST.md`

**Interfaces:**
- Consumes: Task 2's durable build state and published recall-certified dense manifest.
- Produces: one durable single-instance waiter, worker-image latency report, final verification report, and explicit terminal state without changing provider approvals.

- [x] **Step 1: Write failing subprocess tests** for condition-based waiting, worker-image command order, read-write artifact override, terminal build failures, latency failure, stale PID recovery, and duplicate-start prevention.
- [x] **Step 2: Run the focused finalizer tests red.** Nine tests failed because the finalizer did not exist.
- [x] **Step 3: Implement a launchd/nohup/caffeinate finalizer with atomic state and no destructive operations or provider path.**
- [x] **Step 4: Run focused finalizer and runner tests, then the complete backend suite.** Focused: 18 passed. Full backend: 531 passed, 2 expected environment skips.
- [x] **Step 5: Launch the finalizer beside the active benchmark, update the checklist, commit, and push.** PID 22608 is launchd-managed under PID 1, reports `waiting` on build phase `benchmarking`, and had not invoked Docker at launch verification.

### Task 4: Apply the measured CPU throughput setting

**Files:**
- Modify: `backend/src/redacted_backend/retrieval/dense.py`
- Modify: `backend/src/redacted_backend/corpus/build_dense_sharded.py`
- Modify: `backend/src/redacted_backend/corpus/cli.py`
- Modify: `backend/tests/retrieval/test_dense.py`
- Modify: `backend/tests/corpus/test_cli.py`
- Modify: `CHECKLIST.md`

**Interfaces:**
- Consumes: the pinned CPU-only BGE encoder and the benchmark/build CLI used by the already-running durable runner.
- Produces: an explicit offline encoder thread setting that defaults to the measured fastest safe value while leaving query-time worker encoding unchanged.

- [x] **Step 1: Record controlled real-corpus batch and thread measurements without interrupting the live benchmark.** Batch 32 measured 5.57/5.76 rows/s versus batch 64 at 5.45/5.53; four threads measured 5.68/5.62 rows/s, six 5.97, eight 6.06, and ten 5.98 on the same real passages. Observed vectors were identical at eight threads.
- [x] **Step 2: Write failing tests** proving thread configuration precedes model loading and both dense CLI commands pass the offline default of eight threads. Three tests failed on the missing interface/default.
- [x] **Step 3: Implement the minimal builder-only thread configuration without changing model identity, prompts, vector normalization, or runtime query encoding.** Focused retrieval/CLI suite: 20 passed.
- [x] **Step 4: Run focused and full backend verification, update the checklist, commit, and push before the production child process starts.** Focused: 20 passed. Full backend: 532 passed, 2 expected environment skips.

### Task 5: Isolate FAISS and resume the completed benchmark

**Files:**
- Create: `backend/src/redacted_backend/corpus/faiss_build_worker.py`
- Create: `backend/tests/corpus/test_faiss_build_worker.py`
- Modify: `backend/src/redacted_backend/corpus/build_dense_sharded.py`
- Modify: `backend/src/redacted_backend/corpus/cli.py`
- Modify: `backend/tests/corpus/test_build_dense_sharded.py`
- Modify: `backend/tests/corpus/test_cli.py`
- Modify: `backend/tests/corpus/test_local_dense_runner.py`
- Modify: `scripts/run-local-dense-build.zsh`
- Modify: `README.md`
- Modify: `CHECKLIST.md`

**Interfaces:**
- Consumes: the checksum-verified 100,000-row int8 benchmark sample and the pinned dense artifact profile.
- Produces: a checksum-bound benchmark checkpoint, a fresh FAISS-only child process for training/add/search, and a one-shot managed runner that cannot loop after a native crash.

- [x] **Step 1: Write failing tests** for FAISS process isolation, complete-sample recovery, checkpoint resume without re-encoding, and managed-job cleanup. The first focused runs failed on the missing worker/recovery interfaces and missing managed-job removal.
- [x] **Step 2: Implement the FAISS-only worker and benchmark checkpoint/recovery path without weakening corpus or artifact identity checks.** The recovered 100,000-row sample is bound to corpus `c28eb9…`, embeddings `1c8017…`, offsets `641395…`, and the pinned dense semantics/training selection.
- [x] **Step 3: Prevent launchd from repeating a failed or completed run, then run focused and complete backend verification.** Focused: 59 passed. Full backend: 539 passed with 2 expected environment skips; zsh syntax and diff checks passed.
- [x] **Step 4: Recover the existing 100,000-row sample, restart exactly one runner, verify it enters FAISS without re-encoding, update the checklist, commit, and push.** The sample mtime remained `2026-11-23T00:13:53-0500`; isolated FAISS finished in 13.91 seconds, the measured storage gates passed, and PID 45140 advanced to the real 14,549,269-row build with 68,572-row resumable shards.

### Task 6: Add a safe reverse-order CPU encoding accelerator

**Files:**
- Modify: `backend/src/redacted_backend/corpus/build_dense_sharded.py`
- Modify: `backend/src/redacted_backend/corpus/cli.py`
- Modify: `backend/tests/corpus/test_build_dense_sharded.py`
- Modify: `backend/tests/corpus/test_cli.py`
- Create: `scripts/run-local-dense-accelerator.zsh`
- Create: `backend/tests/corpus/test_local_dense_accelerator.py`
- Modify: `README.md`
- Modify: `CHECKLIST.md`

**Interfaces:**
- Consumes: the verified production offset table, the exact same pinned BGE semantics/profile, and the primary builder's atomic embedding-checkpoint contract.
- Produces: a second CPU-only encoder that works from the final shard backward, publishes only checksum-compatible checkpoints, and stops with a safety gap before it can race the forward builder.

- [x] **Step 1: Record concurrent real-corpus throughput and write failing tests** for reverse checkpoint order, forward-builder reuse, the safety gap, CLI identity, and single-instance durable execution. One extra four-thread encoder measured 11.08 combined rows/s; two extra three-thread encoders measured 11.25, so exactly one accelerator is justified.
- [x] **Step 2: Implement the reverse accelerator using the existing atomic checkpoint manifest and corpus/offset verification boundaries.** It shares no forward iterator or temporary directory, works from the final shard backward, and stops four shards before the first incomplete forward checkpoint.
- [x] **Step 3: Run focused and complete verification, commit, and push.** Focused: 64 passed. Full backend: 544 passed with 2 expected environment skips; zsh syntax and diff checks passed.
- [x] **Step 4: Run the sustained two-encoder acceptance check and keep only the fastest configuration.** The processes were non-overlapping, but sustained memory compression reduced combined throughput to about 3.2 rows/s. Stop only the accelerator, preserve the primary process, and keep one encoder; the primary recovered to 8.53 rows/s.
