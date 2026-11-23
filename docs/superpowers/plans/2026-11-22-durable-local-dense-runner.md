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

- [ ] **Step 1: Run full no-spend regression verification and invoke the runner start command.**
- [ ] **Step 2: Confirm the PID is live under `caffeinate`, the state advances, and no competing runner exists.**
- [ ] **Step 3: Record benchmark size/rate/checkpoint decision and confirm the production build starts only if both storage gates pass.**
- [ ] **Step 4: Update and push the checklist with measured progress while leaving runtime artifacts ignored and all full four-list approvals unchanged.**
