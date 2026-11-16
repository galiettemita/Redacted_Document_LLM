# Corpus Source Audit and Unblock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the fixed storage permission gate and produce a streaming, reproducible audit of the pinned official History Lab gzip that precisely identifies what is still required for Andrew's complete corpus.

**Architecture:** Keep capacity reporting but make enforcement an explicit operator option. Add a gzip-streaming audit boundary that verifies the pinned Hugging Face snapshot, validates the official nine-field rows, accepts collection identity only from an explicit canonical label, compares computed counts to a frozen official catalog, and emits a checksummed fail-closed report without expanding the source file. The current source is expected to remain unready; its report drives an exact, unsent maintainer request rather than a scientific workaround.

**Tech Stack:** Python 3.12, gzip/streaming JSONL, Pydantic, Hugging Face snapshot metadata, pytest, existing corpus CLI.

**Spec:** `docs/superpowers/specs/2026-11-15-always-on-worker-and-corpus-unblock-design.md`

## Global Constraints

- Fixed free-space estimates are advisory by default; actual `ENOSPC` remains an ordinary resumable write failure.
- The official source artifact stays compressed and outside Git; the audit must not create a second uncompressed raw copy.
- Never infer collection identity from document IDs, URLs, retrieval results, or target-dependent evidence.
- Only the 12 canonical labels in `backend/corpus/exclusions/v1.yaml` are accepted; `cables` must be present and non-empty.
- The paper-derived pre-exclusion count remains 5,550,000 ±5,000 and exactly ten 2019–2024 FRUS volumes remain required by the later coverage gate.
- An unready source produces evidence and a maintainer request, never a ready manifest or index.
- No external email is sent by this plan.

## Review Focus

- A low-space report must exit successfully by default but still fail under explicit enforcement; Task 1 tests both modes.
- Gzip corruption or a snapshot checksum mismatch must stop before an audit can be trusted; Task 2 tests both.
- URL/null `source` values must not be mistaken for collection labels; Task 2 tests unknown-value counts and capped samples.
- A row-level `corpus` value must take precedence over `source`, and conflicting canonical labels must fail; Task 2 tests the conflict.
- An audit interrupted by malformed JSON must not publish the final report; Task 3 tests atomic report output through the CLI.

---

### Task 1: Advisory corpus capacity CLI

**Files:**
- Create: `backend/tests/corpus/test_cli.py`
- Modify: `backend/src/redacted_backend/corpus/cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: existing `run_preflight(paths) -> PreflightReport`.
- Produces: `corpus preflight --path ...` always reports and exits 0 by default; `--enforce-minimum` exits 2 when `report.sufficient` is false.

- [ ] **Step 1: Write failing CLI policy tests**

Patch only the external disk probe boundary by replacing `cli.run_preflight` with literal insufficient/recommended `PreflightReport` fixtures. Assert:

- insufficient default returns 0 and prints `"status": "insufficient"`;
- insufficient plus `--enforce-minimum` returns 2;
- recommended plus enforcement returns 0.

- [ ] **Step 2: Run the focused tests and verify RED**

Run from `backend/`: `uv run pytest tests/corpus/test_cli.py -q`

Expected: FAIL because `--enforce-minimum` is unknown and insufficient default currently returns 2.

- [ ] **Step 3: Implement explicit enforcement**

Add `preflight.add_argument("--enforce-minimum", action="store_true")` and return 2 only when that flag is set and the report is insufficient. Keep every report field and threshold unchanged.

- [ ] **Step 4: Run focused and full backend tests**

Run from `backend/`:

- `uv run pytest tests/corpus/test_cli.py tests/corpus/test_preflight.py -q`
- `uv run pytest -q`

Expected: focused and full suites PASS with only the existing optional Tesseract skip allowed.

- [ ] **Step 5: Update documentation, checklist, commit, and push**

Change README commands/copy so capacity is advisory unless `--enforce-minimum` is supplied. Mark the Task 14 preflight line complete after tests pass.

```bash
git add backend/src/redacted_backend/corpus/cli.py backend/tests/corpus/test_cli.py README.md CHECKLIST.md
git commit -m "feat: make corpus capacity preflight advisory"
git push origin feat/real-retrieval-backend
```

### Task 2: Streaming official-bundle audit

**Files:**
- Create: `backend/src/redacted_backend/corpus/bulk_export.py`
- Create: `backend/tests/corpus/test_bulk_export.py`
- Create: `backend/corpus/historylab-huggingface-2025-catalog.json`

**Interfaces:**
- Consumes: `SnapshotLock`, one locked local `.json.gz`/`.jsonl.gz`, a `BulkExportCatalog`, and `ExclusionManifest`.
- Produces: `BulkExportCatalog`, `BulkExportAudit`, and `audit_historylab_bulk_export(source_path, *, snapshot, catalog, exclusions) -> BulkExportAudit`.

- [ ] **Step 1: Write failing audit tests with tiny real gzip fixtures**

Tests must cover:

- a valid canonical `source` label is counted without creating an expanded raw file;
- explicit `corpus` takes precedence, while conflicting canonical `corpus`/`source` labels raise `BulkExportAuditError`;
- URL/null/unknown `source` values increment `unresolved_collection_count`, retain at most 20 sorted samples, and make `ready_for_import=false`;
- catalog/document count mismatch and missing `cables` produce literal reasons;
- malformed JSON, missing documented fields, gzip corruption, size mismatch, and SHA-256 mismatch raise `BulkExportAuditError`;
- report `manifest_hash()` is deterministic.

Expected values must be hand-authored. Use actual `gzip.open` fixtures and real snapshot hashing; do not mock decompression or filesystem behavior.

- [ ] **Step 2: Run the focused tests and verify RED**

Run from `backend/`: `uv run pytest tests/corpus/test_bulk_export.py -q`

Expected: FAIL during collection because `redacted_backend.corpus.bulk_export` does not exist.

- [ ] **Step 3: Implement the audit boundary**

Define frozen Pydantic models with explicit schema versions. Stream the gzip line by line, require the nine documented fields plus optional `corpus`, validate snapshot filename/size/SHA before decompression, cap unresolved samples at 20, and compare computed canonical counts to the frozen catalog. Return reasons sorted and deduplicated. Do not write any normalized document artifact.

The catalog file must pin repository `HistoryLab/foiarchive`, revision `411cf6f86b38342683e27148be167c6094d1a10c`, the 11 official dataset-card collection counts totaling 4,949,871, and the official release URL.

- [ ] **Step 4: Run focused and full backend tests**

Run from `backend/`:

- `uv run pytest tests/corpus/test_bulk_export.py tests/corpus/test_snapshot.py tests/corpus/test_corpus_coverage.py -q`
- `uv run pytest -q`

Expected: focused and full suites PASS with only the existing optional Tesseract skip allowed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/redacted_backend/corpus/bulk_export.py backend/tests/corpus/test_bulk_export.py backend/corpus/historylab-huggingface-2025-catalog.json
git commit -m "feat: audit official History Lab corpus bundle"
```

### Task 3: Audit CLI, official evidence, and exact external gate

**Files:**
- Modify: `backend/src/redacted_backend/corpus/cli.py`
- Modify: `backend/tests/corpus/test_cli.py`
- Create: `docs/corpus/historylab-complete-export-request.md`
- Modify: `README.md`
- Modify: `CHECKLIST.md`

**Interfaces:**
- Consumes: Task 2's `audit_historylab_bulk_export` and models.
- Produces: `corpus audit-bulk-export --source --snapshot-lock --catalog --exclusions --output`; an atomic JSON audit report; a prepared but unsent maintainer request.

- [ ] **Step 1: Write failing CLI audit tests**

Use a tiny locked gzip fixture and assert that the CLI:

- writes a valid atomic audit report and returns 0 when `ready_for_import=true`;
- writes an unready report and returns 2 for missing `cables`/unresolved labels;
- leaves no final output file when parsing fails.

- [ ] **Step 2: Run the focused tests and verify RED**

Run from `backend/`: `uv run pytest tests/corpus/test_cli.py -q`

Expected: FAIL because `audit-bulk-export` is not a recognized command.

- [ ] **Step 3: Implement the CLI and atomic report write**

Load the snapshot lock, catalog, and exclusions with their Pydantic models; call Task 2's audit; write with the existing atomic JSON helper pattern; print the same report; return 0 only when `ready_for_import` is true.

- [ ] **Step 4: Run focused and full backend tests**

Run from `backend/`:

- `uv run pytest tests/corpus/test_cli.py tests/corpus/test_bulk_export.py -q`
- `uv run pytest -q`

Expected: focused and full suites PASS with only the existing optional Tesseract skip allowed.

- [ ] **Step 5: Acquire and audit the pinned official artifact**

Use the already authenticated Hugging Face account and the immutable revision to download/cache `foiarchive-202409.json.gz` outside Git. Generate a snapshot lock with `corpus lock-snapshot`, run `corpus audit-bulk-export`, and retain only the small lock/report paths documented in README. Do not copy the 7,976,348,471-byte artifact into the repository.

Expected: the command completes without expanding the gzip. Based on prior inspection, the report is expected to return 2 with exact evidence for unusable collection labels and absent `cables`; treat any different result as evidence to inspect, not permission to weaken the gate.

- [ ] **Step 6: Prepare the exact unsent maintainer request**

Document the required paper-compatible export properties: 12 canonical collection labels, explicit `corpus` per record, complete text/metadata, immutable revision/checksum, exact per-collection counts, total 5.55M±5,000, non-empty `cables`, and FRUS volume identity. Include the audit report facts. Clearly label the request `NOT SENT`.

- [ ] **Step 7: Update documentation/checklist, verify, commit, and push**

Run from repository root:

- `git diff --check`
- `cd backend && uv run pytest -q`
- `cd .. && npm test -- --run`

Expected: diff check exits 0; backend and frontend suites PASS with only the existing optional Tesseract skip allowed.

Update README with the streaming audit commands and current evidence. Mark only audit/tooling checklist items complete; leave complete-source acquisition and corpus readiness unchecked.

```bash
git add backend/src/redacted_backend/corpus/cli.py backend/tests/corpus/test_cli.py docs/corpus/historylab-complete-export-request.md README.md CHECKLIST.md
git commit -m "docs: record complete corpus source gate"
git push origin feat/real-retrieval-backend
```
