# Local int8 four-list artifacts implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and serve the real best-available FOIArchive four-list dense artifacts locally without a cloud GPU, then make the production prediction basis operational for a separately approved real two-stage model call.

**Architecture:** Store each normalized BGE document vector as a deterministic symmetric int8 direction in resumable one-million-row shards. Dequantize and renormalize rows to float32 only for FAISS IVF-PQ training/addition, recall certification, and candidate rescoring; preserve independent corpus-wide dense searches and every existing fail-closed provenance gate.

**Tech Stack:** Python 3.12, NumPy 2, Pydantic 2, FAISS CPU 1.16, sentence-transformers 5, PyTorch CPU, pytest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-21-compressed-four-list-dense-design.md`

## Global constraints

- Keep `BAAI/bge-base-en-v1.5` revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` and the frozen query instruction.
- Preserve four independent top-300 lists and the existing document RRF/top-100/two-excerpt behavior.
- Use `symmetric-int8-direction-v1`: per row, divide by maximum absolute value, multiply by 127, round, clip to `[-127, 127]`, and store signed bytes.
- Renormalize stored int8 directions in float32 before all similarities and FAISS operations.
- Require mean recall@300 `>= 0.98`, every local warm search `<= 5.0` seconds, and final artifacts `<= 16 GiB`.
- Keep all paid-model approvals false until operational/unapproved readiness passes and exact two-stage cost is approved.
- Do not delete Docker cache, corpus/BM25 artifacts, user files, or the fictional demo.

## Review focus

- A zero, nonfinite, or wrong-width vector must fail quantization instead of producing a corrupt row.
- A float16/v2 or unknown quantizer manifest must fail readiness; it must never be interpreted as int8.
- Resume must verify int8 size/checksum and never re-encode a complete compatible shard.
- Runtime rescoring, exact certification, and FAISS population must all use the same dequantize-and-renormalize function.
- The benchmark must project the int8 bytes actually written and fail before a full build that cannot fit current storage.

---

### Task 1: Version and test the int8 storage codec

**Files:**
- Create: `backend/src/redacted_backend/retrieval/int8_embeddings.py`
- Modify: `backend/src/redacted_backend/retrieval/dense_profile.py`
- Modify: `backend/src/redacted_backend/retrieval/types.py`
- Create: `backend/tests/retrieval/test_int8_embeddings.py`
- Modify: `backend/tests/retrieval/test_dense_profile.py`

**Interfaces:**
- Consumes: normalized `NDArray[np.float32]` document rows.
- Produces: `quantize_directions(values) -> NDArray[np.int8]`, `dequantize_directions(values) -> NDArray[np.float32]`, schema-v3 manifests, and the `symmetric-int8-direction-v1` artifact identity.

- [ ] **Step 1: Write failing codec and manifest tests**

Assert deterministic literal bytes, unit-length reconstructed rows, bounded cosine loss, rejection of zero/nonfinite/wrong-rank input, one byte per dimension, schema-v3/int8 filenames, and rejection of old or mismatched artifact identities.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `cd backend && uv run pytest -q tests/retrieval/test_int8_embeddings.py tests/retrieval/test_dense_profile.py`

- [ ] **Step 3: Implement the minimal codec and schemas**

Use `np.rint`, explicit clipping, contiguous arrays, and one shared dequantization function. Set the production candidate identity to `faiss-ivfpq-int8-rescore-v1`, manifest schema to `sharded-dense-index-manifest-v3`, dtype to `int8`, and maximum artifact bytes to `16 * 1024**3`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command and require zero failures.

- [ ] **Step 5: Commit**

Commit: `feat: version int8 dense artifacts`

### Task 2: Write resumable int8 shards and truthful projections

**Files:**
- Modify: `backend/src/redacted_backend/corpus/build_dense_sharded.py`
- Modify: `backend/tests/corpus/test_build_dense_sharded.py`
- Modify: `backend/tests/corpus/test_dense_storage_budget.py`
- Modify: `backend/tests/corpus/test_cli.py`

**Interfaces:**
- Consumes: Task 1 codec/schema and existing verified chunks/corpus manifest.
- Produces: `embeddings.i8` checkpoints/final hard links, dequantized FAISS inputs, and benchmark reports with `embedding_bytes_per_row == dimension`.

- [ ] **Step 1: Write failing builder, resume, projection, and corruption tests**

Cover one-byte row sizing, no float16 file, byte-identical resume, corrupt int8 rejection, candidate rebuild without re-encoding, hard-link publication, 16-GiB ceiling, and benchmark projection from physical int8 bytes.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `cd backend && uv run pytest -q tests/corpus/test_build_dense_sharded.py tests/corpus/test_dense_storage_budget.py tests/corpus/test_cli.py`

- [ ] **Step 3: Implement the minimal int8 builder changes**

Quantize each encoded batch before writing. Open all stored rows as `np.int8`, call Task 1 dequantization before FAISS training/addition and recall math, publish `embeddings.i8`, and retain atomic/checksum/resume behavior.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command and require zero failures.

- [ ] **Step 5: Commit**

Commit: `feat: build resumable int8 dense shards`

### Task 3: Serve and certify the int8 representation

**Files:**
- Modify: `backend/src/redacted_backend/retrieval/dense_sharded.py`
- Modify: `backend/src/redacted_backend/domain/readiness.py`
- Modify: `backend/src/redacted_backend/api/readiness.py`
- Modify: `backend/src/redacted_backend/jobs/runtime.py`
- Modify: `backend/tests/retrieval/test_dense_sharded.py`
- Modify: `backend/tests/retrieval/test_dense_recall_sharded.py`
- Modify: `backend/tests/test_readiness.py`
- Modify: `backend/tests/jobs/test_runtime_provider.py`
- Modify: `scripts/transfer-compressed-dense.zsh`
- Modify: `backend/tests/corpus/test_dense_transfer_script.py`

**Interfaces:**
- Consumes: schema-v3 int8 artifacts from Task 2.
- Produces: cached local four-list dense searches, recall/latency evidence, and fail-closed operational readiness.

- [ ] **Step 1: Write failing runtime/readiness/transfer tests**

Assert global candidate rescoring uses reconstructed unit vectors, stable ties remain unchanged, complete top-300 results are required, missing/corrupt int8 files fail, old v2 manifests fail, transfer validates schema v3 and `embeddings.i8`, and runtime performs both dense lists independently of BM25.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `cd backend && uv run pytest -q tests/retrieval/test_dense_sharded.py tests/retrieval/test_dense_recall_sharded.py tests/test_readiness.py tests/jobs/test_runtime_provider.py tests/corpus/test_dense_transfer_script.py`

- [ ] **Step 3: Implement the minimal runtime and gate changes**

Memory-map signed bytes, use Task 1 dequantization for every score path, verify exact size/checksum/name/schema/profile identity, and keep readiness closed until recall and same-host latency evidence pass.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command and require zero failures.

- [ ] **Step 5: Commit**

Commit: `feat: serve int8 four-list retrieval`

### Task 4: Verify regressions and start the real local build

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-11-21-compressed-four-list-dense-design.md`
- Modify: `docs/superpowers/plans/2026-11-22-local-int8-four-list-artifacts.md`

**Interfaces:**
- Consumes: Tasks 1-3, the existing 14,549,269-passage artifact, and live BM25 index.
- Produces: measured local benchmark, resumable real artifact build, operational/unapproved readiness, and exact two-stage model cost.

- [ ] **Step 1: Run all regression gates**

Run the full backend suite, frontend tests/typecheck/build, Compose config, migration round-trip, four-list loopback, and fictional-demo isolation with all paid flags false.

- [ ] **Step 2: Run a bounded real CPU benchmark**

Use the pinned production corpus and int8 profile. Record rows/second, peak RAM, projected bytes/time, and actual available bytes. Proceed only if projected final bytes are `<= 16 GiB` and leave at least 4 GiB free.

- [ ] **Step 3: Start the resumable real build**

Run the production build under a durable local process with logs and PID/status files under ignored `var/`. Never treat background process startup as completion; verify the first real shard checkpoint and report measured progress.

- [ ] **Step 4: Complete recall, latency, and readiness gates**

Require recall@300 `>= 0.98`, all 100 warm searches `<= 5.0s`, exact checksums/counts, live BM25 count `14,549,269`, and `operational=true`, `approved=false`, `ready=false`.

- [ ] **Step 5: Count and separately approve the real model requests**

Count one own-query call plus one evidence-bound prediction call without generation where supported. Show expected/maximum combined cost and wait for exact approval before enabling the two paid flags for one job.

- [ ] **Step 6: Commit and push measured documentation**

Commit: `docs: record local int8 retrieval build`
