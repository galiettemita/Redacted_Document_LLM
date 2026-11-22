# Compressed Four-List Dense Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `best_available_foiarchive_v1` operational with two independent corpus-wide dense lists, two BM25 lists, recall-certified compressed local artifacts, and warm dense-search latency no greater than five seconds per query.

**Architecture:** Preserve the existing corpus, BM25 index, BGE model, float16 exact rescoring, and four-list fusion. Replace HNSW-flat candidate storage with a versioned FAISS IVF-PQ profile, replace per-shard metadata copies with one checksum-bound global chunk-offset table, build embeddings once on a temporary cloud GPU, and serve all final artifacts locally with candidate indexes cached in memory and bounded parallel shard search.

**Tech Stack:** Python 3.12, Pydantic 2, NumPy 2, FAISS CPU 1.16, sentence-transformers 5, PyTorch 2, pytest, FastAPI, Dramatiq, Docker Compose, RunPod only after explicit spending approval.

**Spec:** `docs/superpowers/specs/2026-11-21-compressed-four-list-dense-design.md`

## Global Constraints

- Preserve the immutable corpus result: 4,300,534 documents and 14,549,269 passages from 4,889,089 parsed rows.
- Preserve the existing BM25 live count of exactly 14,549,269 and its published manifest.
- Preserve `BAAI/bge-base-en-v1.5` revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, normalized 768-dimensional vectors, and the existing query instruction.
- Preserve 300 results per raw list, document-level RRF, top 100 documents, and no more than two excerpts per document.
- Dense retrieval must be corpus-wide and independent of BM25 candidates for both visible context and the model-produced own query.
- Final rankings use exact inner products over the union of approximate candidates stored as float16 embeddings.
- Publication requires the frozen 100-query certification artifact and mean recall@300 `>= 0.98`.
- Warm dense search must take `<= 5.0` seconds for every certification query on the target Mac.
- Final compressed dense artifacts must be `<= 28 * 1024**3` bytes.
- Keep the fictional demo isolated and always available; do not expose internal queries/evidence/limitations in ordinary prediction cards.
- Keep `paper_exact_v1` distinct and fail closed; do not change a scheduled job's basis or silently fall back.
- Keep all paid-model flags false throughout implementation, artifact construction, readiness verification, and cost counting.
- No cloud GPU action starts without an exact benchmark cap; no full cloud build starts without a separate measured maximum-cost approval.
- Never commit source documents, chunk data, credentials, embeddings, indexes, or runtime reports.

## Review Focus

- A truncated, reordered, or checksum-mismatched `chunks.jsonl`/offset pair must fail before returning metadata; Task 2 pins malformed lines, wrong counts, stale checksums, out-of-range rows, and chunk-ID mismatches.
- FAISS may return duplicate, `-1`, or out-of-range row IDs when a shard is small or underfilled; Tasks 3 and 5 pin filtering/rejection and deterministic score/ID tie breaking.
- A crash may occur after embedding publication but before candidate publication, or after a candidate write but before its manifest; Task 4 proves resume without re-encoding verified embeddings and refusal of incomplete/corrupt shards.
- Fifteen shards can cause repeated index loads, unbounded threads, or excessive latency/memory; Task 5 proves one-time candidate loading, bounded parallel search, immutable mmap reuse, and the five-second report gate.
- A profile, recall report, latency report, offset file, shard, live BM25 count, or query hash can disagree while individual files still parse; Task 6 proves every mismatch keeps readiness and scheduling closed.

---

### Task 1: Version the compressed artifact profile and manifests

**Files:**
- Create: `backend/src/redacted_backend/retrieval/dense_profile.py`
- Modify: `backend/src/redacted_backend/retrieval/types.py`
- Create: `backend/tests/retrieval/test_dense_profile.py`
- Modify: `backend/tests/retrieval/test_dense_sharded.py`
- Modify: `backend/tests/retrieval/test_dense_recall_sharded.py`
- Modify: `backend/tests/corpus/test_build_dense_sharded.py`
- Modify: `backend/tests/corpus/test_cli.py`
- Modify: `backend/tests/test_readiness.py`

**Interfaces:**
- Consumes: `ScientificConfig` and the existing corpus/BM25 manifest hashes.
- Produces: `DenseArtifactProfile`, `dense_semantics_hash(scientific: ScientificConfig) -> str`, `CompressedDenseShardManifest`, `CompressedShardedDenseIndexManifest`, and `DenseLatencyReport`.

- [x] **Step 1: Write failing profile and schema-v2 tests**

Add tests asserting:

```python
profile = DenseArtifactProfile()
assert profile.candidate_index == "faiss-ivfpq-exact-rescore-v1"
assert profile.ivf_nlist == 4096
assert profile.pq_subvectors == 96
assert profile.pq_bits == 8
assert profile.nprobe == 256
assert profile.candidate_count_per_shard == 4_000
assert profile.training_sample_count == 200_000
assert profile.max_parallel_shards == 4
assert profile.max_artifact_bytes == 28 * 1024**3
assert profile.max_warm_search_seconds == 5.0
assert profile.manifest_hash() == DenseArtifactProfile().manifest_hash()
```

Assert schema v2 rejects unsafe shard paths, nonconsecutive ranges, wrong artifact-profile hashes, unsupported candidate names, offset sizes other than `chunk_count * 8`, PQ subvector counts that do not divide 768, and latency reports whose dense-manifest/query hashes differ.

- [x] **Step 2: Run the focused tests and confirm they fail**

Run: `cd backend && uv run pytest -q tests/retrieval/test_dense_profile.py tests/retrieval/test_dense_sharded.py tests/retrieval/test_dense_recall_sharded.py tests/corpus/test_build_dense_sharded.py tests/corpus/test_cli.py tests/test_readiness.py`

Expected: FAIL because the new profile and schema-v2 models do not exist.

- [x] **Step 3: Implement the immutable profile and manifest models**

In `dense_profile.py`, implement:

```python
class DenseArtifactProfile(BaseModel): ...
def dense_semantics_hash(scientific: ScientificConfig) -> str: ...
```

The semantics hash includes only model name/revision, query instruction, normalization requirement, dimension, inner-product metric, top-300 limit, and `0.98` recall threshold. The artifact-profile hash includes FAISS `1.16.0`, IVF/PQ/training/search values, `shared-chunk-offsets-v1`, maximum artifact bytes, maximum parallel shards, and maximum warm latency.

In `types.py`, add schema-v2 models with:

```python
class CompressedDenseShardManifest(BaseModel): ...
class CompressedShardedDenseIndexManifest(BaseModel): ...
class DenseLatencyReport(BaseModel): ...
```

Keep exact-paper `DenseIndexManifest` unchanged. Replace best-available test fixtures with schema v2; no production schema-v1 best-available artifact exists to migrate.

- [x] **Step 4: Run focused tests and confirm they pass**

Run the Step 2 command.

Expected: PASS.

- [x] **Step 5: Commit and update the design checklist**

```bash
git add backend/src/redacted_backend/retrieval/dense_profile.py backend/src/redacted_backend/retrieval/types.py backend/tests docs/superpowers/specs/2026-11-21-compressed-four-list-dense-design.md
git commit -m "feat: version compressed dense artifacts"
git push origin HEAD
```

### Task 2: Build and verify one shared chunk-offset index

**Files:**
- Create: `backend/src/redacted_backend/retrieval/chunk_offsets.py`
- Create: `backend/tests/retrieval/test_chunk_offsets.py`

**Interfaces:**
- Consumes: immutable `chunks.jsonl`, expected chunk count, and expected chunk SHA-256.
- Produces: `build_chunk_offsets(chunks_path: Path, offsets_path: Path, *, expected_count: int, expected_chunks_sha256: str) -> ChunkOffsetsReport` and `ChunkOffsetIndex.open(...) -> ChunkOffsetIndex`, with `read(row: int, *, expected_chunk_id: str | None = None) -> CorpusChunk`.

- [x] **Step 1: Write failing offset build/read tests**

Create fixtures with three `CorpusChunk` lines and assert:

```python
report = build_chunk_offsets(chunks, offsets, expected_count=3, expected_chunks_sha256=sha)
assert report.chunk_count == 3
assert report.offsets_size_bytes == 24
index = ChunkOffsetIndex.open(chunks, offsets, expected_count=3, expected_chunks_sha256=sha, expected_offsets_sha256=report.offsets_sha256)
assert index.read(2).id == chunks_fixture[2].id
```

Add the Review Focus cases: wrong chunk hash, missing final newline, blank line, malformed JSON, wrong count, truncated offset bytes, negative/out-of-range row, and mismatched `expected_chunk_id` all raise a fail-closed error without returning a chunk.

- [x] **Step 2: Run and confirm failure**

Run: `cd backend && uv run pytest -q tests/retrieval/test_chunk_offsets.py`

Expected: FAIL because the module is absent.

- [x] **Step 3: Implement atomic offsets and thread-safe random reads**

Write offsets as little-endian unsigned 64-bit values to a temporary sibling, fsync, verify the exact count, then rename atomically. Use `os.pread` or an equivalent per-call immutable read path so concurrent runtime reads never share a mutable seek position. Hash the chunks while building rather than trusting a filename.

- [x] **Step 4: Run focused and corpus-manifest tests**

Run: `cd backend && uv run pytest -q tests/retrieval/test_chunk_offsets.py tests/corpus/test_build_dense_sharded.py`

Expected: PASS.

- [x] **Step 5: Commit and push**

```bash
git add backend/src/redacted_backend/retrieval/chunk_offsets.py backend/tests/retrieval/test_chunk_offsets.py
git commit -m "feat: add shared dense chunk offsets"
git push origin HEAD
```

### Task 3: Add the real FAISS IVF-PQ candidate adapter

**Files:**
- Create: `backend/src/redacted_backend/retrieval/faiss_ivfpq.py`
- Create: `backend/tests/retrieval/test_faiss_ivfpq.py`
- Modify: `backend/uv.lock` only if `uv lock --check` proves the existing FAISS lock is not `1.16.0` on supported platforms.

**Interfaces:**
- Consumes: normalized float32 training/add vectors and `DenseArtifactProfile`.
- Produces: `train_ivfpq_template(training_vectors: NDArray[np.float32], output_path: Path, *, dimension: int, profile: DenseArtifactProfile) -> str` and `FaissIvfPqIndex.create(template_path: Path, *, profile: DenseArtifactProfile)`, `.load(path, *, profile)`, `.add(vectors)`, `.search(query, candidates)`, and `.save(path)`.

- [x] **Step 1: Write failing real-FAISS tests**

Use a small test profile (`dimension=8`, `nlist=2`, `pq_subvectors=2`, adequate deterministic training rows) and assert a trained template can create two independent indexes, saved/loaded rankings are stable, `nprobe` is applied, row IDs are local insertion order, and candidate files are smaller than a float32 matrix for a sufficiently large fixture.

Add tests that reject wrong dimensions, nonfinite/unnormalized vectors, untrained templates, wrong FAISS version/profile hash, over-add, and invalid requested candidate counts. Assert `search` removes FAISS `-1` sentinels and duplicates while preserving deterministic order.

- [x] **Step 2: Run and confirm failure**

Run: `cd backend && uv run pytest -q tests/retrieval/test_faiss_ivfpq.py`

Expected: FAIL because the adapter is absent.

- [x] **Step 3: Implement the adapter using `faiss.IndexIVFPQ`**

Use `faiss.IndexFlatIP` as quantizer, `METRIC_INNER_PRODUCT`, frozen profile parameters, and `faiss.clone_index`/`reset` to create empty shard indexes from one trained template. Persist only through FAISS serialization, fsync after writes, and expose no mutable raw FAISS object outside the adapter.

- [x] **Step 4: Run adapter and legacy exact-paper dense tests**

Run: `cd backend && uv run pytest -q tests/retrieval/test_faiss_ivfpq.py tests/corpus/test_build_dense.py tests/retrieval/test_dense.py`

Expected: PASS, proving the exact-paper HNSW path remains unchanged.

- [x] **Step 5: Commit and push**

```bash
git add backend/src/redacted_backend/retrieval/faiss_ivfpq.py backend/tests/retrieval/test_faiss_ivfpq.py backend/uv.lock
git commit -m "feat: add compressed dense candidates"
git push origin HEAD
```

### Task 4: Make the compressed dense build resumable by phase

**Files:**
- Modify: `backend/src/redacted_backend/corpus/build_dense_sharded.py`
- Modify: `backend/tests/corpus/test_build_dense_sharded.py`
- Create: `backend/tests/corpus/test_dense_storage_budget.py`

**Interfaces:**
- Consumes: Task 1 profile/models, Task 2 offsets, Task 3 template/index adapter, existing corpus manifest/chunk JSONL/certification queries/BGE encoder.
- Produces: the existing public `build_sharded_dense_from_files(...) -> ShardedDenseBuildReport`, now returning `CompressedShardedDenseIndexManifest`; verified embedding checkpoints; a trained-template artifact; complete candidate shards; and a `DenseBuildBenchmarkReport` for bounded samples.

- [x] **Step 1: Rewrite builder tests to fail on the compressed contract**

Assert a five-row fixture produces one global offsets file, embedding shards, one trained template, and candidate files—but no `metadata.jsonl` or per-shard offsets. Assert the manifest records profile/semantics hashes, exact ranges, sizes/checksums, and total artifact bytes below its ceiling.

Add phase-resume assertions:

```python
assert first_encoder.encoded_rows == 2
assert second_encoder.encoded_rows == 3  # verified first embedding shard reused
assert rebuilt_candidate_encoder.encoded_rows == 0  # candidate corruption never re-encodes
```

Add actual-write failure, insufficient-free-space, incomplete temporary directory, stale chunk checksum, wrong training-row file, corrupt embedding checkpoint, corrupt candidate, and final-size-over-28-GiB tests.

- [x] **Step 2: Run and confirm failure**

Run: `cd backend && uv run pytest -q tests/corpus/test_build_dense_sharded.py tests/corpus/test_dense_storage_budget.py`

Expected: FAIL against metadata-copy/HNSW behavior.

- [x] **Step 3: Implement explicit build phases**

Keep `build_sharded_dense_from_files` as the CLI boundary and split internal phases into focused helpers:

```python
_prepare_offsets_and_training_rows(...) -> ChunkOffsetsReport
_build_embedding_shards(...) -> list[EmbeddingShardCheckpoint]
_train_candidate_template(...) -> TrainedTemplateManifest
_build_candidate_shards(...) -> list[CompressedDenseShardManifest]
_publish_certified_manifest(...) -> CompressedShardedDenseIndexManifest
```

Select 200,000 training rows by stable hash over `(chunk.id, global_row)`, store sorted little-endian row IDs, train from completed float16 shard memmaps as float32, and build every candidate without invoking the encoder. Check real free bytes before each allocation and surface the exact path/required/available byte counts.

- [x] **Step 4: Implement the bounded 100,000-row benchmark report**

Add `benchmark_sharded_dense_build(...) -> DenseBuildBenchmarkReport` that uses the same stable selection rule, pinned encoder, and real file writes; record sample rows, seconds, rows/second, peak process RSS when available, bytes/row by artifact type, projected full bytes, and projected GPU hours. Because the approved benchmark has 100,000 rows and the production training sample has 200,000, use all 100,000 benchmark rows for a non-publishable benchmark-only IVF-PQ training pass; never treat that template or its recall as production evidence. It must not extrapolate dollar cost because price comes from the selected pod at approval time.

- [x] **Step 5: Run builder, offset, candidate, and resume tests**

Run: `cd backend && uv run pytest -q tests/corpus/test_build_dense_sharded.py tests/corpus/test_dense_storage_budget.py tests/retrieval/test_chunk_offsets.py tests/retrieval/test_faiss_ivfpq.py`

Expected: PASS.

- [x] **Step 6: Commit and push**

```bash
git add backend/src/redacted_backend/corpus/build_dense_sharded.py backend/tests/corpus/test_build_dense_sharded.py backend/tests/corpus/test_dense_storage_budget.py
git commit -m "feat: build resumable compressed dense shards"
git push origin HEAD
```

### Task 5: Serve fast independent dense searches locally

**Files:**
- Modify: `backend/src/redacted_backend/retrieval/dense_sharded.py`
- Modify: `backend/tests/retrieval/test_dense_sharded.py`
- Create: `backend/tests/retrieval/test_dense_latency.py`

**Interfaces:**
- Consumes: schema-v2 manifest, Task 2 `ChunkOffsetIndex`, Task 3 candidate loader, float16 shard embeddings, `QueryEncoder`.
- Produces: `ShardedDenseIndex(root: Path, chunks_path: Path, manifest: CompressedShardedDenseIndexManifest, encoder: QueryEncoder, candidate_loader=FaissIvfPqIndex.load, profile: DenseArtifactProfile | None = None)`, cached candidate indexes/memmaps, deterministic `search`, and `certify_dense_latency(...) -> DenseLatencyReport`.

- [x] **Step 1: Write failing runtime and latency tests**

Update the two-shard search fixture to one global chunk file/offset table. Assert both shards are searched even when BM25 is not present, candidate loaders run once at construction and never again across searches, exact global rescoring determines rank, duplicate/`-1` candidates are ignored, out-of-range rows fail, equal scores sort by chunk ID, and query metadata/provenance remain unchanged.

Use injected barriers/counters to assert at most four shard searches run concurrently. Use an injected monotonic clock to assert a 4.99-second maximum publishes a report and a 5.01-second result raises without writing one. Assert the report covers exactly the frozen 100 queries and binds their hash and the dense-manifest hash.

- [x] **Step 2: Run and confirm failure**

Run: `cd backend && uv run pytest -q tests/retrieval/test_dense_sharded.py tests/retrieval/test_dense_latency.py`

Expected: FAIL because the runtime still opens HNSW/metadata per shard per query.

- [x] **Step 3: Implement resident candidate indexes and bounded shard search**

Load and verify all candidate indexes once during construction, mmap embeddings once, open the global `ChunkOffsetIndex`, and retain immutable handles for the cached runtime's lifetime. Use a `ThreadPoolExecutor(max_workers=4)` only around per-shard candidate search/exact rescoring; merge in the caller thread with deterministic sorting. Materialize metadata only for the final global top 300.

- [x] **Step 4: Implement local latency certification**

Warm the encoder/runtime once, execute all 100 queries, retain every elapsed value, and atomically write `DenseLatencyReport` only when every result is `<= 5.0` seconds. Record platform/machine strings for audit without making model-specific assumptions in readiness.

- [x] **Step 5: Run focused tests and an un-timed small real-FAISS smoke**

Run: `cd backend && uv run pytest -q tests/retrieval/test_dense_sharded.py tests/retrieval/test_dense_latency.py tests/retrieval/test_faiss_ivfpq.py`

Expected: PASS.

- [x] **Step 6: Commit and push**

```bash
git add backend/src/redacted_backend/retrieval/dense_sharded.py backend/tests/retrieval/test_dense_sharded.py backend/tests/retrieval/test_dense_latency.py
git commit -m "feat: serve compressed dense retrieval"
git push origin HEAD
```

### Task 6: Certify recall and wire fail-closed readiness/runtime

**Files:**
- Modify: `backend/src/redacted_backend/retrieval/dense_sharded.py`
- Modify: `backend/src/redacted_backend/domain/readiness.py`
- Modify: `backend/src/redacted_backend/api/readiness.py`
- Modify: `backend/src/redacted_backend/config.py`
- Modify: `backend/src/redacted_backend/jobs/runtime.py`
- Modify: `compose.yaml`
- Modify: `.env.example`
- Modify: `backend/tests/retrieval/test_dense_recall_sharded.py`
- Modify: `backend/tests/test_readiness.py`
- Modify: `backend/tests/jobs/test_runtime_provider.py`

**Interfaces:**
- Consumes: Tasks 1–5 artifacts/runtime and the existing live BM25/corpus/model/worker gates.
- Produces: compressed exact-scan recall certification, schema-v2 API/worker verification, `REDACTED_BEST_AVAILABLE_CHUNKS_PATH`, `REDACTED_BEST_AVAILABLE_LATENCY_REPORT_PATH`, and operational/unapproved best-available readiness.

- [x] **Step 1: Write failing recall/readiness/runtime tests**

Adapt recall fixtures to global offsets and IVF-PQ loader semantics. Assert exact streamed top-300 is compared to candidate-union exact-rescore top-300 and that `0.9799` refuses publication while `0.98` passes.

Add one readiness assertion per Review Focus mismatch: unsupported profile, profile hash, semantics hash, chunks hash, offsets hash/size, shard range/file size/checksum, candidate template hash, recall query hash/count/value, latency dense/query hash/count/max, missing chunks path, and live BM25 count. Assert a fully compatible fixture yields:

```python
{"operational": True, "approved": False, "ready": False,
 "detail": "operational; spending not approved"}
```

Assert runtime construction passes `chunks_path`, uses `FaissIvfPqIndex.load`, performs one BM25 and one dense readiness probe, and remains lazy/cached by immutable prediction basis.

- [x] **Step 2: Run and confirm failure**

Run: `cd backend && uv run pytest -q tests/retrieval/test_dense_recall_sharded.py tests/test_readiness.py tests/jobs/test_runtime_provider.py`

Expected: FAIL because readiness/runtime require schema-v1 HNSW artifacts and have no latency report/chunk path.

- [x] **Step 3: Implement compressed recall certification**

Retain one bounded pass per shard for all 100 exact queries, but materialize only row IDs/scores until final comparison. Permit a CUDA torch matrix backend in cloud builds and NumPy fallback in tests/local verification; both must produce deterministic top-300 IDs under the same score/ID tie rule.

- [x] **Step 4: Implement settings, Compose, readiness, and worker wiring**

Add optional `Path` settings for chunks and latency report, mount no new writable directories, and pass the existing ignored artifact root read-only. Validate schema-v2 and every external file before live BM25 count. Cache checksum verification by manifest plus file size/mtime signatures in the API; reconstruct and verify once again in the worker.

- [x] **Step 5: Run focused tests and Compose validation**

Run:

```bash
cd backend && uv run pytest -q tests/retrieval/test_dense_recall_sharded.py tests/test_readiness.py tests/jobs/test_runtime_provider.py
cd .. && docker compose config --quiet
```

Expected: all tests PASS and Compose exits `0`.

- [x] **Step 6: Commit and push**

```bash
git add backend/src/redacted_backend/retrieval/dense_sharded.py backend/src/redacted_backend/domain/readiness.py backend/src/redacted_backend/api/readiness.py backend/src/redacted_backend/config.py backend/src/redacted_backend/jobs/runtime.py compose.yaml .env.example backend/tests
git commit -m "feat: verify compressed four-list readiness"
git push origin HEAD
```

### Task 7: Expose safe build, benchmark, latency, and verification commands

**Files:**
- Modify: `backend/src/redacted_backend/corpus/cli.py`
- Modify: `backend/tests/corpus/test_cli.py`
- Modify: `README.md`
- Create: `docs/runbooks/runpod-compressed-dense-build.md`

**Interfaces:**
- Consumes: Task 4 build/benchmark, Task 5 latency certification, Task 6 verification.
- Produces: `corpus benchmark-best-available-dense`, compressed `corpus build-best-available-dense`, `corpus certify-best-available-dense-latency`, and `corpus verify-best-available-dense`, plus a credential-safe RunPod runbook.

- [x] **Step 1: Write failing CLI contract tests**

Assert exact path/number arguments reach each function and JSON stdout validates as `DenseBuildBenchmarkReport`, `ShardedDenseBuildReport`, `DenseLatencyReport`, or verification report. Assert benchmark defaults to exactly `100_000`, production build has no flag that can skip checksum/recall/storage gates, latency has no flag above `5.0`, and verification performs no provider calls.

- [x] **Step 2: Run and confirm failure**

Run: `cd backend && uv run pytest -q tests/corpus/test_cli.py`

Expected: FAIL because three commands and compressed arguments are absent.

- [x] **Step 3: Implement CLI commands**

Keep the frozen production profile internal; do not expose arbitrary IVF/PQ values on the production CLI. Allow only batch size, shard size up to 1,000,000, and bounded exact-scan block size as operational knobs. Print canonical JSON reports and use nonzero exit status for failed recall, latency, storage, or verification gates.

- [x] **Step 4: Write the RunPod runbook and local commands**

Document button-by-button account/pod/API-key/SSH steps, but mark them as future actions requiring the user's explicit benchmark approval. Specify 48-GiB-or-larger NVIDIA GPU, 150-GiB persistent workspace, no OpenAI key/user PDFs, public source download, benchmark-first stop, exact cost calculation from measured hours and selected live rate, full-build second approval, shard-by-shard checksum transfer, and pod/volume shutdown choices.

- [x] **Step 5: Run CLI tests and validate every documented local command with `--help` or no-spend fixtures**

Run:

```bash
cd backend
uv run pytest -q tests/corpus/test_cli.py
uv run corpus benchmark-best-available-dense --help
uv run corpus build-best-available-dense --help
uv run corpus certify-best-available-dense-latency --help
uv run corpus verify-best-available-dense --help
```

Expected: PASS/exit `0`; no network or provider generation.

- [x] **Step 6: Commit and push**

```bash
git add backend/src/redacted_backend/corpus/cli.py backend/tests/corpus/test_cli.py README.md docs/runbooks/runpod-compressed-dense-build.md
git commit -m "feat: add compressed dense build workflow"
git push origin HEAD
```

### Task 8: Preserve the real four-list product flow and fictional demo

**Files:**
- Modify: `backend/src/redacted_backend/retrieval/faiss_ivfpq.py`
- Modify: `backend/tests/integration/test_best_available_loopback.py`
- Modify: `backend/tests/retrieval/test_service.py`
- Modify: `backend/tests/jobs/test_orchestrator.py`
- Modify: `backend/tests/test_import_safety.py`
- Modify: `src/features/workspace/DocumentWorkspace.test.tsx` only if the current prediction-ready assertions do not cover basis preservation.
- Modify: `docs/superpowers/specs/2026-11-21-compressed-four-list-dense-design.md`

**Interfaces:**
- Consumes: the completed compressed runtime/readiness and existing orchestrator/UI.
- Produces: regression proof that the two dense lists are independent, all four raw lists persist/fuse, no UI/mock leakage occurs, and the implementation checklist is current.

- [x] **Step 1: Strengthen the no-provider four-list integration test**

Use real small BM25 and compressed-dense fixtures with an injected fake two-stage provider. Assert calls occur in the exact four-list shape, dense results include at least one document absent from both BM25 candidate sets, raw list provenance uses distinct BM25/dense manifest hashes, RRF persists, and the final SSE card remains `best_available_foiarchive_v1` without exposing query/evidence text.

- [x] **Step 2: Run the strengthened test and confirm it fails before fixture wiring**

Run: `cd backend && uv run pytest -q tests/integration/test_best_available_loopback.py tests/retrieval/test_service.py tests/jobs/test_orchestrator.py`

Expected: FAIL until the real compressed fixture is connected.

- [x] **Step 3: Add only the fixture/wiring needed for the real no-spend loopback**

Do not change production UI copy or reintroduce mock routes into `DocumentWorkspace`. Keep `FictionalDemoWorkspace` as the only consumer of invented document/prediction data.

- [x] **Step 4: Run all local verification**

Run:

```bash
cd backend && uv run pytest -q
cd .. && npm test -- --run
npm run typecheck
npm run build
cd backend && uv run alembic heads
cd .. && docker compose config --quiet
git diff --check
```

Expected: backend/frontend suites PASS, one Alembic head remains, build/typecheck/Compose/diff checks pass, and no paid-call flag changes.

- [x] **Step 5: Rebuild current-main API/worker with approvals false and smoke the demo**

Run the current Compose stack with the ignored artifact root, verify `/v1/health`, worker heartbeat, and that the fictional demo opens/closes while preserving an uploaded document. Until real dense artifacts exist, best-available remains fail closed; model-only stays operational/unapproved.

- [ ] **Step 6: Commit, push, and request whole-branch review**

```bash
git add backend/tests src docs/superpowers/specs/2026-11-21-compressed-four-list-dense-design.md
git commit -m "test: verify compressed four-list product flow"
git push origin HEAD
```

### Task 9: Cross the paid cloud and model boundaries only with explicit approvals

**Files:**
- Runtime artifacts only under ignored `var/corpus/best-available/dense/`
- Update after each completed gate: `docs/superpowers/specs/2026-11-21-compressed-four-list-dense-design.md`
- Update after measured results: `README.md`

**Interfaces:**
- Consumes: completed/pushed Tasks 1–8 and the user's RunPod credentials/approvals.
- Produces: verified local compressed artifacts, operational/unapproved best-available readiness, exact two-stage model cost, and—only after another approval—one real four-list prediction.

- [ ] **Step 1: Stop and provide exact RunPod setup instructions**

Guide the user through account, billing, API/SSH credential creation, and local untracked storage one button at a time. Confirm the credential exists without printing it.

- [ ] **Step 2: Quote and obtain one benchmark cap**

Use the selected live pod rate and a fixed maximum benchmark duration to state an exact maximum dollar amount. Start no pod before the user explicitly approves that amount.

- [ ] **Step 3: Run only the 100,000-passage benchmark, stop the GPU, and report measurements**

Verify its JSON/checksums, calculate expected and conservative maximum full-build GPU/storage/transfer cost from measured throughput and bytes, and obtain a separate explicit full-build approval.

- [ ] **Step 4: Build, certify, and transfer only after full-build approval**

Run resumably; report checkpoints at least hourly. Require recall@300 `>= 0.98`, final artifacts `<= 28 GiB`, and local checksum verification before stopping/deleting anything recoverable.

- [ ] **Step 5: Certify local latency and fail-closed readiness**

Run all 100 warm searches locally and require each `<= 5.0` seconds. Rebuild API/worker with all paid flags false and require `best_available_foiarchive_v1.operational=true`, `approved=false`, `ready=false`.

- [ ] **Step 6: Count the exact two provider requests without generation**

Show expected and maximum combined own-query plus evidence-bound prediction cost. Obtain a new explicit approval before changing `REDACTED_ALLOW_PAID_MODEL_CALLS` and `REDACTED_ALLOW_BEST_AVAILABLE_CALLS` for one prediction.

- [ ] **Step 7: Run one real prediction, close approvals, verify the frontend, and commit measured documentation**

Record request IDs/usage/cost/provenance without secrets or evidence text, restore every approval flag to false, verify the fictional demo still works, update the checklist/readme, commit, and push.
