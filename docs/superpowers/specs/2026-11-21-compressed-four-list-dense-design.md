# Compressed four-list dense retrieval design

## Intent

Make the existing `best_available_foiarchive_v1` production path operational with Andrew Tang's full four-list retrieval shape: visible context and a model-generated own query each run independently through BM25 and corpus-wide dense retrieval, followed by document-level reciprocal-rank fusion and an evidence-bound prediction. The result must be a real, usable prediction path, not a demo or a sparse-only result presented as dense retrieval.

The existing fictional demo remains always available and isolated. The existing exact-paper basis remains separate and fail closed. The four fixed corpus limitations remain unchanged.

## Current verified state

- The checksum-locked HistoryLab source contains 4,889,089 parsed rows.
- The completed best-available corpus contains 4,300,534 non-empty documents and 14,549,269 300-word, zero-overlap passages.
- The completed OpenSearch BM25 index has an exact live count of 14,549,269 and a published manifest.
- The pinned model is `BAAI/bge-base-en-v1.5` at revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, with 768-dimensional normalized embeddings and the existing query instruction.
- The existing HNSW-flat design has a deterministic 62.55-GiB minimum for float16 embeddings, duplicate float32 candidate vectors, and offsets before graph or metadata storage. Only 33.3 GiB was free after the corpus and BM25 artifacts were completed.
- The pinned BGE model measured 17.35 passages per second on the local Apple MPS device, making the full local build operationally impractical.

## Goals

- Preserve independent corpus-wide dense retrieval for both query forms; dense results must not be reranked BM25 candidates.
- Preserve the pinned BGE model, revision, query instruction, normalized embeddings, top-300 list shape, document-level RRF, top-100 documents, and two excerpts per document.
- Retain float16 corpus embeddings so the final ranking is an exact inner-product rescore over the union of approximate candidates.
- Replace only the candidate-generation and metadata-storage implementation with a compressed, versioned artifact profile.
- Require measured recall@300 of at least 0.98 on the existing 100-query certification set before publication.
- Keep final local dense artifacts within 28 GiB, leaving practical headroom on the current machine.
- Complete each warm local dense search in no more than five seconds on the target Mac, so the two independent dense lists add no more than approximately ten seconds before model generation.
- Make the build resumable, checksum-bound, deterministic, and safe to perform once on a rented cloud GPU.
- Run the final dense retrieval locally so ordinary product use does not depend on a permanently rented GPU or managed vector database.
- Keep all paid model-call approvals false throughout artifact construction and verification.

## Non-goals

- This does not claim the unavailable exact Andrew-paper corpus or metadata.
- This does not change the fictional demo, the uploaded-document UI, or expose queries/evidence/research diagnostics in ordinary prediction cards.
- This does not enable or perform an OpenAI generation.
- This does not upload the user's redacted PDFs, OpenAI key, or local database to the GPU provider.
- This does not infer missing corpus labels or publication dates.
- This does not delete the current source, chunk, BM25, Docker, or user-owned files to make space.

## Anti-goals

- Do not label a BM25 candidate reranker as independent dense retrieval.
- Do not weaken or skip recall certification to make a compressed index publish.
- Do not silently fall back to sparse-only or model-only prediction when dense artifacts are missing or invalid.
- Do not change the prediction basis after a job has been scheduled.
- Do not require an always-on high-memory cloud server merely to keep the demo or local product usable.

## Approaches considered

### Selected: compressed local candidate indexes, built once on a cloud GPU

Store normalized float16 embeddings for exact rescoring, use a versioned FAISS IVF-PQ candidate index, and reuse the existing chunk JSONL through a checksum-bound byte-offset table instead of copying metadata into every dense shard. Build on a temporary cloud GPU, download the final artifacts, verify them locally, and serve both BM25 and dense retrieval from the current machine.

This keeps the full four-list method, removes the ongoing cloud dependency, fits the available storage, and preserves a quantitative recall gate. The user selected this approach.

### Rejected: managed Qdrant or another hosted vector database

This would remove local storage pressure and simplify serving, but it adds a permanent paid dependency, another production credential, provider-specific indexing behavior, and a harder exact-rescore/provenance story. It remains a future scale-out option, not the first implementation.

### Rejected: current HNSW-flat artifacts served remotely

This requires the least algorithmic code change, but the artifacts exceed local capacity and production serving would require a costly always-on machine with enough RAM or repeated large disk reads. It solves the build but creates an avoidable recurring infrastructure requirement.

## Artifact architecture

### Compatibility and versioning

Introduce a dense artifact profile distinct from the already-published corpus and BM25 scientific identity. The existing corpus and BM25 manifests remain immutable and valid.

The new sharded dense manifest records:

- the existing corpus manifest hash and chunk-file SHA-256;
- the existing pinned semantic settings: BGE model/revision, query instruction, normalization, dimensions, metric, result limit, and recall threshold;
- `candidate_index = "faiss-ivfpq-exact-rescore-v1"`;
- an immutable artifact-profile hash covering training-sample selection, IVF/PQ settings, probe count, candidate count, shard layout, offset format, FAISS version, and metadata lookup version;
- exact file sizes, SHA-256 values, row ranges, and passage counts for every shard;
- the certification-query hash/count and measured recall@300.

Candidate implementation parameters are versioned as an artifact profile because they may be tuned without changing passage text, embedding semantics, or the already-built BM25 index. Readiness accepts only explicitly supported profiles and never treats an unknown profile as compatible.

### Shared metadata lookup

Create one `chunks.offsets.u64` file containing the starting byte offset of every non-empty JSONL passage in the existing `chunks.jsonl`, in corpus order. Its count, byte size, SHA-256, source chunk SHA-256, and format version are recorded in the dense manifest.

Dense shards do not create `metadata.jsonl`. At result materialization time, the runtime maps a global row to `chunks.offsets.u64`, seeks into the immutable checksum-verified `chunks.jsonl`, reads exactly one line, validates its chunk ID, and returns the existing `CorpusChunk`. This removes approximately one full duplicate of the 25-GiB chunk file.

### Dense shards

Keep consecutive shards of at most 1,000,000 passages. Each shard contains:

- `embeddings.f16`: normalized 768-dimensional vectors used for exact rescoring;
- `candidate.faiss`: a trained IVF-PQ index containing compressed candidate codes and stable local row IDs;
- `manifest.json`: immutable range, training, parameter, size, and checksum data.

Shard construction remains atomic: write to a temporary sibling directory, fsync every artifact, verify sizes/checksums, publish the shard manifest, then rename the directory. A completed compatible shard is reused after full verification. An incomplete temporary directory is never treated as a shard.

### Deterministic IVF-PQ training

Use stable-hash reservoir selection from the verified corpus rather than the first passages or random process state. The artifact profile declares the sample rule and sample count. Training input is normalized float32 derived from the pinned encoder. The initial profile may tune `nlist`, PQ subvector count, bits per code, `nprobe`, and retained candidates, but publication freezes the selected values and their hash.

The builder must be able to rebuild candidate files from completed float16 embeddings without re-encoding the corpus. This permits recall tuning after the expensive encoding pass.

## Retrieval behavior

For each visible-context query or model-produced own query:

1. Encode the query once using the pinned BGE query instruction.
2. Search each shard's IVF-PQ candidate index independently.
3. Combine stable global row IDs from all shards.
4. Read their float16 vectors and compute exact float32 inner products.
5. Apply deterministic score and row-ID tie breaking.
6. Return the global top 300 passages.

The existing orchestrator then obtains four independent raw lists:

1. visible context → BM25;
2. visible context → dense;
3. own query → BM25;
4. own query → dense.

Existing self-document exclusion, document merging, reciprocal-rank fusion, top-100 document selection, two-excerpt limit, provenance persistence, basis binding, and evidence-bound prediction remain unchanged.

The local runtime must benchmark warm searches over the complete artifact set. A supported compressed profile is not production-ready unless each of the 100 certification queries completes its dense search in at most five seconds on the target Mac. The runtime may keep candidate indexes resident, search shards concurrently within a bounded worker pool, and cache immutable file mappings, but it must remain within measured local memory limits and preserve deterministic rankings.

## Recall certification

Reuse the existing 50 application-shaped queries plus 50 deterministic corpus-derived queries. For every query, compare the approximate candidate-plus-exact-rescore result against a streaming exact scan of all float16 embeddings.

Publication requires mean recall@300 of at least 0.98 and records per-query results in a build report. If certification fails, the top-level manifest is not written and readiness stays false. Tuning may increase `nprobe`, retained candidates, or adjust IVF-PQ profile settings; completed embeddings remain reusable.

## Storage budget

The fixed float16 embeddings require 20.81 GiB and the shared offsets require approximately 0.11 GiB. IVF-PQ codes, row IDs, centroids, shard manifests, and certification data are budgeted at no more than 5 GiB, with a hard final-artifact ceiling of 28 GiB.

The builder performs a real free-space check before every shard and before artifact download. It stops without deleting data on an actual allocation or write failure. Local transfer is shard-by-shard and checksum-verified so a failed transfer can resume without keeping duplicate complete copies.

## Cloud build workflow

The cloud machine is temporary build infrastructure, not a runtime dependency.

1. Complete and verify all local code/tests without cloud spending.
2. Create a temporary RunPod GPU pod with at least 48 GiB VRAM and a 150-GiB persistent workspace.
3. Run a 100,000-passage benchmark first. Record encoding throughput, peak VRAM/RAM, artifact bytes per passage, and estimated full-build time/cost.
4. Stop and show the user the measured full-build expected and maximum cost. Do not start the full corpus build without a separate explicit spending approval.
5. Download the public pinned source and repository code on the pod; do not upload local secrets or user PDFs.
6. Reproduce/verify the corpus manifest identity, build the offset table and resumable dense shards, then run recall certification.
7. Transfer final shards and manifests to the ignored local artifact directory, verify every checksum, and delete no local artifact.
8. Stop the GPU pod after verification. Retain or delete its volume only according to the user's explicit choice and after confirming local recovery.

## What the user must do

Nothing is required from the user while the code, tests, local fixtures, and benchmark automation are being built.

At the cloud boundary, the user will need to:

1. create or sign into a RunPod account;
2. add a payment method or small prepaid balance;
3. create a narrowly scoped API key or SSH access credential and save it in the local untracked credential location specified by the runbook, never paste it into chat or commit it;
4. approve the exact benchmark cap before the first paid GPU run;
5. after the benchmark, approve or decline the separately stated maximum full-build spend;
6. keep the Mac online during the final resumable artifact transfer.

The runbook will give button-by-button RunPod instructions at that point. The user does not need to supply a corpus, configure FAISS, run commands, or manage the pod manually.

## Runtime and readiness

The API and worker continue mounting the ignored local artifact root read-only. Dense artifact verification is cached by manifest and file signatures in the API, and independently repeated when the worker constructs a runtime.

`best_available_foiarchive_v1` becomes operational only when:

- the existing corpus and BM25 manifests verify;
- the live OpenSearch count is exactly 14,549,269;
- the supported compressed dense profile and all shard checksums/ranges verify;
- the chunk-offset table matches the immutable chunk file;
- certification contains the frozen 100-query hash and recall@300 is at least 0.98;
- the worker heartbeat is current and the model key is configured.

Provider approval remains a separate gate. With approvals false, readiness must report `operational=true`, `approved=false`, `ready=false`.

## Provider-call boundary

Artifact building and retrieval verification use no OpenAI generation. Once operational readiness is proven, count the exact own-query request and the exact evidence-bound prediction request without generation where the provider permits. Show expected and maximum combined cost and obtain a new explicit two-stage approval before enabling the master and best-available flags for one real prediction.

## Failure handling

- Unsupported or mismatched artifact profiles fail closed.
- Missing, truncated, or checksum-mismatched offset/vector/candidate files fail readiness and runtime construction.
- Failed candidate tuning never invalidates completed embeddings.
- Interrupted cloud builds resume only from complete verified shards.
- Interrupted downloads retain verified completed shards and retry the missing shard.
- A failed provider request follows the existing durable-attempt rules and is never silently resent when outcome is uncertain.
- No fallback changes a scheduled job's prediction basis.

## Test and verification strategy

- Test-first unit coverage for offset creation/random lookup, profile hashing, manifest validation, deterministic training selection, compressed candidate persistence, exact rescoring, tie breaking, corruption, and incompatible profiles.
- Small real-FAISS fixtures proving restart/resume and candidate rebuild without re-encoding.
- Certification tests proving publish at or above 0.98 and refusal below 0.98.
- Runtime tests proving both dense lists are corpus-wide and independent of BM25 candidates.
- Integration coverage preserving all four raw ranked lists, provenance, RRF, and basis binding.
- Readiness tests for operational/unapproved state and every fail-closed artifact condition.
- Existing backend, frontend, migration, Compose, demo-isolation, and fake-transport loopback suites must all remain green.
- One no-provider local smoke query must return real BM25 and dense evidence before any paid model approval is requested.

## Checklist

- [x] Select compressed local dense artifacts with a one-time cloud GPU build.
- [x] Freeze intent, constraints, safety gates, and user responsibilities in this design.
- [x] Approve this design, including the five-second maximum per warm dense search.
- [x] Write the implementation plan.
- [x] Approve the implementation plan.
- [x] Version the compressed artifact profile and schema-v2 manifests.
- [x] Build and verify the shared checksum-bound chunk-offset index.
- [ ] Implement profile/offset/candidate/runtime changes test-first.
- [ ] Verify local suites and the fictional demo.
- [ ] Complete the capped cloud benchmark after explicit approval.
- [ ] Complete the full cloud build after separate explicit approval.
- [ ] Download and verify artifacts locally.
- [ ] Verify four-list operational/unapproved readiness.
- [ ] Count and separately approve one real two-stage prediction.

