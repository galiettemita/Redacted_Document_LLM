# Always-On Prediction Worker and Corpus Unblock Design

## Intent

Move the uploaded-document path materially closer to real predictions without weakening Andrew Tang's disclosed retrieval method and without disturbing the permanent fictional demo. The next production increment makes the real Dramatiq worker an ordinary, observable service even while corpus or model gates are closed. It then removes the artificial storage preflight as a corpus blocker and pursues a complete official corpus source that can support the paper's four locally reproduced retrieval lists.

The real path must remain fail-closed. A live worker is not a prediction, a hosted search result is not the paper's retriever, and unavailable dependencies never cause a simulated result to appear in the uploaded-document workspace.

## Goals

- Start the real worker with PostgreSQL and Redis during the normal Compose lifecycle.
- Publish the worker heartbeat without loading corpus artifacts, contacting OpenSearch, or constructing a model gateway during module import.
- Initialize the existing real retrieval/model runtime only after all API-side scheduling gates pass and an actual job arrives.
- Cache one successfully initialized runtime per worker process while allowing a failed initialization to be retried after dependencies recover.
- Preserve bounded retries, durable job failure, idempotency, cancellation, and the existing no-paid-call gate.
- Keep the fictional demo mounted, isolated, and available throughout this work.
- Stop treating a fixed free-space threshold as permission to begin corpus work. Continue until the filesystem actually refuses a write, retaining resumable checkpoints and checksums.
- Identify and acquire one complete, versioned official corpus source with explicit collection labels, including the separate `cables` collection, before calling the corpus ready.
- Keep the UI simple: uploaded-document users continue to see concise preparation, unavailable, running, and model-prediction states rather than implementation diagnostics.

## Non-goals

- No mock runtime, sample route, synthetic retrieval evidence, or simulated prediction in the real uploaded-document path.
- No paid model call, production deployment, or external message to corpus maintainers without separate explicit approval.
- No substitution of the History Lab MCP's capped semantic/full-text search for local BM25 and BGE top-300 lists.
- No relaxation of corpus counts, collection coverage, exclusion rules, self-document exclusion, manifest matching, or retrieval provenance.
- No frontend redesign. The fictional demo remains the only place where invented content is rendered.
- No claim that a worker heartbeat means retrieval or the model is ready.

## Current failure and evidence

`redacted_backend.jobs.tasks` currently calls `build_analysis_orchestrator()` at import time. That builder requires the complete corpus and dense artifacts, a reachable OpenSearch index, an approved API key, and paid-call approval. Dramatiq cannot finish importing the task module when any one is absent, so its heartbeat middleware never starts. The worker liveness signal is therefore incorrectly coupled to prediction readiness.

The corpus source is a separate problem:

- History Lab's hosted MCP health endpoint responds, but a direct `list_corpora` tool call currently returns `Corpus API error (502)`. Its implementation uses the same public PostgREST base URL for corpus operations.
- The official `HistoryLab/foiarchive` revision `411cf6f86b38342683e27148be167c6094d1a10c` contains one 7,976,348,471-byte gzip. Its card totals 4,949,871 documents across 11 collections.
- The current official MCP catalog documentation describes a later separate `cables` collection of roughly 718,000 documents. That collection is absent from the frozen Hugging Face revision, so the revision cannot by itself meet the paper-era coverage contract.
- History Lab's official release page offers research access to more specific slices or different metadata. A current complete export with explicit `corpus` labels, or a repaired enumerable PostgREST endpoint, is therefore the scientifically valid unblock. A top-k search endpoint is not.

Free space is not the source-completeness problem. The fixed 150 GiB preflight will become advisory and will not prevent capture or source audit.

## Approaches considered

### Chosen: independently live worker, then complete-source acquisition

Decouple process liveness from the expensive real runtime. Start the worker normally, retain all prediction gates, and lazily construct exactly the existing real orchestrator for a queued job. In parallel after that increment is verified, adapt the corpus tooling to accept a complete official export efficiently and pursue the official export/API route.

This produces an immediately testable production improvement while keeping the scientific bottleneck explicit.

### Rejected: initialize the prediction runtime before publishing a heartbeat

This is the current behavior. It makes a missing model key look like a dead queue worker, prevents independent diagnosis, and blocks ordinary service startup.

### Rejected: publish a heartbeat and queue jobs without stronger retrieval checks

Worker liveness alone is insufficient. API scheduling must still require compatible corpus/index manifests, required dense artifact files, a reachable matching OpenSearch index, model configuration, paid-call approval, and a current worker heartbeat. If those checks race with a later dependency failure, the worker retries and then records an explicit durable failure.

### Rejected: build from the 2025 Hugging Face file and silently add hosted search results

That would combine source versions, omit a complete enumerable `cables` collection, and change retrieval rankings. It would not reproduce the disclosed experiment.

### Rejected: wait for 150 GiB or 250 GiB free before any corpus action

Those were conservative planning estimates, not scientific requirements. Capture is already resumable and checksummed. The operator has directed the project to continue until actual filesystem exhaustion, so capacity will be reported but not used as a hard gate.

## Worker architecture

### Process boot

The task module may construct only lightweight operational dependencies:

1. settings;
2. the Redis broker;
3. the Redis heartbeat middleware;
4. a PostgreSQL `Database` handle used for job state and failure recording; and
5. an `AnalysisRuntimeProvider` that holds a builder callable but does not invoke it.

Redis and PostgreSQL remain required because a worker that cannot receive a message or durably record job state is not operational. Corpus files, OpenSearch, and model approval are not worker-process boot requirements.

The worker service moves out of the `retrieval` Compose profile and depends only on healthy PostgreSQL and Redis. OpenSearch remains in the `retrieval` profile. Starting the normal stack therefore creates a real, idle worker and a current heartbeat; starting the retrieval profile adds the corpus search engine when artifacts are available.

### Lazy real runtime

`AnalysisRuntimeProvider.get()` is thread-safe and follows these rules:

- The constructor performs no runtime build.
- The first job calls the unchanged real builder with the shared database handle.
- One successful runtime is cached for the life of the process.
- A failed build is not cached. A later retry or later job may build again after configuration or services recover.
- Concurrent callers cannot build two runtimes.
- There is no fallback builder and no mock implementation in production code.

Settings are still process configuration. Changing corpus paths, the model key, or paid-call approval requires a worker restart. Service recovery at the same configured addresses does not.

### Scheduling and operational readiness

The API keeps independent database, worker, corpus/retrieval, and model components. Worker readiness continues to mean only that the expiring Redis heartbeat is current.

Before a job can be queued, corpus/retrieval readiness additionally verifies:

- compatible corpus, BM25, and dense manifests;
- the required dense artifact files exist;
- OpenSearch is configured and reachable; and
- the manifest-named BM25 index exists with the manifest's chunk count.

These request-time checks use metadata and file existence only. They do not re-hash multi-gigabyte artifacts on every readiness request. The worker's one-time real runtime initialization performs the authoritative dense checksums, metadata count validation, BM25 probe, and dense probe before any model call.

The existing model gate still requires both a configured key and explicit paid-call approval. This design does not enable either.

### Job error and retry behavior

Runtime initialization failure is treated like an operational dependency failure:

- retry under the existing bounded Dramatiq policy;
- never call the model unless initialization completes;
- after the final attempt, fail the owning job with `analysis_runtime_unavailable`;
- store only a sanitized exception class/category, not secrets, URLs containing credentials, file contents, or provider responses; and
- use the separately constructed database handle to record failure even when the orchestrator never existed.

The boundary is exact: every exception raised before `AnalysisRuntimeProvider.get()` returns a complete orchestrator is wrapped as a sanitized `analysis_runtime_unavailable`, including manifest validation, dense checksum/load, encoder load, OpenSearch probe, and gateway construction. Existing typed model and retrieval codes apply only to exceptions raised by `orchestrator.run()` after successful construction. Passage revision/deletion cancellation and PostgreSQL duplicate-execution locking remain unchanged.

## Corpus unblock architecture

### Source acceptance contract

A source is acceptable only if it is official and supports all of the following:

- stable complete enumeration, not ranked top-k search;
- explicit collection identity for every record;
- `cables` present and non-empty;
- exact per-collection counts before and after capture, or an immutable official artifact with byte size and SHA-256;
- full document text and the metadata required by the exclusion manifest;
- deterministic document identifiers and ordering; and
- a frozen source revision or locally checksummed capture.

“Complete” is the paper-derived contract already frozen in `backend/corpus/exclusions/v1.yaml`, not merely the newest available data. Before exclusions, a candidate capture must contain only the canonical labels `briefing`, `cabinet`, `cables`, `cfpf`, `cia`, `clinton`, `cpdoc`, `frus`, `kissinger`, `nato`, `un`, and `worldbank`; contain a non-empty `cables` collection; total 5,550,000 documents with the paper's ±5,000 rounding tolerance; map the CIA reading-room exclusion to `briefing`; and contain exactly the ten FRUS volumes published during 2019–2024 identified from the frozen official FRUS metadata. A later official export is acceptable only if it passes those gates. Otherwise it is retained as evidence but reported as a source-version gap, and no index built from it becomes ready.

The repaired History Lab PostgREST API would satisfy the transport side through the existing keyset capture. An official paper-compatible bulk export would satisfy it through the streaming bundle adapter below. If neither is publicly reachable, the next external step is a concise request to History Lab for the paper-compatible research export with explicit collection labels. Preparing that request is in scope; sending it is not authorized by this design.

### Streaming bundle adapter

The first adapter targets the concrete official Hugging Face artifact contract rather than guessing a future export: a local `.jsonl.gz` at a pinned dataset revision, with the nine documented fields `doc_id`, `classification`, `authored`, `title`, `body`, `source`, `pg_cnt`, `char_cnt`, and `word_cnt`. The authoritative collection value is `corpus` when a future official row supplies it; for the nine-field artifact, `source` is accepted as a collection value only when its normalized value is one of the 12 canonical labels. A URL, null, or unknown value is never reinterpreted. The dataset-card catalog is frozen as a small JSON sidecar containing `corpus` and `doc_cnt`; computed counts must match it.

The corpus CLI will gain an audit/import path for that official `.jsonl.gz` plus catalog sidecar:

1. stream decompression; never create an uncompressed copy of the source bundle;
2. validate the frozen source SHA-256 and byte size;
3. validate every row against the normalized corpus schema;
4. count and hash each collection while reading;
5. fail on absent, unknown, or unsafe collection identifiers;
6. write atomic, resumable normalized collection artifacts only when downstream tooling requires them; and
7. emit the same `ApiCaptureManifest`-level provenance needed by coverage, chunk, and index builds, with an explicit bulk-export schema version.

This adapter will not infer a collection from retrieval results or quietly map arbitrary origin URLs. If the official artifact lacks an explicit usable collection field, the audit report remains unready and records the observed field distribution for a maintainer request.

A materially different future export is audited first and receives a new explicit adapter/schema version only after its official schema and catalog are available. The plan does not pre-build an importer against an imagined format.

### Storage behavior

`corpus preflight` remains a read-only capacity report, but insufficient estimated headroom no longer exits with a blocking status by default. An explicit operator flag may enforce a chosen minimum for unattended environments. Capture/import stages catch `ENOSPC`, preserve the last atomic checkpoint, mark the capture incomplete, and can resume after space is added.

Scientific readiness depends on source and artifact completeness, hashes, counts, exclusions, and index validation—not on a forecast of free bytes.

## UI behavior

No new research diagnostics are added to the sidebar. In the uploaded-document workspace:

- an idle live worker does not create a success claim;
- closed corpus/model gates continue to produce a concise unavailable state;
- queued/running states appear only for a real job; and
- only a provider result from the real orchestrator is labelled `Model prediction`.

The `View fictional demo` button and its isolated simulated progression remain available from upload and real workspaces. No worker or corpus change imports demo data into backend APIs.

## Verification

### Unit and integration tests

- Provider construction does not call the runtime builder.
- A successful build is cached exactly once, including concurrent calls.
- A failed build is retried on the next call and never replaced by a fallback.
- Importing the task module with PostgreSQL/Redis configured but corpus/model unavailable does not build the orchestrator.
- Final runtime-initialization failure durably marks the job with a sanitized code.
- Scheduling remains blocked when corpus manifests, dense files, OpenSearch index/count, model approval, or worker heartbeat is absent.
- The worker service is in default Compose configuration and no longer depends on OpenSearch.
- Corpus preflight reports low capacity without blocking by default.
- Streaming source audit rejects missing/unknown labels and proves counts/hashes without expanding the gzip to a second raw copy.

### Live local proof, without a paid call

1. Rebuild and start PostgreSQL, Redis, backend, and worker without corpus artifacts or a model key.
2. Confirm `/v1/readiness` reports `worker.ready=true`, `corpus.ready=false`, `model.ready=false`, and `prediction_ready=false`.
3. Use the fictional demo and an uploaded real document to confirm no regression and no real job dispatch.
4. Start the retrieval profile without complete artifacts and confirm readiness remains closed.
5. Inspect logs and persisted jobs to confirm no model request was made.

The first paid one-passage browser trace remains a later, separately approved gate after the complete corpus and indexes pass.

## Research references

- History Lab's official bulk-release announcement: <https://lab.history.columbia.edu/news/foiarchive-hugging-face>
- Frozen official dataset: <https://huggingface.co/datasets/HistoryLab/foiarchive>
- Current official MCP implementation and corpus catalog: <https://github.com/history-lab/history-lab-mcp>
- Official retired API notice: <https://github.com/history-lab/foiarchive-api>
- Official PostgREST schema/build repository: <https://github.com/history-lab/foiarchive-postgres>

Endpoint probes and artifact metadata in this design were rechecked on 2026-11-15. They are evidence for source selection, not durable readiness assertions; the capture command revalidates the selected source at execution time.

## Delivery and rollback

The worker increment is complete when its code, tests, Compose lifecycle, and live no-paid-call proof pass. The corpus-tooling increment is complete when the official frozen bundle has a reproducible streaming audit, the source gap is recorded, and an exact maintainer request is prepared if no acceptable public source is reachable. Task 14 and corpus readiness remain incomplete until a candidate official source actually passes the paper-derived coverage contract. External source availability is never converted into a false implementation failure or a false readiness claim.

The increment will be test-driven, committed, and pushed in focused commits:

1. lazy runtime/provider and durable failure behavior;
2. default worker Compose lifecycle plus operational retrieval readiness;
3. advisory storage preflight and streaming official-source audit/import support; and
4. documentation/checklist evidence.

Each change is additive or locally reversible. If the worker change fails, the API's gates keep real jobs blocked and the fictional demo remains usable. No schema migration or destructive corpus rewrite is required for the worker increment.
