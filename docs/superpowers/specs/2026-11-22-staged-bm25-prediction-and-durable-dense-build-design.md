# Staged full-corpus BM25 prediction and durable dense build design

## Intent

Ship one honest, real archive-assisted prediction through the ordinary uploaded-document **Predict** flow by November 23, 2026, while the complete 14,549,269-passage dense index continues building locally in the background. The staged result uses the real OpenAI model boundary and real evidence from the full best-available FOIArchive BM25 index. It must never be presented as Andrew Tang's completed four-list method.

## User-visible outcome

The user uploads a PDF, the existing production workspace detects and prepares its redactions, and the first visible redaction can be predicted with the existing **Predict passage** control. There is no new preview route, preview button, or simulated result.

The customer-facing card uses only the existing generic prediction presentation and the provider result. It does not display a staged-basis label or mention BM25, dense retrieval, the background build, Andrew's method, implementation details, or research limitations.

The result is a genuine two-stage provider workflow:

1. GPT produces the passage's own search query from the visible-only example.
2. The visible-context query and own query independently search the complete 14.5-million-passage BM25 index.
3. The two real ranked lists are self-document filtered and fused at document level.
4. GPT receives the fused evidence and produces the passage-linked prediction.

No basis, query, evidence, benchmark, limitation, token, or build diagnostic is shown in the ordinary prediction card. Existing provider provenance, request hashes, costs, retrieval evidence, immutable prediction basis, and the honest scientific boundary remain stored in the backend and documentation.

The software, no-spend loopback, and local deployment can be completed without user input. Producing the first live result additionally depends on the user uploading the intended PDF and configuring the hard staged-provider spending ceiling. The delivery target does not silently waive either requirement.

## Honest scientific boundary

The staged basis is `best_available_foiarchive_bm25_v1`, separate from both `best_available_foiarchive_v1` and `paper_exact_v1`.

- It searches all 14,549,269 passages, not a sample.
- It produces two independent BM25 lists, not four lists.
- It does not fabricate empty dense lists or claim dense retrieval happened.
- It does not satisfy full best-available four-list readiness.
- Existing full four-list and exact-paper gates remain fail closed.
- Previously completed staged predictions retain their immutable BM25 basis after full dense retrieval becomes available.

The retrieval trace schema records its list shape and version. The existing `FourRankedLists` trace remains unchanged for full retrieval. A separate two-list trace contains only `context_bm25` and `own_bm25`, uses the same self-document exclusion, document-level RRF constant, top-100 document limit, two-excerpt limit, evidence serialization, and provenance rules, and is accepted only by the staged basis.

## Staged passage limit

The backend, not the browser, enforces an initial maximum of one completed or active staged BM25 prediction per uploaded document. Document order is the persisted passage order used by the production workspace. The first active prepared passage is eligible; attempting to schedule another staged passage returns a durable, concise blocker and makes no retrieval or provider call.

Failed and cancelled jobs do not consume the document limit. A completed or currently queued/running staged job does consume it. Raising the configured limit from one to two later makes the next prepared passage eligible without changing or rerunning the first result.

The limit is a production setting named `REDACTED_BM25_STAGE_MAX_PREDICTIONS_PER_DOCUMENT`, initially fixed to `1`. It may later be raised deliberately after cost and operational review. The browser must not be able to override it. **Predict all passages** schedules only the eligible first passage under this basis and reports the remaining passages as not yet enabled; it never silently creates paid jobs for all passages.

## Readiness and spending gates

The staged basis is operational only when all of these are true:

- PostgreSQL is ready;
- the real worker heartbeat is ready;
- the OpenAI key is configured;
- the pinned best-available corpus and BM25 manifests verify;
- the live OpenSearch index count is exactly 14,549,269;
- the staged runtime can execute a real BM25 readiness probe.

The user grants standing authorization for staged-BM25 provider calls and does not require per-call approval. The application cannot currently read the OpenAI account's available-credit balance, so it must enforce a separate local hard ceiling, `REDACTED_STAGED_BM25_SPEND_CAP_USD=10.00`, before unattended spending is enabled. The ceiling is stored only in untracked runtime configuration.

All paid flags stay false during implementation, tests, deployment, and retrieval checks. For a live eligible job, the backend computes the maximum query-call cost before dispatch. After the query returns and evidence is available, it computes the maximum prediction-call cost. It dispatches a call only when the maximum of that call plus durable staged spend already claimed remains within the configured ceiling. Provider attempts are claimed before dispatch, actual usage is recorded when known, and an uncertain provider outcome retains its maximum reservation. When the ceiling is missing, invalid, exhausted, or insufficient, the job fails closed before another provider request. The browser cannot raise or bypass the ceiling.

The master and staged-BM25 approval flags may be enabled after the no-spend deployment checks because this written standing authorization replaces individual call approvals, but only after the hard dollar ceiling is configured. Other basis flags remain separately controlled.

Readiness priority for new jobs is:

1. `paper_exact_v1` when separately operational and approved;
2. `best_available_foiarchive_v1` when the complete four-list artifacts are operational and separately approved;
3. `best_available_foiarchive_bm25_v1` when the staged full-corpus BM25 path is operational and separately approved;
4. `model_only_v1` when separately operational and approved.

No job changes basis after scheduling and no failure falls back inside a job.

## Durable background dense build

After the staged feature passes its no-spend local deployment checks, the complete int8 dense pipeline starts even if the first paid demonstration is still waiting for its document. It runs as an operating-system-level background process rather than a Codex tool session. The runner uses the committed production code and:

1. runs the fixed 100,000-row CPU benchmark offline;
2. records the JSON report, logs, PID, timestamps, current phase, and exit code under ignored `var/corpus/best-available/dense-build-state/`;
3. requires projected final artifacts `<= 16 GiB` and at least 4 GiB projected free host storage;
4. chooses the smallest physical shard size whose measured fixed overhead still passes the 16-GiB ceiling, preferring approximately two-to-four-hour embedding checkpoints;
5. starts or resumes the real 14,549,269-row int8 build only after those gates pass;
6. publishes no top-level dense manifest until checksum, recall@300 `>= 0.98`, and total-size gates pass;
7. runs same-worker latency certification and requires every one of the 100 warm dense searches to complete in `<= 5.0` seconds;
8. verifies live BM25 count and reports full four-list operational readiness with spending still unapproved.

The runner is started with `nohup` and `caffeinate` on the Mac so it survives Codex turns and prevents sleep while active. It never deletes corpus, BM25, Docker cache, user files, or completed checkpoints. A second invocation detects the live PID or resumes verified production checkpoints instead of starting a competing build. Benchmark interruption may require a benchmark restart; production embedding checkpoints remain reusable.

The estimated 12-to-14-day duration is an operational estimate, not a promise. Completion is determined only by the published checksummed manifest, recall report, latency report, and readiness gates.

## Failure behavior

- Missing or mismatched BM25/corpus identities close staged readiness.
- An unavailable worker or model key closes staged readiness.
- A later-than-eligible passage is blocked before provider dispatch.
- Empty or incomplete BM25 lists fail retrieval instead of producing unsupported evidence.
- Provider uncertainty is durably claimed and never automatically resent.
- Passage edits, deletion, expiry, or basis mismatch cancel the job using the existing revision-bound rules.
- Dense benchmark/build failure leaves the staged BM25 feature intact and leaves paid flags unchanged.
- Dense completion does not initiate a provider call or expand the staged spending ceiling.

## Approaches considered

### Selected: permanent staged full-corpus BM25 basis

This searches the complete available archive immediately, uses real query generation and evidence-bound prediction, fits the existing production UI, and remains honest about the missing dense lists. It creates a useful permanent fallback while full four-list retrieval builds.

### Rejected: small-corpus four-list preview

This would imitate the four-list shape sooner but would search less than one percent of the archive, create a second corpus identity and BM25 index, and risk looking more complete while having materially worse evidence coverage.

### Rejected: reuse the full four-list basis with empty or BM25-derived dense lists

This would corrupt provenance and falsely imply Andrew-shaped four-list retrieval. It is prohibited.

### Rejected: model-only result as the staged archive feature

Model-only prediction is already available as a separately labeled basis, but it does not demonstrate real archive searching or evidence use.

## Non-goals

- Completing the dense index by November 23.
- Calling the staged two-list path Andrew's full method.
- Adding JPG or PNG upload; the production upload contract remains PDF.
- Spending without the configured hard local dollar ceiling.
- Removing or weakening the fictional demo, full four-list basis, or exact-paper basis.
- Showing research diagnostics in customer-facing prediction cards.

## Acceptance checklist

- [ ] The normal PDF upload and existing Predict control can schedule the first prepared passage on the staged basis.
- [ ] The staged trace contains exactly two independent full-corpus BM25 lists and real fused evidence.
- [ ] The customer prediction card contains no staged-basis or implementation label.
- [ ] A second passage is blocked server-side before provider or retrieval work at the initial limit of one.
- [ ] All paid flags default false and fake-provider loopbacks prove two stages without spending.
- [ ] The fictional demo and existing bases have no regression.
- [ ] The full backend/frontend suites, typecheck, production build, Compose validation, and migration round-trip pass.
- [ ] A hard staged-spend ceiling is configured, enforced across durable provider claims, and cannot be bypassed by the browser.
- [ ] The durable dense runner survives the Codex turn, has a single live PID, and publishes measurable progress without claiming completion.
- [ ] Full dense readiness remains unapproved until recall, latency, checksum, count, and storage gates all pass.
