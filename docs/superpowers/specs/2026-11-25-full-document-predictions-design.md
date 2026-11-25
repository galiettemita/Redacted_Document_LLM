# Full-document real predictions design

## Intent

Allow one ordinary production action to predict every current redacted passage in an uploaded PDF, using the strongest approved retrieval basis available for that run. Once the certified dense artifacts are ready, new full-document runs use the best-available four-list path shaped after Andrew Tang's method. The fictional demo remains isolated and unchanged.

## User-visible behavior

- Before full four-list readiness, the existing **Predict passage** action continues to run one selected passage through the honest staged BM25 path.
- When `best_available_foiarchive_v1` or `paper_exact_v1` is the strongest ready basis, the same clean workspace offers **Predict all passages** (or **Predict remaining passages** when durable results already exist).
- One click schedules every current-revision passage that does not already have a recoverable job. Results appear progressively and remain linked to their passage.
- Completed predictions are reused. A retry never repeats a completed provider call.
- The customer UI shows the GPT model and predictions, not retrieval, spending, queue, basis, or research diagnostics.

## Scientific boundary

- A full-document run never promotes the staged two-list BM25 basis to the four-list basis.
- The bulk action is available only when the backend reports full four-list or exact-paper readiness.
- Every scheduled job retains its immutable basis and passage revision.
- The current best-available corpus limitations remain unchanged; `paper_exact_v1` remains fail closed until its separate corpus is available.

## Durable execution

The existing document scheduling transaction and single-process/single-thread worker are the queue. The API creates one revision-bound job per missing passage in geometric document order and dispatches every newly queued job. Dramatiq durably holds them and the worker executes them one at a time. Existing queued, running, or completed jobs are recovered instead of duplicated.

The browser subscribes to the existing document event stream with the exact job/basis allow-list returned by scheduling. It preserves completed cards across runs and reports progress without determining scientific eligibility.

## Spending safety

Every real provider basis uses the existing durable cumulative microdollar ledger, including full best-available and future paper-exact calls. A provider request is reserved before dispatch, settled to actual cost after a known response, and retains its maximum reservation after an uncertain outcome.

`REDACTED_PROVIDER_SPEND_CAP_USD` is the general hard ceiling. For backward compatibility, the existing ignored `REDACTED_STAGED_BM25_SPEND_CAP_USD` supplies the effective ceiling when the general setting is absent. The browser cannot raise or bypass it. Exhaustion fails each not-yet-dispatched provider request before network traffic; completed results remain resumable after an operator deliberately raises the local ceiling.

No implementation, test, deployment, readiness probe, or artifact build initiates a provider request. This increment does not raise the current $10.00 local ceiling or enable the still-unready full basis.

## Failure behavior

- Missing full readiness keeps the UI on selected-passage prediction.
- A scheduling response containing another document, unknown passage, stale revision, duplicate passage, or mixed basis is rejected before opening the event stream.
- Passage edits or document deletion keep using the existing cancellation and revision checks.
- Provider-budget exhaustion creates no provider request beyond the ceiling and never changes prior completed jobs.
- Closing the browser does not cancel durable queued jobs; reopening recovers their current state and events.

## Approaches

### Selected: existing durable job queue plus a full-readiness UI switch

This is the smallest production architecture that already has transaction locking, idempotency, revision binding, durable events, retries, and one-at-a-time worker execution. It avoids a second run table whose state could disagree with jobs.

### Rejected: browser schedules the next passage after each completion

This would stop when the tab closes or the connection drops and is not a durable product workflow.

### Rejected: use the staged basis to predict the whole document now

This would spend heavily before the four-list artifacts are ready and would not meet the requested Andrew-shaped method.

## Acceptance criteria

- Full readiness changes the production action to predict every remaining passage with one document scheduling request.
- Staged readiness continues to target exactly the selected passage.
- Bulk scheduling validates document, passage, revision, uniqueness, and one immutable full basis before streaming.
- All real provider bases share the durable hard spending ceiling.
- Existing completed results survive retries and later runs.
- No paid call or approval/configuration increase occurs during this implementation.
- Backend/frontend suites, typecheck, production build, Compose validation, and no-spend deployment checks pass.

