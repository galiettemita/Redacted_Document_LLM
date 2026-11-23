# Durable Prediction Scheduling and Recovery Design

## Intent

Make the real uploaded-document **Predict passage** action reliably display an accepted real prediction, including after a lost scheduling response or browser refresh, without creating a second paid provider call. Keep the fictional demonstration unchanged and keep implementation, research-method, cost, query, and evidence diagnostics out of the customer interface.

## Observed failure

The browser currently schedules every prepared passage independently. Under the one-prediction document limit, the backend truthfully accepts the first passage and blocks the remaining passages. The browser then treats any blocked response as failure for the entire batch, discards the accepted job ID, and never opens the event stream. The accepted worker job can complete and persist a prediction that the interface cannot recover.

## Approaches considered

### Frontend-only response partitioning

The browser could keep accepted responses and ignore blocked responses. This is the smallest patch, but it would continue creating durable blocked jobs for passages the server already knows are ineligible, leave passage selection duplicated across browser and server, and retain race windows across tabs and retries.

### Duplicate the staged eligibility rules in the browser

The browser could schedule only the first visual passage. This would avoid the immediate mixed response, but client and server ordering or limit rules could drift. It would also fail to support later limit increases safely.

### Chosen: document-level server scheduling and recovery

Add a document-level scheduling endpoint that runs under the existing analysis write/document lock. It validates all current prepared passage revisions, reuses current recoverable jobs, selects and schedules only passages allowed by the server's existing readiness and document-limit rules, and returns accepted jobs separately from deferred passage IDs. Add a read-only document analysis-state endpoint that returns only queued, running, or completed jobs tied to current live passage revisions.

The browser calls the document endpoint once, opens the existing filtered SSE stream for every accepted job even when other passages are deferred, and checks analysis state after automatic context preparation. A completed event is replayed from durable storage; no generation is repeated.

## Backend contract

`POST /v1/documents/{document_id}/analysis` requires an ASCII `Idempotency-Key` and returns:

- `document_id`
- `accepted_jobs`: current recoverable jobs plus newly accepted jobs
- `deferred_passage_ids`: current prepared passages not accepted in this request
- `blockers`: readiness blockers only when no prediction can be accepted

The endpoint derives a stable passage-scoped key from the document request key and passage ID. A replay returns the same job. Before creating anything, it includes queued, running, and completed jobs for current passage revisions, so an intentional second click cannot replace an existing paid result.

`GET /v1/documents/{document_id}/analysis` returns the same current recoverable job set without scheduling, dispatching, retrieval, or model traffic. Jobs for deleted passages, old revisions, blocked attempts, failures, and cancellations are excluded.

The existing passage-level endpoint remains available for compatibility, but the production browser no longer fans out through it.

## Browser behavior

After detection and visible-context preparation succeed, the workspace asks for current analysis state. When recoverable jobs exist, it records their IDs and immutable bases, starts the existing SSE reader, and replays only events belonging to those jobs. A terminal stored prediction therefore appears after recovery without another model call.

On a new click, accepted jobs always start streaming even if other passages are deferred. A generic unavailable message appears only when the response has no accepted jobs. The customer control is the singular **Predict passage** while the production document limit is one.

Correction, removal, adding a passage, and document replacement are disabled while scheduling or accepted work is active. Backend passage-mutation endpoints also reject conflicting changes while a queued or running analysis exists, closing refresh and multi-client race windows. Once the job is terminal, normal corrections are available again.

## Idempotency and failure handling

- A double click is suppressed by the disabled button and protected by the same in-memory request key.
- An interrupted scheduling response is retried with the same key.
- A refresh recovers the server job and durable event instead of scheduling.
- A completed current-revision job is reused even if a later request supplies a new key.
- A queued or running current-revision job is reused and streamed.
- The event stream continues to filter by accepted job IDs and immutable prediction bases.
- No recovery or verification step is allowed to enable provider spending or issue a new provider request.

## Verification

Backend tests cover one accepted plus one deferred passage, all unavailable, same-key replay, completed-job recovery, no new dispatch during recovery, and mutation rejection only while work is active. Frontend tests cover mixed accepted/deferred streaming, all-deferred messaging, refresh recovery, duplicate-click suppression, singular copy, and mutation/navigation locking during generation.

Live verification records the provider-claim count and spend ledger before and after deployment, recovers the existing completed prediction through the state and SSE endpoints, and requires those provider counters to remain unchanged.
