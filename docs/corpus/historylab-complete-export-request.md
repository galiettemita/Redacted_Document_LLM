# History Lab complete export request — NOT SENT

Status: **NOT SENT.** This is a prepared request for the exact source needed to reproduce the retrieval corpus in Tang et al. It is not evidence that History Lab has received or agreed to it.

## Draft request

Subject: Request for a paper-compatible, collection-labeled FOIArchive export

Hello History Lab team,

We are reproducing the public-document retrieval corpus used by Tang et al. for research on predicting text hidden by redaction. Could you provide either an immutable bulk export or a complete enumerable API snapshot corresponding to the paper-era corpus?

The source needs these properties so we can reproduce the paper without inferring labels from document content or search results:

- one explicit `corpus` value per record, limited to `briefing`, `cabinet`, `cables`, `cfpf`, `cia`, `clinton`, `cpdoc`, `frus`, `kissinger`, `nato`, `un`, or `worldbank`;
- a non-empty `cables` collection;
- complete enumeration rather than ranked or capped search;
- the complete document text and, at minimum, `doc_id`, `corpus`, `classification`, `authored`, `title`, `body`, origin `source`, `pg_cnt`, `char_cnt`, and `word_cnt`;
- stable document identifiers and deterministic ordering;
- exact per-collection counts and a pre-exclusion total of 5,550,000 documents within the paper-derived ±5,000 rounding tolerance;
- an immutable source revision, byte size, SHA-256 checksum, license, and schema description; and
- FRUS volume identity and publication metadata sufficient to identify exactly the ten 2019–2024 volumes excluded by the paper.

The collection label must be separate from the origin URL. We will not derive it from document IDs, URLs, text, retrieval results, or other target-dependent evidence.

For context, we authenticated and audited the official Hugging Face artifact at revision `411cf6f86b38342683e27148be167c6094d1a10c`:

- file: `foiarchive-202409.json.gz`;
- size: 7,976,348,471 bytes;
- SHA-256: `dd73e590134f3a0cd1e22f81ffd3732054fd469b09c56d78cd25e1cc90b5e5e4`;
- dataset-card catalog total: 4,949,871 documents across 11 collections;
- streamed row count: 4,889,089, a difference of 60,782 from the card total;
- usable canonical collection labels: zero;
- unresolved collection labels: 4,889,089 total—1,304,229 HTTP(S) origin URLs and 3,584,860 nulls, with zero blank or unknown values;
- `cables`: absent; and
- audit result: not ready for import.

The current hosted search/MCP interface is useful for discovery but cannot replace a complete enumeration for local BM25 and dense index construction. A repaired enumerable PostgREST endpoint would also meet the transport requirement if it exposes explicit collection labels and stable exact counts.

Thank you for any guidance on obtaining the paper-compatible research export.

## Local evidence

The source gzip remains outside Git in the authenticated Hugging Face cache. The reproducible audit uses:

- frozen catalog: `backend/corpus/historylab-huggingface-2025-catalog.json`;
- snapshot lock: `~/.cache/redacted-document-llm/corpus-audit/historylab-411cf6f-snapshot-lock.json`;
- atomic audit report: `~/.cache/redacted-document-llm/corpus-audit/historylab-411cf6f-audit.json`; and
- exclusion contract: `backend/corpus/exclusions/v1.yaml`.

No email, form submission, issue, or other external message was sent as part of this work.
