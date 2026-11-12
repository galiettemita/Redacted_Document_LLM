# Corpus exclusions

`v1.yaml` is deliberately not corpus-ready yet. Inspection of the authenticated
Hugging Face export showed that its `source` values are origin URLs or null, not
the collection labels described by its card. The complete History Lab API has a
separate `corpus` field and an additional `cables` collection, so only a frozen,
verified API capture can become the retrieval source.

Before a real corpus build, generate and review two identifier maps from public
source metadata:

- The exact API collection corresponding to the paper's whole CIA reading-room
  bucket. Do not infer this from a collection name or document count.
- Official FRUS volume IDs and publication years from the Office of the
  Historian metadata feed, excluding publications from 2019 through 2024.

Set each `*_coverage_complete` flag only after the full capture passes its
coverage report. The report requires non-empty `cables`, no unknown collections
or FRUS IDs, and exactly the ten 2019-2024 FRUS volumes reported in the paper.
A different count is a source-version gap, not permission to alter the rule.
Unknown identifiers fail closed. Never choose exclusions after inspecting
retrieval results for a target.

