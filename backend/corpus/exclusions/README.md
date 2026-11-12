# Corpus exclusions

`v1.yaml` is deliberately not corpus-ready yet. Inspection of the authenticated
Hugging Face export showed that its `source` values are origin URLs or null, not
the collection labels described by its card. The complete History Lab API has a
separate `corpus` field and an additional `cables` collection, so only a frozen,
verified API capture can become the retrieval source.

The CIA mapping is now fixed to `briefing`. Appendix C says CREST remains 57%
of archive text while the removed whole bucket contains the CIA's 2019+ reading-
room release. History Lab describes `briefing` as the CIA Current/Central
Intelligence Bulletin releases and the separate `cia` collection as CREST. The
mapping and both citations are hashed into the policy; it was selected before
retrieval and without inspecting a target result.

Before a real corpus build, finish the remaining public metadata work:

- Official FRUS volume IDs and publication years from the Office of the
  Historian metadata feed, excluding publications from 2019 through 2024.

Set each `*_coverage_complete` flag only after the full capture passes its
coverage report. The report requires non-empty `cables`, no unknown collections
or FRUS IDs, and exactly the ten 2019-2024 FRUS volumes reported in the paper.
A different count is a source-version gap, not permission to alter the rule.
Unknown identifiers fail closed. Never choose exclusions after inspecting
retrieval results for a target.

