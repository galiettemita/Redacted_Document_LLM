# Corpus exclusions

`v1.yaml` is deliberately not corpus-ready yet. The nine public FOIArchive
fields identify a document and its source collection, but they do not include
the CIA release bucket or the FRUS volume publication year needed to reproduce
the paper's date exclusions.

Before a real corpus build, generate and review two identifier maps from public
source metadata:

- CIA document IDs in the reading-room bucket carrying 2019-and-later releases,
  plus the CIA IDs confirmed to predate that cutoff.
- FRUS document IDs mapped to Office of the Historian volume publication years,
  excluding publications from 2019 through 2024.

Set each `*_coverage_complete` flag only after every corresponding FOIArchive
record is covered. Unknown identifiers fail closed and keep corpus readiness
false. Never choose exclusions after inspecting retrieval results for a target.

