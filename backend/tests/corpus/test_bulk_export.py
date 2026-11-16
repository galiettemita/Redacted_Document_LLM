from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from redacted_backend.corpus.bulk_export import (
    BulkExportAuditError,
    BulkExportCatalog,
    audit_historylab_bulk_export,
)
from redacted_backend.corpus.exclusions import ExclusionManifest
from redacted_backend.corpus.historylab_export import CatalogEntry
from redacted_backend.corpus.snapshot import SnapshotFile, SnapshotLock

REVISION = "411cf6f86b38342683e27148be167c6094d1a10c"
RELEASE_URL = "https://lab.history.columbia.edu/news/foiarchive-hugging-face"


def _row(
    *,
    doc_id: str = "doc-1",
    source: str | None = "cia",
    corpus: str | None = None,
    include_corpus: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "doc_id": doc_id,
        "classification": "SECRET",
        "authored": "1974-01-01",
        "title": "A title",
        "body": "Document body",
        "source": source,
        "pg_cnt": 1,
        "char_cnt": 13,
        "word_cnt": 2,
    }
    if include_corpus:
        row["corpus"] = corpus
    return row


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, sort_keys=True))
            target.write("\n")


def _snapshot_file(path: Path) -> SnapshotFile:
    return SnapshotFile(
        path=path.name,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _snapshot(item: SnapshotFile) -> SnapshotLock:
    return SnapshotLock(
        repo_id="HistoryLab/foiarchive",
        revision=REVISION,
        license="cc-by-nc-4.0",
        files=[item],
    )


def _catalog(
    item: SnapshotFile,
    counts: dict[str, int],
) -> BulkExportCatalog:
    return BulkExportCatalog(
        schema_version="historylab-huggingface-catalog-v1",
        repo_id="HistoryLab/foiarchive",
        revision=REVISION,
        release_url=RELEASE_URL,
        source=item,
        document_count=sum(counts.values()),
        collections=[
            CatalogEntry(corpus=corpus, doc_cnt=count)
            for corpus, count in sorted(counts.items())
        ],
    )


def _exclusions() -> ExclusionManifest:
    known = {
        "briefing",
        "cabinet",
        "cables",
        "cfpf",
        "cia",
        "clinton",
        "cpdoc",
        "frus",
        "kissinger",
        "nato",
        "un",
        "worldbank",
    }
    return ExclusionManifest(
        version="test-v1",
        known_corpora=known,
        required_corpora={"cables"},
        cia_reading_room_corpus="briefing",
        cia_mapping_evidence=["test evidence"],
        cia_coverage_complete=True,
        frus_publication_years={},
        frus_metadata_sha256=None,
        frus_coverage_complete=False,
        expected_post_cutoff_frus_volumes=10,
        expected_archive_document_count=5_550_000,
        archive_document_count_tolerance=5_000,
    )


def test_audit_counts_canonical_source_without_expanding_gzip(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiny.json.gz"
    _write_rows(
        source,
        [
            _row(doc_id="cia-1", source="cia"),
            _row(doc_id="cable-1", source="cables"),
        ],
    )
    item = _snapshot_file(source)
    names_before = sorted(path.name for path in tmp_path.iterdir())

    audit = audit_historylab_bulk_export(
        source,
        snapshot=_snapshot(item),
        catalog=_catalog(item, {"cables": 1, "cia": 1}),
        exclusions=_exclusions(),
    )

    assert audit.document_count == 2
    assert audit.corpus_counts == {"cables": 1, "cia": 1}
    assert audit.unresolved_collection_count == 0
    assert audit.ready_for_import is True
    assert audit.reasons == []
    assert sorted(path.name for path in tmp_path.iterdir()) == names_before
    assert audit.manifest_hash() == audit.model_copy().manifest_hash()


def test_explicit_corpus_takes_precedence_over_noncanonical_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "explicit.jsonl.gz"
    _write_rows(
        source,
        [
            _row(
                source="https://history.state.gov/example",
                corpus="cables",
                include_corpus=True,
            )
        ],
    )
    item = _snapshot_file(source)

    audit = audit_historylab_bulk_export(
        source,
        snapshot=_snapshot(item),
        catalog=_catalog(item, {"cables": 1}),
        exclusions=_exclusions(),
    )

    assert audit.corpus_counts == {"cables": 1}
    assert audit.unresolved_collection_count == 0
    assert audit.ready_for_import is True


def test_conflicting_canonical_corpus_and_source_are_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "conflict.json.gz"
    _write_rows(
        source,
        [_row(source="cia", corpus="cables", include_corpus=True)],
    )
    item = _snapshot_file(source)

    with pytest.raises(BulkExportAuditError, match="conflicting canonical labels"):
        audit_historylab_bulk_export(
            source,
            snapshot=_snapshot(item),
            catalog=_catalog(item, {"cables": 1}),
            exclusions=_exclusions(),
        )


def test_unresolved_sources_are_counted_capped_sorted_and_unready(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unresolved.json.gz"
    values: list[str | None] = [None, "https://history.state.gov"] + [
        f"unknown-{index:02d}" for index in range(23)
    ]
    _write_rows(
        source,
        [
            _row(doc_id=f"doc-{index}", source=value)
            for index, value in enumerate(values)
        ],
    )
    item = _snapshot_file(source)

    audit = audit_historylab_bulk_export(
        source,
        snapshot=_snapshot(item),
        catalog=_catalog(item, {}),
        exclusions=_exclusions(),
    )

    assert audit.unresolved_collection_count == 25
    assert len(audit.unresolved_collection_samples) == 20
    assert audit.unresolved_collection_samples == sorted(
        audit.unresolved_collection_samples
    )
    assert audit.ready_for_import is False
    assert "unresolved_collection_labels" in audit.reasons


def test_catalog_mismatches_and_missing_cables_have_literal_reasons(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mismatch.json.gz"
    _write_rows(source, [_row(source="cia")])
    item = _snapshot_file(source)

    audit = audit_historylab_bulk_export(
        source,
        snapshot=_snapshot(item),
        catalog=_catalog(item, {"cia": 2}),
        exclusions=_exclusions(),
    )

    assert audit.ready_for_import is False
    assert audit.reasons == [
        "catalog_collection_count_mismatch:cia",
        "catalog_document_count_mismatch",
        "required_corpus_missing:cables",
    ]


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        (b"{not-json}\n", "invalid row"),
        (
            json.dumps(
                {
                    key: value
                    for key, value in _row().items()
                    if key != "word_cnt"
                }
            ).encode("utf-8")
            + b"\n",
            "invalid row",
        ),
    ],
)
def test_malformed_or_incomplete_rows_are_rejected(
    tmp_path: Path,
    contents: bytes,
    error: str,
) -> None:
    source = tmp_path / "invalid.json.gz"
    with gzip.open(source, "wb") as target:
        target.write(contents)
    item = _snapshot_file(source)

    with pytest.raises(BulkExportAuditError, match=error):
        audit_historylab_bulk_export(
            source,
            snapshot=_snapshot(item),
            catalog=_catalog(item, {}),
            exclusions=_exclusions(),
        )


def test_gzip_corruption_is_rejected_after_snapshot_verification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corrupt.json.gz"
    source.write_bytes(b"not a gzip stream")
    item = _snapshot_file(source)

    with pytest.raises(BulkExportAuditError, match="gzip"):
        audit_historylab_bulk_export(
            source,
            snapshot=_snapshot(item),
            catalog=_catalog(item, {}),
            exclusions=_exclusions(),
        )


@pytest.mark.parametrize(("field", "error"), [("size", "size"), ("sha", "checksum")])
def test_snapshot_size_and_checksum_mismatches_are_rejected(
    tmp_path: Path,
    field: str,
    error: str,
) -> None:
    source = tmp_path / "locked.json.gz"
    _write_rows(source, [_row(source="cables")])
    good = _snapshot_file(source)
    locked = SnapshotFile(
        path=good.path,
        size_bytes=good.size_bytes + (1 if field == "size" else 0),
        sha256="0" * 64 if field == "sha" else good.sha256,
    )

    with pytest.raises(BulkExportAuditError, match=error):
        audit_historylab_bulk_export(
            source,
            snapshot=_snapshot(locked),
            catalog=_catalog(locked, {"cables": 1}),
            exclusions=_exclusions(),
        )


def test_frozen_official_catalog_has_exact_documented_counts() -> None:
    path = Path(__file__).parents[2] / "corpus/historylab-huggingface-2025-catalog.json"
    catalog = BulkExportCatalog.model_validate_json(path.read_text(encoding="utf-8"))

    assert catalog.repo_id == "HistoryLab/foiarchive"
    assert catalog.revision == REVISION
    assert catalog.release_url == RELEASE_URL
    assert catalog.source.path == "foiarchive-202409.json.gz"
    assert catalog.source.size_bytes == 7_976_348_471
    assert catalog.source.sha256 == (
        "dd73e590134f3a0cd1e22f81ffd3732054fd469b09c56d78cd25e1cc90b5e5e4"
    )
    assert catalog.collection_counts == {
        "briefing": 9_680,
        "cabinet": 42_539,
        "cfpf": 3_214_293,
        "cia": 935_716,
        "clinton": 54_149,
        "cpdoc": 10_279,
        "frus": 311_866,
        "kissinger": 4_552,
        "nato": 46_002,
        "un": 192_541,
        "worldbank": 128_254,
    }
    assert catalog.document_count == 4_949_871
