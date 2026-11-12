from __future__ import annotations

import hashlib
import json
from pathlib import Path

from redacted_backend.corpus.exclusions import ExclusionManifest
from redacted_backend.corpus.frus_metadata import FrusPublicationManifest
from redacted_backend.corpus.historylab_export import (
    CAPTURE_SELECT_FIELDS,
    ApiCaptureArtifact,
    ApiCaptureManifest,
    validate_capture_for_paper,
)


def document(doc_id: str, corpus: str) -> dict[str, object]:
    body = f"Body for {doc_id}"
    return {
        "authored": "1975-01-01",
        "body": body,
        "char_cnt": len(body),
        "classification": "unclassified",
        "corpus": corpus,
        "doc_id": doc_id,
        "doc_lang": "en",
        "pg_cnt": 1,
        "source_url": None,
        "title": doc_id,
        "word_cnt": len(body.split()),
    }


def write_artifact(
    root: Path,
    corpus: str,
    rows: list[dict[str, object]],
) -> ApiCaptureArtifact:
    relative = f"collections/{corpus}.jsonl"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return ApiCaptureArtifact(
        corpus=corpus,
        path=relative,
        document_count=len(rows),
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        first_doc_id=str(rows[0]["doc_id"]) if rows else None,
        last_doc_id=str(rows[-1]["doc_id"]) if rows else None,
    )


def capture(root: Path, artifacts: list[ApiCaptureArtifact]) -> ApiCaptureManifest:
    counts = {item.corpus: item.document_count for item in artifacts}
    catalog = root / "catalog.json"
    catalog.write_text(
        json.dumps(
            [
                {"corpus": corpus, "doc_cnt": count}
                for corpus, count in sorted(counts.items())
            ],
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return ApiCaptureManifest(
        schema_version="historylab-postgrest-v1",
        base_url="https://api.example.test",
        captured_at="2026-11-12T12:00:00Z",
        select_fields=list(CAPTURE_SELECT_FIELDS),
        order="doc_id.asc",
        catalog_path="catalog.json",
        catalog_size_bytes=catalog.stat().st_size,
        catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        counts_before=counts,
        counts_after=counts,
        artifacts=artifacts,
        complete=True,
    )


def official_metadata(years: dict[str, int]) -> FrusPublicationManifest:
    return FrusPublicationManifest(
        source_url="https://history.state.gov/open/frus-metadata.xml",
        retrieved_at="2026-11-12T12:00:00Z",
        source_sha256="a" * 64,
        publication_years=years,
    )


def exclusions(years: dict[str, int]) -> ExclusionManifest:
    return ExclusionManifest(
        version="v2-test",
        known_corpora={"briefing", "cables", "cfpf", "frus"},
        required_corpora={"cables"},
        cia_reading_room_corpus="briefing",
        cia_coverage_complete=True,
        frus_publication_years=years,
        frus_metadata_sha256="a" * 64,
        frus_coverage_complete=True,
        expected_post_cutoff_frus_volumes=10,
    )


def ten_post_cutoff_volumes() -> dict[str, int]:
    return {f"frus-volume-{index:02d}": 2019 + index % 6 for index in range(10)} | {
        "frus-old": 2018
    }


def valid_capture(root: Path, years: dict[str, int]) -> ApiCaptureManifest:
    artifacts = [
        write_artifact(root, "briefing", [document("briefing-1", "briefing")]),
        write_artifact(root, "cables", [document("cable-1", "cables")]),
        write_artifact(root, "cfpf", [document("cfpf-1", "cfpf")]),
        write_artifact(
            root,
            "frus",
            [document(f"{volume_id}d1", "frus") for volume_id in years],
        ),
    ]
    return capture(root, artifacts)


def test_paper_coverage_requires_cables_and_exactly_ten_post_cutoff_frus_volumes(
    tmp_path: Path,
) -> None:
    years = ten_post_cutoff_volumes()

    report = validate_capture_for_paper(
        valid_capture(tmp_path, years),
        tmp_path,
        exclusions(years),
        official_metadata(years),
    )

    assert report.ready is True
    assert report.reasons == []
    assert len(report.post_cutoff_frus_volumes) == 10
    assert report.unmatched_frus_document_count == 0


def test_paper_coverage_fails_when_cables_are_absent(tmp_path: Path) -> None:
    years = ten_post_cutoff_volumes()
    full = valid_capture(tmp_path, years)
    without_cables = capture(
        tmp_path,
        [item for item in full.artifacts if item.corpus != "cables"],
    )

    report = validate_capture_for_paper(
        without_cables,
        tmp_path,
        exclusions(years),
        official_metadata(years),
    )

    assert report.ready is False
    assert "required_corpus_missing:cables" in report.reasons


def test_paper_coverage_fails_on_unknown_collections(tmp_path: Path) -> None:
    years = ten_post_cutoff_volumes()
    full = valid_capture(tmp_path, years)
    mystery = write_artifact(tmp_path, "mystery", [document("mystery-1", "mystery")])
    with_unknown = capture(tmp_path, [*full.artifacts, mystery])

    report = validate_capture_for_paper(
        with_unknown,
        tmp_path,
        exclusions(years),
        official_metadata(years),
    )

    assert report.ready is False
    assert "unknown_corpus:mystery" in report.reasons


def test_paper_coverage_reports_source_version_gap_at_nine_frus_volumes(
    tmp_path: Path,
) -> None:
    years = ten_post_cutoff_volumes()
    missing_volume = next(
        volume_id for volume_id, year in sorted(years.items()) if 2019 <= year <= 2024
    )
    present_years = {
        key: value for key, value in years.items() if key != missing_volume
    }

    report = validate_capture_for_paper(
        valid_capture(tmp_path, present_years),
        tmp_path,
        exclusions(years),
        official_metadata(years),
    )

    assert report.ready is False
    assert "post_cutoff_frus_volume_count:9:expected:10" in report.reasons


def test_paper_coverage_fails_closed_on_unmatched_frus_ids(tmp_path: Path) -> None:
    years = ten_post_cutoff_volumes()
    full = valid_capture(tmp_path, years)
    unmatched = write_artifact(
        tmp_path,
        "frus",
        [document("frus-does-not-exist-d1", "frus")],
    )
    replaced = capture(
        tmp_path,
        [item for item in full.artifacts if item.corpus != "frus"] + [unmatched],
    )

    report = validate_capture_for_paper(
        replaced,
        tmp_path,
        exclusions(years),
        official_metadata(years),
    )

    assert report.ready is False
    assert report.unmatched_frus_document_count == 1
    assert "unmatched_frus_document_ids:1" in report.reasons
