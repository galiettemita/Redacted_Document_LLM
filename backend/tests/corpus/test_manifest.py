from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from redacted_backend.corpus.exclusions import ExclusionManifest
from redacted_backend.corpus.historylab_export import (
    CAPTURE_SELECT_FIELDS,
    ApiCaptureArtifact,
    ApiCaptureManifest,
    CorpusCaptureError,
    CorpusCoverageReport,
)
from redacted_backend.corpus.manifest import CorpusBuildError, build_chunk_file
from redacted_backend.scientific_config import ScientificConfig


def record(doc_id: str, body: str, corpus: str = "cfpf") -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "classification": "UNCLASSIFIED",
        "authored": "1975-01-01",
        "title": "Visible title",
        "body": body,
        "corpus": corpus,
        "source_url": "https://example.test/source",
        "pg_cnt": 1,
        "char_cnt": len(body),
        "word_cnt": len(body.split()),
        "doc_lang": "en",
    }


def exclusions() -> ExclusionManifest:
    return ExclusionManifest(
        version="v2-test",
        known_corpora={"briefing", "cables", "cfpf", "frus"},
        required_corpora={"cables"},
        cia_reading_room_corpus="briefing",
        cia_mapping_evidence=["paper Appendix C", "History Lab collections"],
        cia_coverage_complete=True,
        frus_publication_years={"frus-old": 2018},
        frus_metadata_sha256="a" * 64,
        frus_coverage_complete=True,
        expected_post_cutoff_frus_volumes=10,
    )


def capture_for(
    root: Path,
    rows_by_corpus: dict[str, list[dict[str, object]]],
) -> ApiCaptureManifest:
    artifacts: list[ApiCaptureArtifact] = []
    for corpus, rows in sorted(rows_by_corpus.items()):
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
        artifacts.append(
            ApiCaptureArtifact(
                corpus=corpus,
                path=relative,
                document_count=len(rows),
                size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                first_doc_id=str(rows[0]["doc_id"]) if rows else None,
                last_doc_id=str(rows[-1]["doc_id"]) if rows else None,
            )
        )
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


def coverage_for(
    capture: ApiCaptureManifest,
    policy: ExclusionManifest,
    *,
    ready: bool = True,
) -> CorpusCoverageReport:
    return CorpusCoverageReport(
        capture_manifest_hash=capture.manifest_hash(),
        exclusion_manifest_hash=policy.manifest_hash(),
        frus_manifest_hash="c" * 64,
        ready=ready,
        reasons=[] if ready else ["source_gap"],
        corpus_counts={item.corpus: item.document_count for item in capture.artifacts},
        post_cutoff_frus_volumes=[],
        unmatched_frus_document_count=0,
        unmatched_frus_document_samples=[],
    )


def test_chunk_build_is_byte_deterministic_and_records_provenance(
    tmp_path: Path,
) -> None:
    source = capture_for(
        tmp_path,
        {"cfpf": [record("doc-1", "Alpha beta."), record("doc-2", "Gamma delta.")]},
    )
    policy = exclusions()
    coverage = coverage_for(source, policy)
    first_output = tmp_path / "chunks-1.jsonl"
    second_output = tmp_path / "chunks-2.jsonl"

    first = build_chunk_file(
        tmp_path,
        first_output,
        capture=source,
        coverage=coverage,
        exclusions=policy,
        scientific=ScientificConfig(),
    )
    second = build_chunk_file(
        tmp_path,
        second_output,
        capture=source,
        coverage=coverage,
        exclusions=policy,
        scientific=ScientificConfig(),
    )

    assert first_output.read_bytes() == second_output.read_bytes()
    assert first.manifest_hash() == second.manifest_hash()
    assert first.document_count == 2
    assert first.chunk_count == 2
    assert first.corpus_counts == {"cfpf": 2}
    assert first.source_manifest_hash == source.manifest_hash()
    assert first.coverage_report_hash == coverage.manifest_hash()
    assert first.chunk_words == 300
    assert first.chunk_overlap == 0


def test_chunk_build_excludes_the_whole_verified_cia_reading_room_collection(
    tmp_path: Path,
) -> None:
    source = capture_for(
        tmp_path,
        {"briefing": [record("briefing-1", "Visible words.", "briefing")]},
    )
    policy = exclusions()

    result = build_chunk_file(
        tmp_path,
        tmp_path / "chunks.jsonl",
        capture=source,
        coverage=coverage_for(source, policy),
        exclusions=policy,
        scientific=ScientificConfig(),
    )

    assert result.document_count == 0
    assert result.exclusion_counts == {"cia_reading_room_2019_plus": 1}


def test_chunk_build_rejects_records_outside_the_normalized_api_schema(
    tmp_path: Path,
) -> None:
    invalid = record("doc-1", "Visible words.") | {"release_text": "not public"}
    source = capture_for(tmp_path, {"cfpf": [invalid]})
    policy = exclusions()

    with pytest.raises(CorpusBuildError, match="normalized History Lab schema"):
        build_chunk_file(
            tmp_path,
            tmp_path / "chunks.jsonl",
            capture=source,
            coverage=coverage_for(source, policy),
            exclusions=policy,
            scientific=ScientificConfig(),
        )


def test_chunk_build_refuses_capture_bytes_that_do_not_match_the_manifest(
    tmp_path: Path,
) -> None:
    source = capture_for(tmp_path, {"cfpf": [record("doc-1", "Visible words.")]})
    policy = exclusions()
    path = tmp_path / source.artifact_for("cfpf").path
    path.write_text("tampered\n", encoding="utf-8")
    output = tmp_path / "chunks.jsonl"

    with pytest.raises(CorpusCaptureError, match="size mismatch|checksum mismatch"):
        build_chunk_file(
            tmp_path,
            output,
            capture=source,
            coverage=coverage_for(source, policy),
            exclusions=policy,
            scientific=ScientificConfig(),
        )

    assert not output.exists()


def test_chunk_build_refuses_an_unready_coverage_report(tmp_path: Path) -> None:
    source = capture_for(tmp_path, {"cfpf": [record("doc-1", "Visible words.")]})
    policy = exclusions()

    with pytest.raises(CorpusBuildError, match="coverage report is not ready"):
        build_chunk_file(
            tmp_path,
            tmp_path / "chunks.jsonl",
            capture=source,
            coverage=coverage_for(source, policy, ready=False),
            exclusions=policy,
            scientific=ScientificConfig(),
        )
