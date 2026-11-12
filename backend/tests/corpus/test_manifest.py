from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from redacted_backend.corpus.exclusions import ExclusionManifest
from redacted_backend.corpus.manifest import CorpusBuildError, build_chunk_file
from redacted_backend.corpus.snapshot import (
    SnapshotChecksumError,
    SnapshotFile,
    SnapshotLock,
)
from redacted_backend.scientific_config import ScientificConfig

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def record(doc_id: str, body: str, source: str = "cfpf") -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "classification": "UNCLASSIFIED",
        "authored": "Visible author",
        "title": "Visible title",
        "body": body,
        "source": source,
        "pg_cnt": 1,
        "char_cnt": len(body),
        "word_cnt": len(body.split()),
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def manifest() -> ExclusionManifest:
    return ExclusionManifest(
        version="v1-test",
        included_sources={"cfpf", "cia", "frus"},
        cia_reading_room_excluded_doc_ids=set(),
        cia_confirmed_pre_cutoff_doc_ids={"cia-known"},
        frus_publication_years={"frus-known": 2018},
        cia_coverage_complete=False,
        frus_coverage_complete=False,
    )


def snapshot(path: Path) -> SnapshotLock:
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return SnapshotLock(
        repo_id="HistoryLab/foiarchive",
        revision=COMMIT,
        license="cc-by-nc-4.0",
        files=[
            SnapshotFile(
                path=path.name,
                size_bytes=path.stat().st_size,
                sha256=checksum,
            )
        ],
    )


def test_chunk_build_is_byte_deterministic_and_records_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    first_output = tmp_path / "chunks-1.jsonl"
    second_output = tmp_path / "chunks-2.jsonl"
    write_jsonl(
        source, [record("doc-1", "Alpha beta."), record("doc-2", "Gamma delta.")]
    )

    first = build_chunk_file(
        [source],
        first_output,
        snapshot=snapshot(source),
        exclusions=manifest(),
        scientific=ScientificConfig(),
    )
    second = build_chunk_file(
        [source],
        second_output,
        snapshot=snapshot(source),
        exclusions=manifest(),
        scientific=ScientificConfig(),
    )

    assert first_output.read_bytes() == second_output.read_bytes()
    assert first.manifest_hash() == second.manifest_hash()
    assert first.document_count == 2
    assert first.chunk_count == 2
    assert first.source_counts == {"cfpf": 2}
    assert first.chunk_words == 300
    assert first.chunk_overlap == 0
    assert len(first.chunks_sha256) == 64


def test_chunk_build_reports_unknown_provenance_without_silently_including_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "chunks.jsonl"
    write_jsonl(source, [record("cia-unmapped", "Visible words.", source="cia")])

    result = build_chunk_file(
        [source],
        output,
        snapshot=snapshot(source),
        exclusions=manifest(),
        scientific=ScientificConfig(),
    )

    assert result.coverage_ready is False
    assert result.document_count == 0
    assert result.exclusion_counts == {"unknown_release_provenance": 1}
    assert output.read_text(encoding="utf-8") == ""


def test_chunk_build_rejects_records_that_do_not_match_the_nine_public_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "chunks.jsonl"
    invalid = record("doc-1", "Visible words.") | {"release_text": "not a public field"}
    write_jsonl(source, [invalid])

    with pytest.raises(CorpusBuildError, match="nine-field schema"):
        build_chunk_file(
            [source],
            output,
            snapshot=snapshot(source),
            exclusions=manifest(),
            scientific=ScientificConfig(),
        )


def test_chunk_build_refuses_input_bytes_that_do_not_match_the_snapshot(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "chunks.jsonl"
    write_jsonl(source, [record("doc-1", "Visible words.")])
    incorrect = SnapshotLock(
        repo_id="HistoryLab/foiarchive",
        revision=COMMIT,
        license="cc-by-nc-4.0",
        files=[
            SnapshotFile(
                path=source.name,
                size_bytes=source.stat().st_size,
                sha256="0" * 64,
            )
        ],
    )

    with pytest.raises(SnapshotChecksumError, match="source.jsonl"):
        build_chunk_file(
            [source],
            output,
            snapshot=incorrect,
            exclusions=manifest(),
            scientific=ScientificConfig(),
        )

    assert not output.exists()


def test_chunk_build_streams_the_corpus_json_lines_from_gzip(tmp_path: Path) -> None:
    source = tmp_path / "foiarchive.json.gz"
    output = tmp_path / "chunks.jsonl"
    encoded = f"{json.dumps(record('doc-gzip', 'Visible compressed words.'), sort_keys=True)}\n"
    with gzip.open(source, "wt", encoding="utf-8") as destination:
        destination.write(encoded)

    result = build_chunk_file(
        [source],
        output,
        snapshot=snapshot(source),
        exclusions=manifest(),
        scientific=ScientificConfig(),
    )

    assert result.document_count == 1
    assert result.chunk_count == 1
    assert json.loads(output.read_text(encoding="utf-8"))["document_id"] == "doc-gzip"
