from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import redacted_backend.corpus.best_available as best_available
from redacted_backend.corpus.best_available import (
    BestAvailableCorpusManifest,
    BestAvailableImportError,
    BestAvailableSourceProfile,
    build_best_available_chunks,
    BestAvailableSourceContractError,
    load_best_available_source,
)
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.corpus.snapshot import SnapshotFile
from redacted_backend.scientific_config import ScientificConfig


REVISION = "411cf6f86b38342683e27148be167c6094d1a10c"
SOURCE_SHA256 = "dd73e590134f3a0cd1e22f81ffd3732054fd469b09c56d78cd25e1cc90b5e5e4"
AUDIT_SHA256 = "901a51d5b468e7dad5d59a3614b73d9b8b46f14bd2a22eb23b797a770a9e2817"
LIMITATIONS_SHA256 = "e1148411dbd8e91378d8e6e926d88f68908327ab7e1f1c9f92b389cf5c5db2e5"
EXPECTED_REASONS = [
    "catalog_collection_count_mismatch:briefing",
    "catalog_collection_count_mismatch:cabinet",
    "catalog_collection_count_mismatch:cfpf",
    "catalog_collection_count_mismatch:cia",
    "catalog_collection_count_mismatch:clinton",
    "catalog_collection_count_mismatch:cpdoc",
    "catalog_collection_count_mismatch:frus",
    "catalog_collection_count_mismatch:kissinger",
    "catalog_collection_count_mismatch:nato",
    "catalog_collection_count_mismatch:un",
    "catalog_collection_count_mismatch:worldbank",
    "catalog_document_count_mismatch",
    "required_corpus_missing:cables",
    "unresolved_collection_labels",
]


def _repo_root() -> Path:
    return Path(__file__).parents[3]


def _contract_paths() -> tuple[Path, Path, Path, Path]:
    root = _repo_root()
    corpus = root / "backend/corpus"
    return (
        corpus / "historylab-huggingface-2025-snapshot-lock.json",
        corpus / "historylab-huggingface-2025-catalog.json",
        corpus / "historylab-huggingface-2025-audit.json",
        root / "4-bulletpoint-limitations.md",
    )


def test_frozen_best_available_source_contract_accepts_only_the_audited_snapshot() -> None:
    snapshot_path, catalog_path, audit_path, limitations_path = _contract_paths()

    profile = load_best_available_source(
        snapshot_path=snapshot_path,
        catalog_path=catalog_path,
        audit_path=audit_path,
        limitations_path=limitations_path,
    )

    assert profile.schema_version == "historylab-best-available-source-v1"
    assert profile.retrieval_profile == "best_available_foiarchive_v1"
    assert profile.repo_id == "HistoryLab/foiarchive"
    assert profile.revision == REVISION
    assert profile.source.path == "foiarchive-202409.json.gz"
    assert profile.source.size_bytes == 7_976_348_471
    assert profile.source.sha256 == SOURCE_SHA256
    assert profile.audit_manifest_hash == AUDIT_SHA256
    assert profile.limitations_sha256 == LIMITATIONS_SHA256
    assert profile.catalog_document_count == 4_949_871
    assert profile.parsed_row_count == 4_889_089
    assert profile.unresolved_collection_count == 4_889_089
    assert profile.unresolved_value_counts == {
        "blank": 0,
        "http_url": 1_304_229,
        "null": 3_584_860,
        "unknown": 0,
    }
    assert profile.canonical_collection_counts == {}
    assert profile.audit_reasons == EXPECTED_REASONS
    assert profile.corpus_label == "historylab-unlabeled"
    assert profile.paper_coverage_ready is False


@pytest.mark.parametrize(
    "artifact",
    ["snapshot", "catalog", "audit", "limitations"],
)
def test_best_available_source_rejects_any_changed_frozen_artifact(
    tmp_path: Path,
    artifact: str,
) -> None:
    snapshot_path, catalog_path, audit_path, limitations_path = _contract_paths()
    paths = {
        "snapshot": tmp_path / snapshot_path.name,
        "catalog": tmp_path / catalog_path.name,
        "audit": tmp_path / audit_path.name,
        "limitations": tmp_path / limitations_path.name,
    }
    paths["snapshot"].write_bytes(snapshot_path.read_bytes())
    paths["catalog"].write_bytes(catalog_path.read_bytes())
    paths["audit"].write_bytes(audit_path.read_bytes())
    paths["limitations"].write_bytes(limitations_path.read_bytes())

    if artifact == "limitations":
        paths[artifact].write_text(
            paths[artifact].read_text(encoding="utf-8") + "\nExtra claim.\n",
            encoding="utf-8",
        )
    else:
        payload = json.loads(paths[artifact].read_text(encoding="utf-8"))
        if artifact == "snapshot":
            payload["revision"] = "0" * 40
        elif artifact == "catalog":
            payload["release_url"] = "https://example.test/changed"
        else:
            payload["unresolved_value_counts"]["http_url"] -= 1
            payload["unresolved_value_counts"]["null"] += 1
        paths[artifact].write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(BestAvailableSourceContractError, match="frozen"):
        load_best_available_source(
            snapshot_path=paths["snapshot"],
            catalog_path=paths["catalog"],
            audit_path=paths["audit"],
            limitations_path=paths["limitations"],
        )


def test_best_available_source_rejects_a_changed_reason_set(tmp_path: Path) -> None:
    snapshot_path, catalog_path, audit_path, limitations_path = _contract_paths()
    changed_audit = tmp_path / "audit.json"
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["reasons"] = payload["reasons"][:-1]
    changed_audit.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BestAvailableSourceContractError, match="frozen"):
        load_best_available_source(
            snapshot_path=snapshot_path,
            catalog_path=catalog_path,
            audit_path=changed_audit,
            limitations_path=limitations_path,
        )


def test_best_available_manifest_can_never_claim_paper_coverage() -> None:
    values = {
        "schema_version": "historylab-best-available-v1",
        "retrieval_profile": "best_available_foiarchive_v1",
        "source_profile_hash": "1" * 64,
        "source_sha256": SOURCE_SHA256,
        "source_size_bytes": 7_976_348_471,
        "audit_manifest_hash": AUDIT_SHA256,
        "limitations_sha256": LIMITATIONS_SHA256,
        "scientific_hash": ScientificConfig().manifest_hash(),
        "chunker_version": "paragraph-sentence-word-v1",
        "chunk_words": 300,
        "chunk_overlap": 0,
        "parsed_row_count": 4_889_089,
        "included_row_count": 4_000_000,
        "empty_body_count": 889_089,
        "malformed_row_count": 0,
        "document_count": 4_000_000,
        "chunk_count": 5_000_000,
        "corpus_counts": {"historylab-unlabeled": 4_000_000},
        "paper_coverage_ready": False,
        "chunks_sha256": "2" * 64,
    }

    manifest = BestAvailableCorpusManifest.model_validate(values)

    assert manifest.paper_coverage_ready is False
    assert manifest.document_count == manifest.included_row_count
    assert manifest.manifest_hash() == manifest.model_copy().manifest_hash()
    with pytest.raises(ValidationError):
        BestAvailableCorpusManifest.model_validate(
            values | {"paper_coverage_ready": True}
        )


def test_best_available_manifest_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValidationError, match="included row count"):
        BestAvailableCorpusManifest(
            source_profile_hash="1" * 64,
            source_sha256=SOURCE_SHA256,
            source_size_bytes=7_976_348_471,
            audit_manifest_hash=AUDIT_SHA256,
            limitations_sha256=LIMITATIONS_SHA256,
            scientific_hash=ScientificConfig().manifest_hash(),
            chunker_version="paragraph-sentence-word-v1",
            chunk_words=300,
            chunk_overlap=0,
            parsed_row_count=10,
            included_row_count=8,
            empty_body_count=2,
            malformed_row_count=0,
            document_count=7,
            chunk_count=9,
            corpus_counts={"historylab-unlabeled": 7},
            chunks_sha256="2" * 64,
        )


def _row(
    doc_id: str,
    body: str | None,
    *,
    source: str | None = "https://example.test/document.pdf",
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "classification": "SECRET",
        "authored": "1974-01-02T00:00:00",
        "title": f"Title {doc_id}",
        "body": body,
        "source": source,
        "pg_cnt": 2,
        "char_cnt": len(body) if body is not None else 0,
        "word_cnt": len(body.split()) if body is not None else 0,
    }


def _write_source(
    path: Path,
    rows: list[dict[str, object] | bytes],
) -> None:
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as destination:
        for row in rows:
            serialized = row if isinstance(row, bytes) else json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            destination.write(serialized + b"\n")


def _profile(source: Path, parsed_rows: int) -> BestAvailableSourceProfile:
    return BestAvailableSourceProfile(
        source=SnapshotFile(
            path=source.name,
            size_bytes=source.stat().st_size,
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ),
        snapshot_manifest_hash="1" * 64,
        catalog_manifest_hash="2" * 64,
        audit_manifest_hash="3" * 64,
        limitations_sha256="4" * 64,
        catalog_document_count=parsed_rows,
        parsed_row_count=parsed_rows,
        unresolved_collection_count=parsed_rows,
        unresolved_value_counts={
            "blank": 0,
            "http_url": parsed_rows,
            "null": 0,
            "unknown": 0,
        },
        canonical_collection_counts={},
        audit_reasons=EXPECTED_REASONS,
    )


def _build(
    root: Path,
    source: Path,
    *,
    profile: BestAvailableSourceProfile | None = None,
    checkpoint_interval_rows: int = 2,
) -> tuple[BestAvailableCorpusManifest, Path, Path, Path]:
    chunks = root / "chunks.jsonl"
    manifest = root / "manifest.json"
    checkpoint = root / "checkpoint.json"
    result = build_best_available_chunks(
        source,
        chunks,
        manifest,
        checkpoint,
        profile=profile or _profile(source, 1),
        scientific=ScientificConfig(),
        checkpoint_interval_rows=checkpoint_interval_rows,
    )
    return result, chunks, manifest, checkpoint


def test_best_available_import_is_deterministic_and_preserves_real_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json.gz"
    long_body = " ".join(f"word-{index}" for index in range(305))
    _write_source(
        source,
        [
            _row("doc-1", long_body),
            _row("doc-empty", "  \n"),
        ],
    )
    profile = _profile(source, 2)

    first, first_chunks, first_manifest, first_checkpoint = _build(
        tmp_path / "first",
        source,
        profile=profile,
    )
    second, second_chunks, _, _ = _build(
        tmp_path / "second",
        source,
        profile=profile,
    )

    assert first == second
    assert first_chunks.read_bytes() == second_chunks.read_bytes()
    assert BestAvailableCorpusManifest.model_validate_json(
        first_manifest.read_text(encoding="utf-8")
    ) == first
    assert first.parsed_row_count == 2
    assert first.included_row_count == 1
    assert first.empty_body_count == 1
    assert first.malformed_row_count == 0
    assert first.document_count == 1
    assert first.chunk_count == 2
    assert first.corpus_counts == {"historylab-unlabeled": 1}
    assert first.paper_coverage_ready is False
    chunks = [
        CorpusChunk.model_validate_json(line)
        for line in first_chunks.read_text(encoding="utf-8").splitlines()
    ]
    assert [chunk.word_count for chunk in chunks] == [300, 5]
    assert {chunk.document_id for chunk in chunks} == {"doc-1"}
    assert {chunk.corpus for chunk in chunks} == {"historylab-unlabeled"}
    assert {chunk.source_url for chunk in chunks} == {
        "https://example.test/document.pdf"
    }
    assert {chunk.classification for chunk in chunks} == {"SECRET"}
    assert {chunk.authored for chunk in chunks} == {"1974-01-02T00:00:00"}
    assert {chunk.title for chunk in chunks} == {"Title doc-1"}
    assert {chunk.doc_lang for chunk in chunks} == {None}
    assert not first_checkpoint.exists()
    assert not (first_checkpoint.parent / "checkpoint.ids.sqlite3").exists()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_row("doc-1", "Visible text."), b"{not-json}"], "malformed row"),
        (
            [_row("duplicate", "First body."), _row("duplicate", "Second body.")],
            "duplicate document ID",
        ),
    ],
)
def test_best_available_import_fails_closed_on_bad_rows_without_a_manifest(
    tmp_path: Path,
    rows: list[dict[str, object] | bytes],
    message: str,
) -> None:
    source = tmp_path / "source.json.gz"
    _write_source(source, rows)

    with pytest.raises(BestAvailableImportError, match=message):
        _build(tmp_path / "build", source, profile=_profile(source, 2))

    assert not (tmp_path / "build/manifest.json").exists()


def test_best_available_import_rejects_changed_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.json.gz"
    _write_source(source, [_row("doc-1", "Visible text.")])
    profile = _profile(source, 1)
    with source.open("ab") as destination:
        destination.write(b"tamper")

    with pytest.raises(BestAvailableImportError, match="source (size|checksum)"):
        _build(tmp_path / "build", source, profile=profile)

    assert not (tmp_path / "build/manifest.json").exists()


def test_best_available_import_resumes_exactly_and_truncates_uncommitted_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json.gz"
    _write_source(
        source,
        [
            _row("doc-1", "First visible body."),
            _row("doc-2", "Second visible body."),
            _row("doc-3", "Third visible body."),
        ],
    )
    profile = _profile(source, 3)
    interrupted_root = tmp_path / "interrupted"
    original_write = best_available._write_checkpoint_atomic
    checkpoint_writes = 0

    def interrupt_after_second_checkpoint(path: Path, value: object) -> None:
        nonlocal checkpoint_writes
        original_write(path, value)
        checkpoint_writes += 1
        if checkpoint_writes == 2:
            raise OSError("simulated process interruption")

    monkeypatch.setattr(
        best_available,
        "_write_checkpoint_atomic",
        interrupt_after_second_checkpoint,
    )
    with pytest.raises(OSError, match="simulated process interruption"):
        _build(
            interrupted_root,
            source,
            profile=profile,
            checkpoint_interval_rows=1,
        )
    assert not (interrupted_root / "manifest.json").exists()
    checkpoint = interrupted_root / "checkpoint.json"
    chunks = interrupted_root / "chunks.jsonl"
    committed = json.loads(checkpoint.read_text(encoding="utf-8"))["output_bytes"]
    with chunks.open("ab") as destination:
        destination.write(b"uncommitted garbage\n")
    assert chunks.stat().st_size > committed

    monkeypatch.setattr(best_available, "_write_checkpoint_atomic", original_write)
    resumed, resumed_chunks, _, resumed_checkpoint = _build(
        interrupted_root,
        source,
        profile=profile,
        checkpoint_interval_rows=1,
    )
    clean, clean_chunks, _, _ = _build(
        tmp_path / "clean",
        source,
        profile=profile,
        checkpoint_interval_rows=1,
    )

    assert resumed == clean
    assert resumed_chunks.read_bytes() == clean_chunks.read_bytes()
    assert not resumed_checkpoint.exists()
    assert not (interrupted_root / "checkpoint.ids.sqlite3").exists()


def test_best_available_import_rejects_a_corrupt_committed_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json.gz"
    _write_source(
        source,
        [_row("doc-1", "First body."), _row("doc-2", "Second body.")],
    )
    profile = _profile(source, 2)
    root = tmp_path / "build"
    original_write = best_available._write_checkpoint_atomic

    def interrupt(path: Path, value: object) -> None:
        original_write(path, value)
        raise OSError("stop")

    monkeypatch.setattr(best_available, "_write_checkpoint_atomic", interrupt)
    with pytest.raises(OSError, match="stop"):
        _build(root, source, profile=profile, checkpoint_interval_rows=1)
    chunks = root / "chunks.jsonl"
    payload = bytearray(chunks.read_bytes())
    payload[0] = ord("X")
    chunks.write_bytes(payload)
    monkeypatch.setattr(best_available, "_write_checkpoint_atomic", original_write)

    with pytest.raises(BestAvailableImportError, match="checkpoint checksum"):
        _build(root, source, profile=profile, checkpoint_interval_rows=1)

    assert not (root / "manifest.json").exists()


def test_best_available_import_preserves_checkpoint_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json.gz"
    _write_source(
        source,
        [_row("doc-1", "First body."), _row("doc-2", "Second body.")],
    )
    profile = _profile(source, 2)
    root = tmp_path / "build"
    original_append = best_available._append_chunk
    writes = 0

    def fail_second_write(destination: object, serialized: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("no space left on device")
        original_append(destination, serialized)

    monkeypatch.setattr(best_available, "_append_chunk", fail_second_write)

    with pytest.raises(BestAvailableImportError, match="write failed"):
        _build(root, source, profile=profile, checkpoint_interval_rows=1)

    checkpoint = json.loads(
        (root / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["source_row_count"] == 1
    assert not (root / "manifest.json").exists()
