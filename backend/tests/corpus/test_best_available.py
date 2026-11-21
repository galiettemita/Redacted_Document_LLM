from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from redacted_backend.corpus.best_available import (
    BestAvailableCorpusManifest,
    BestAvailableSourceContractError,
    load_best_available_source,
)
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
