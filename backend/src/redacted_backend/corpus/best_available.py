from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from redacted_backend.corpus.bulk_export import BulkExportAudit, BulkExportCatalog
from redacted_backend.corpus.snapshot import SnapshotFile, SnapshotLock


_REVISION = "411cf6f86b38342683e27148be167c6094d1a10c"
_SOURCE_SHA256 = "dd73e590134f3a0cd1e22f81ffd3732054fd469b09c56d78cd25e1cc90b5e5e4"
_SOURCE_SIZE_BYTES = 7_976_348_471
_SNAPSHOT_MANIFEST_HASH = (
    "1c7b055a7079fe93e563ee11d764194ffa2a402b3fc8b0f72cb212204f44c3be"
)
_CATALOG_MANIFEST_HASH = (
    "823b9b74c0fd52f44a1dddf5d35eb64f4ff8341cf097bbef54697e8914563518"
)
_AUDIT_MANIFEST_HASH = (
    "901a51d5b468e7dad5d59a3614b73d9b8b46f14bd2a22eb23b797a770a9e2817"
)
_LIMITATIONS_SHA256 = (
    "e1148411dbd8e91378d8e6e926d88f68908327ab7e1f1c9f92b389cf5c5db2e5"
)
_CORPUS_LABEL = "historylab-unlabeled"
_EXPECTED_REASONS = (
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
)


class BestAvailableSourceContractError(ValueError):
    pass


class BestAvailableSourceProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["historylab-best-available-source-v1"] = (
        "historylab-best-available-source-v1"
    )
    retrieval_profile: Literal["best_available_foiarchive_v1"] = (
        "best_available_foiarchive_v1"
    )
    repo_id: Literal["HistoryLab/foiarchive"] = "HistoryLab/foiarchive"
    revision: Literal[_REVISION] = _REVISION
    source: SnapshotFile
    snapshot_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    limitations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_document_count: int = Field(ge=0)
    parsed_row_count: int = Field(ge=0)
    unresolved_collection_count: int = Field(ge=0)
    unresolved_value_counts: dict[str, int]
    canonical_collection_counts: dict[str, int]
    audit_reasons: list[str]
    corpus_label: Literal["historylab-unlabeled"] = _CORPUS_LABEL
    paper_coverage_ready: Literal[False] = False

    def manifest_hash(self) -> str:
        return _canonical_hash(self.model_dump(mode="json"))


class BestAvailableCorpusManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["historylab-best-available-v1"] = (
        "historylab-best-available-v1"
    )
    retrieval_profile: Literal["best_available_foiarchive_v1"] = (
        "best_available_foiarchive_v1"
    )
    source_profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: int = Field(ge=0)
    audit_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    limitations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunker_version: str = Field(min_length=1)
    chunk_words: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)
    parsed_row_count: int = Field(ge=0)
    included_row_count: int = Field(ge=0)
    empty_body_count: int = Field(ge=0)
    malformed_row_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    corpus_counts: dict[str, int]
    paper_coverage_ready: Literal[False] = False
    chunks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_counts(self) -> BestAvailableCorpusManifest:
        if self.document_count != self.included_row_count:
            raise ValueError("document count must equal included row count")
        if (
            self.included_row_count
            + self.empty_body_count
            + self.malformed_row_count
            != self.parsed_row_count
        ):
            raise ValueError(
                "included row count, empty body count, and malformed row count "
                "must equal parsed row count"
            )
        if self.malformed_row_count != 0:
            raise ValueError("a published manifest cannot contain malformed rows")
        if self.corpus_counts != {_CORPUS_LABEL: self.document_count}:
            raise ValueError("corpus counts must use the unlabeled History Lab identity")
        if self.document_count and self.chunk_count < self.document_count:
            raise ValueError("every included document must produce at least one chunk")
        return self

    def manifest_hash(self) -> str:
        return _canonical_hash(self.model_dump(mode="json"))


def load_best_available_source(
    *,
    snapshot_path: Path,
    catalog_path: Path,
    audit_path: Path,
    limitations_path: Path,
) -> BestAvailableSourceProfile:
    try:
        snapshot = SnapshotLock.model_validate_json(
            snapshot_path.read_text(encoding="utf-8")
        )
        catalog = BulkExportCatalog.model_validate_json(
            catalog_path.read_text(encoding="utf-8")
        )
        audit = BulkExportAudit.model_validate_json(
            audit_path.read_text(encoding="utf-8")
        )
        limitations_sha256 = _sha256_file(limitations_path)
    except (OSError, ValidationError, ValueError) as error:
        raise BestAvailableSourceContractError(
            "frozen best-available source artifact is invalid"
        ) from error

    snapshot_hash = snapshot.manifest_hash()
    catalog_hash = catalog.manifest_hash()
    audit_hash = audit.manifest_hash()
    if (
        snapshot_hash != _SNAPSHOT_MANIFEST_HASH
        or catalog_hash != _CATALOG_MANIFEST_HASH
        or audit_hash != _AUDIT_MANIFEST_HASH
        or limitations_sha256 != _LIMITATIONS_SHA256
    ):
        raise BestAvailableSourceContractError(
            "frozen best-available source artifact does not match its approved profile"
        )
    if len(snapshot.files) != 1:
        raise BestAvailableSourceContractError(
            "frozen best-available snapshot must contain one source file"
        )
    source = snapshot.files[0]
    if (
        snapshot.repo_id != "HistoryLab/foiarchive"
        or snapshot.revision != _REVISION
        or source.path != "foiarchive-202409.json.gz"
        or source.size_bytes != _SOURCE_SIZE_BYTES
        or source.sha256 != _SOURCE_SHA256
        or catalog.source != source
        or audit.source_path != source.path
        or audit.source_size_bytes != source.size_bytes
        or audit.source_sha256 != source.sha256
        or audit.snapshot_manifest_hash != snapshot_hash
        or audit.catalog_manifest_hash != catalog_hash
        or audit.document_count != 4_889_089
        or audit.unresolved_collection_count != 4_889_089
        or audit.unresolved_value_counts
        != {"blank": 0, "http_url": 1_304_229, "null": 3_584_860, "unknown": 0}
        or audit.corpus_counts != {}
        or tuple(audit.reasons) != _EXPECTED_REASONS
        or audit.ready_for_import
    ):
        raise BestAvailableSourceContractError(
            "frozen best-available source metadata is inconsistent"
        )

    return BestAvailableSourceProfile(
        source=source,
        snapshot_manifest_hash=snapshot_hash,
        catalog_manifest_hash=catalog_hash,
        audit_manifest_hash=audit_hash,
        limitations_sha256=limitations_sha256,
        catalog_document_count=catalog.document_count,
        parsed_row_count=audit.document_count,
        unresolved_collection_count=audit.unresolved_collection_count,
        unresolved_value_counts=audit.unresolved_value_counts,
        canonical_collection_counts=audit.corpus_counts,
        audit_reasons=audit.reasons,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()
