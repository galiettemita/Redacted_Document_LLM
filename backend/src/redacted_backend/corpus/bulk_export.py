from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from redacted_backend.corpus.exclusions import ExclusionManifest
from redacted_backend.corpus.historylab_export import CatalogEntry
from redacted_backend.corpus.snapshot import (
    SnapshotChecksumError,
    SnapshotFile,
    SnapshotLock,
    verify_snapshot_inputs,
)

CATALOG_SCHEMA_VERSION = "historylab-huggingface-catalog-v1"
AUDIT_SCHEMA_VERSION = "historylab-huggingface-audit-v2"
_UNRESOLVED_SAMPLE_LIMIT = 20
_UNRESOLVED_CATEGORIES = ("blank", "http_url", "null", "unknown")


class BulkExportAuditError(ValueError):
    pass


class BulkExportCatalog(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["historylab-huggingface-catalog-v1"]
    repo_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    release_url: str
    source: SnapshotFile
    document_count: int = Field(ge=0)
    collections: list[CatalogEntry]

    @field_validator("release_url")
    @classmethod
    def require_https_release_url(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("release URL must use HTTPS")
        return value

    @model_validator(mode="after")
    def validate_collection_totals(self) -> BulkExportCatalog:
        names = [entry.corpus for entry in self.collections]
        if len(names) != len(set(names)):
            raise ValueError("catalog collections must be unique")
        if sum(entry.doc_cnt for entry in self.collections) != self.document_count:
            raise ValueError("catalog document count must equal collection counts")
        return self

    @property
    def collection_counts(self) -> dict[str, int]:
        return {
            entry.corpus: entry.doc_cnt
            for entry in sorted(self.collections, key=lambda item: item.corpus)
        }

    def manifest_hash(self) -> str:
        return _manifest_hash(self.model_dump(mode="json"))


class BulkExportAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["historylab-huggingface-audit-v2"]
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_path: str
    source_size_bytes: int = Field(ge=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    exclusion_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(ge=0)
    catalog_document_count: int = Field(ge=0)
    corpus_counts: dict[str, int]
    catalog_corpus_counts: dict[str, int]
    unresolved_collection_count: int = Field(ge=0)
    unresolved_value_counts: dict[str, int]
    unresolved_collection_samples: list[str] = Field(
        max_length=_UNRESOLVED_SAMPLE_LIMIT
    )
    ready_for_import: bool
    reasons: list[str]

    @model_validator(mode="after")
    def validate_unresolved_counts(self) -> BulkExportAudit:
        if set(self.unresolved_value_counts) != set(_UNRESOLVED_CATEGORIES):
            raise ValueError("unresolved value counts must contain every category")
        if any(count < 0 for count in self.unresolved_value_counts.values()):
            raise ValueError("unresolved value counts cannot be negative")
        if sum(self.unresolved_value_counts.values()) != self.unresolved_collection_count:
            raise ValueError(
                "unresolved value counts must equal unresolved collection count"
            )
        return self

    def manifest_hash(self) -> str:
        return _manifest_hash(self.model_dump(mode="json"))


class _BulkExportRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(min_length=1)
    classification: str | None
    authored: str | None
    title: str | None
    body: str | None
    source: str | None
    pg_cnt: int | None = Field(ge=0)
    char_cnt: int | None = Field(ge=0)
    word_cnt: int | None = Field(ge=0)
    corpus: str | None = None


def audit_historylab_bulk_export(
    source_path: Path,
    *,
    snapshot: SnapshotLock,
    catalog: BulkExportCatalog,
    exclusions: ExclusionManifest,
) -> BulkExportAudit:
    _validate_provenance(source_path, snapshot, catalog, exclusions)

    counts: Counter[str] = Counter()
    unresolved_count = 0
    unresolved_samples: set[str] = set()
    unresolved_value_counts: Counter[str] = Counter(
        {category: 0 for category in _UNRESOLVED_CATEGORIES}
    )
    document_count = 0

    try:
        with gzip.open(source_path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = _BulkExportRow.model_validate_json(line)
                except ValidationError as exc:
                    raise BulkExportAuditError(
                        f"invalid row at line {line_number}: {exc.errors()[0]['type']}"
                    ) from None
                corpus, unresolved_sample, unresolved_category = _resolve_corpus(
                    row,
                    exclusions.known_corpora,
                    line_number=line_number,
                )
                document_count += 1
                if corpus is None:
                    unresolved_count += 1
                    if unresolved_category is None:
                        raise AssertionError("unresolved values require a category")
                    unresolved_value_counts[unresolved_category] += 1
                    if len(unresolved_samples) < _UNRESOLVED_SAMPLE_LIMIT:
                        unresolved_samples.add(unresolved_sample)
                else:
                    counts[corpus] += 1
    except BulkExportAuditError:
        raise
    except (gzip.BadGzipFile, EOFError, OSError, UnicodeError) as exc:
        raise BulkExportAuditError(
            f"gzip stream could not be audited: {type(exc).__name__}"
        ) from None

    corpus_counts = dict(sorted(counts.items()))
    catalog_counts = catalog.collection_counts
    reasons: list[str] = []
    if unresolved_count:
        reasons.append("unresolved_collection_labels")
    if document_count != catalog.document_count:
        reasons.append("catalog_document_count_mismatch")
    for corpus in sorted(set(corpus_counts) | set(catalog_counts)):
        if corpus_counts.get(corpus, 0) != catalog_counts.get(corpus, 0):
            reasons.append(f"catalog_collection_count_mismatch:{corpus}")
    for corpus in sorted(exclusions.required_corpora):
        if corpus_counts.get(corpus, 0) < 1:
            reasons.append(f"required_corpus_missing:{corpus}")
    reasons = sorted(set(reasons))

    return BulkExportAudit(
        schema_version=AUDIT_SCHEMA_VERSION,
        repo_id=snapshot.repo_id,
        revision=snapshot.revision,
        source_path=catalog.source.path,
        source_size_bytes=catalog.source.size_bytes,
        source_sha256=catalog.source.sha256,
        snapshot_manifest_hash=snapshot.manifest_hash(),
        catalog_manifest_hash=catalog.manifest_hash(),
        exclusion_manifest_hash=exclusions.manifest_hash(),
        document_count=document_count,
        catalog_document_count=catalog.document_count,
        corpus_counts=corpus_counts,
        catalog_corpus_counts=catalog_counts,
        unresolved_collection_count=unresolved_count,
        unresolved_value_counts=dict(sorted(unresolved_value_counts.items())),
        unresolved_collection_samples=sorted(unresolved_samples)[
            :_UNRESOLVED_SAMPLE_LIMIT
        ],
        ready_for_import=not reasons,
        reasons=reasons,
    )


def _validate_provenance(
    source_path: Path,
    snapshot: SnapshotLock,
    catalog: BulkExportCatalog,
    exclusions: ExclusionManifest,
) -> None:
    if len(snapshot.files) != 1:
        raise BulkExportAuditError("bulk export snapshot must contain exactly one file")
    if catalog.repo_id != snapshot.repo_id:
        raise BulkExportAuditError("catalog repository does not match snapshot")
    if catalog.revision != snapshot.revision:
        raise BulkExportAuditError("catalog revision does not match snapshot")
    if catalog.source != snapshot.files[0]:
        raise BulkExportAuditError("catalog source does not match snapshot")
    if not catalog.source.path.lower().endswith((".json.gz", ".jsonl.gz")):
        raise BulkExportAuditError("bulk export source must be JSON Lines gzip")
    unknown_catalog_corpora = set(catalog.collection_counts) - exclusions.known_corpora
    if unknown_catalog_corpora:
        raise BulkExportAuditError(
            "catalog contains unknown corpora: "
            + ", ".join(sorted(unknown_catalog_corpora))
        )
    try:
        verify_snapshot_inputs(snapshot, [source_path])
    except SnapshotChecksumError as exc:
        raise BulkExportAuditError(str(exc)) from None


def _resolve_corpus(
    row: _BulkExportRow,
    known_corpora: set[str],
    *,
    line_number: int,
) -> tuple[str | None, str, str | None]:
    canonical_source = _canonical_label(row.source, known_corpora)
    if row.corpus is not None:
        canonical_corpus = _canonical_label(row.corpus, known_corpora)
        if (
            canonical_corpus is not None
            and canonical_source is not None
            and canonical_corpus != canonical_source
        ):
            raise BulkExportAuditError(
                f"conflicting canonical labels at line {line_number}"
            )
        if canonical_corpus is not None:
            return canonical_corpus, "", None
        sample, category = _describe_unresolved(row.corpus)
        return None, sample, category
    if canonical_source is not None:
        return canonical_source, "", None
    sample, category = _describe_unresolved(row.source)
    return None, sample, category


def _canonical_label(value: str | None, known_corpora: set[str]) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized if normalized in known_corpora else None


def _describe_unresolved(value: str | None) -> tuple[str, str]:
    if value is None:
        return "null", "null"
    normalized = value.strip()
    if not normalized:
        return "blank", "blank"
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        return normalized, "http_url"
    return normalized, "unknown"


def _manifest_hash(payload: object) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
