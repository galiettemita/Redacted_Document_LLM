from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from redacted_backend.corpus.bulk_export import BulkExportAudit, BulkExportCatalog
from redacted_backend.corpus.chunker import chunk_document
from redacted_backend.corpus.historylab_export import NormalizedCorpusDocument
from redacted_backend.corpus.snapshot import SnapshotFile, SnapshotLock
from redacted_backend.scientific_config import ScientificConfig


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
BEST_AVAILABLE_BM25_INDEX_NAME = "foiarchive-best-available-bm25-v1"
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


class BestAvailableImportError(ValueError):
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


class _BestAvailableRow(BaseModel):
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


class _ChunkCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["historylab-best-available-checkpoint-v1"] = (
        "historylab-best-available-checkpoint-v1"
    )
    source_profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_name: str = Field(min_length=1)
    source_row_count: int = Field(ge=0)
    included_row_count: int = Field(ge=0)
    empty_body_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    output_bytes: int = Field(ge=0)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_counts(self) -> _ChunkCheckpoint:
        if self.included_row_count + self.empty_body_count != self.source_row_count:
            raise ValueError("checkpoint row counts are inconsistent")
        if self.included_row_count and self.chunk_count < self.included_row_count:
            raise ValueError("checkpoint chunk count is inconsistent")
        return self


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


def build_best_available_chunks(
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    *,
    profile: BestAvailableSourceProfile,
    scientific: ScientificConfig | None = None,
    checkpoint_interval_rows: int = 10_000,
) -> BestAvailableCorpusManifest:
    if checkpoint_interval_rows < 1:
        raise ValueError("checkpoint interval must be positive")
    config = scientific or ScientificConfig()
    source_profile_hash = profile.manifest_hash()
    scientific_hash = config.manifest_hash()
    if source_path.name != profile.source.path:
        raise BestAvailableImportError("source filename does not match its profile")
    if not source_path.is_file():
        raise BestAvailableImportError("source file is missing")
    if source_path.stat().st_size != profile.source.size_bytes:
        raise BestAvailableImportError("source size does not match its profile")
    if _sha256_file(source_path) != profile.source.sha256:
        raise BestAvailableImportError("source checksum does not match its profile")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    identifiers_path = checkpoint_path.with_suffix(".ids.sqlite3")

    existing = _completed_manifest(
        manifest_path,
        output_path,
        profile=profile,
        scientific=config,
    )
    if existing is not None:
        return existing

    checkpoint = _load_checkpoint(
        checkpoint_path,
        output_path=output_path,
        source_profile_hash=source_profile_hash,
        scientific_hash=scientific_hash,
    )
    if checkpoint is None:
        output_path.unlink(missing_ok=True)
        identifiers_path.unlink(missing_ok=True)
        output_path.touch()
        output_digest = hashlib.sha256()
        source_row_count = 0
        included_row_count = 0
        empty_body_count = 0
        chunk_count = 0
    else:
        output_digest = _restore_output(output_path, checkpoint)
        source_row_count = checkpoint.source_row_count
        included_row_count = checkpoint.included_row_count
        empty_body_count = checkpoint.empty_body_count
        chunk_count = checkpoint.chunk_count

    connection = _open_identifier_database(
        identifiers_path,
        checkpoint=checkpoint,
    )
    try:
        with output_path.open("ab") as destination:
            try:
                with gzip.open(source_path, "rt", encoding="utf-8") as source:
                    seen_rows = 0
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        seen_rows += 1
                        if seen_rows <= source_row_count:
                            continue
                        try:
                            row = _BestAvailableRow.model_validate_json(line)
                        except ValidationError as error:
                            raise BestAvailableImportError(
                                f"malformed row at line {line_number}"
                            ) from error
                        try:
                            connection.execute(
                                "INSERT INTO document_ids (doc_id, source_row_number) "
                                "VALUES (?, ?)",
                                (row.doc_id, seen_rows),
                            )
                        except sqlite3.IntegrityError as error:
                            raise BestAvailableImportError(
                                f"duplicate document ID at line {line_number}"
                            ) from error

                        source_row_count = seen_rows
                        body = row.body or ""
                        if not body.strip():
                            empty_body_count += 1
                        else:
                            document = NormalizedCorpusDocument(
                                doc_id=row.doc_id,
                                corpus=profile.corpus_label,
                                classification=row.classification,
                                authored=row.authored,
                                title=row.title,
                                body=body,
                                source_url=row.source,
                                pg_cnt=row.pg_cnt,
                                char_cnt=row.char_cnt,
                                word_cnt=row.word_cnt,
                                doc_lang=None,
                            )
                            chunks = chunk_document(
                                document,
                                max_words=config.chunk_words,
                            )
                            included_row_count += 1
                            for chunk in chunks:
                                serialized = (
                                    json.dumps(
                                        chunk.model_dump(mode="json"),
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    ).encode("utf-8")
                                    + b"\n"
                                )
                                try:
                                    _append_chunk(destination, serialized)
                                except OSError as error:
                                    raise BestAvailableImportError(
                                        "chunk output write failed"
                                    ) from error
                                output_digest.update(serialized)
                                chunk_count += 1

                        if source_row_count % checkpoint_interval_rows == 0:
                            _commit_checkpoint(
                                connection,
                                destination,
                                checkpoint_path,
                                source_profile_hash=source_profile_hash,
                                scientific_hash=scientific_hash,
                                output_name=output_path.name,
                                source_row_count=source_row_count,
                                included_row_count=included_row_count,
                                empty_body_count=empty_body_count,
                                chunk_count=chunk_count,
                                output_digest=output_digest,
                            )
            except BestAvailableImportError:
                raise
            except (gzip.BadGzipFile, EOFError, UnicodeError) as error:
                raise BestAvailableImportError(
                    "source gzip stream could not be read"
                ) from error

            if source_row_count != profile.parsed_row_count:
                raise BestAvailableImportError(
                    "source row count does not match the frozen audit"
                )
            _commit_checkpoint(
                connection,
                destination,
                checkpoint_path,
                source_profile_hash=source_profile_hash,
                scientific_hash=scientific_hash,
                output_name=output_path.name,
                source_row_count=source_row_count,
                included_row_count=included_row_count,
                empty_body_count=empty_body_count,
                chunk_count=chunk_count,
                output_digest=output_digest,
            )

        manifest = BestAvailableCorpusManifest(
            source_profile_hash=source_profile_hash,
            source_sha256=profile.source.sha256,
            source_size_bytes=profile.source.size_bytes,
            audit_manifest_hash=profile.audit_manifest_hash,
            limitations_sha256=profile.limitations_sha256,
            scientific_hash=scientific_hash,
            chunker_version=config.corpus_chunker_version,
            chunk_words=config.chunk_words,
            chunk_overlap=config.chunk_overlap,
            parsed_row_count=source_row_count,
            included_row_count=included_row_count,
            empty_body_count=empty_body_count,
            malformed_row_count=0,
            document_count=included_row_count,
            chunk_count=chunk_count,
            corpus_counts={profile.corpus_label: included_row_count},
            chunks_sha256=output_digest.hexdigest(),
        )
        _write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    finally:
        connection.close()

    checkpoint_path.unlink(missing_ok=True)
    identifiers_path.unlink(missing_ok=True)
    return manifest


def _completed_manifest(
    manifest_path: Path,
    output_path: Path,
    *,
    profile: BestAvailableSourceProfile,
    scientific: ScientificConfig,
) -> BestAvailableCorpusManifest | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = BestAvailableCorpusManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise BestAvailableImportError("published manifest is invalid") from error
    if (
        manifest.source_profile_hash != profile.manifest_hash()
        or manifest.scientific_hash != scientific.manifest_hash()
    ):
        raise BestAvailableImportError("published manifest is incompatible")
    if not output_path.is_file() or _sha256_file(output_path) != manifest.chunks_sha256:
        raise BestAvailableImportError("published chunk checksum does not match")
    return manifest


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    output_path: Path,
    source_profile_hash: str,
    scientific_hash: str,
) -> _ChunkCheckpoint | None:
    if not checkpoint_path.exists():
        return None
    try:
        checkpoint = _ChunkCheckpoint.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise BestAvailableImportError("chunk checkpoint is invalid") from error
    if (
        checkpoint.source_profile_hash != source_profile_hash
        or checkpoint.scientific_hash != scientific_hash
        or checkpoint.output_name != output_path.name
    ):
        raise BestAvailableImportError("chunk checkpoint is incompatible")
    return checkpoint


def _restore_output(
    output_path: Path,
    checkpoint: _ChunkCheckpoint,
) -> hashlib._Hash:
    if not output_path.is_file() or output_path.stat().st_size < checkpoint.output_bytes:
        raise BestAvailableImportError("checkpoint output is missing or truncated")
    digest = hashlib.sha256()
    remaining = checkpoint.output_bytes
    with output_path.open("rb") as source:
        while remaining:
            block = source.read(min(1024 * 1024, remaining))
            if not block:
                raise BestAvailableImportError("checkpoint output is truncated")
            digest.update(block)
            remaining -= len(block)
    if digest.hexdigest() != checkpoint.output_sha256:
        raise BestAvailableImportError("checkpoint checksum does not match output")
    with output_path.open("r+b") as destination:
        destination.truncate(checkpoint.output_bytes)
        destination.flush()
        os.fsync(destination.fileno())
    return digest


def _open_identifier_database(
    path: Path,
    *,
    checkpoint: _ChunkCheckpoint | None,
) -> sqlite3.Connection:
    if checkpoint is not None and not path.is_file():
        raise BestAvailableImportError("checkpoint document identity index is missing")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS document_ids ("
        "doc_id TEXT PRIMARY KEY, source_row_number INTEGER NOT NULL UNIQUE)"
    )
    if checkpoint is not None:
        connection.execute(
            "DELETE FROM document_ids WHERE source_row_number > ?",
            (checkpoint.source_row_count,),
        )
        connection.commit()
        count = int(
            connection.execute("SELECT count(*) FROM document_ids").fetchone()[0]
        )
        if count != checkpoint.source_row_count:
            connection.close()
            raise BestAvailableImportError(
                "checkpoint document identity index is inconsistent"
            )
    return connection


def _commit_checkpoint(
    connection: sqlite3.Connection,
    destination: BinaryIO,
    checkpoint_path: Path,
    *,
    source_profile_hash: str,
    scientific_hash: str,
    output_name: str,
    source_row_count: int,
    included_row_count: int,
    empty_body_count: int,
    chunk_count: int,
    output_digest: hashlib._Hash,
) -> None:
    destination.flush()
    os.fsync(destination.fileno())
    connection.commit()
    checkpoint = _ChunkCheckpoint(
        source_profile_hash=source_profile_hash,
        scientific_hash=scientific_hash,
        output_name=output_name,
        source_row_count=source_row_count,
        included_row_count=included_row_count,
        empty_body_count=empty_body_count,
        chunk_count=chunk_count,
        output_bytes=destination.tell(),
        output_sha256=output_digest.hexdigest(),
    )
    _write_checkpoint_atomic(
        checkpoint_path,
        checkpoint.model_dump(mode="json"),
    )


def _append_chunk(destination: BinaryIO, serialized: bytes) -> None:
    written = destination.write(serialized)
    if written != len(serialized):
        raise OSError("short chunk write")


def _write_checkpoint_atomic(path: Path, value: object) -> None:
    _write_json_atomic(path, value)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            json.dump(value, destination, sort_keys=True, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
