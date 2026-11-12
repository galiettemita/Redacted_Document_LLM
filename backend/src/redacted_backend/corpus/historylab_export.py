from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

DEFAULT_HISTORYLAB_API_URL = "https://api.foiarchive.org"
CAPTURE_SCHEMA_VERSION = "historylab-postgrest-v1"
CAPTURE_SELECT_FIELDS = (
    "doc_id",
    "corpus",
    "classification",
    "authored",
    "title",
    "body",
    "source",
    "pg_cnt",
    "char_cnt",
    "word_cnt",
    "doc_lang",
)
_CORPUS_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class CorpusCaptureError(ValueError):
    pass


class CatalogEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    corpus: str
    doc_cnt: int = Field(ge=0)
    title: str | None = None

    @field_validator("corpus")
    @classmethod
    def safe_corpus(cls, value: str) -> str:
        return _validate_corpus(value)


class NormalizedCorpusDocument(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )

    doc_id: str = Field(min_length=1)
    corpus: str
    classification: str | None
    authored: str | None
    title: str | None
    body: str
    source_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("source", "source_url"),
    )
    pg_cnt: int | None = Field(default=None, ge=0)
    char_cnt: int | None = Field(default=None, ge=0)
    word_cnt: int | None = Field(default=None, ge=0)
    doc_lang: str | None = None

    @field_validator("doc_id")
    @classmethod
    def normalize_doc_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("document ID cannot be blank")
        return normalized

    @field_validator("corpus")
    @classmethod
    def safe_corpus(cls, value: str) -> str:
        return _validate_corpus(value)

    @field_validator("body", mode="before")
    @classmethod
    def normalize_null_body(cls, value: object) -> str:
        return "" if value is None else str(value)


class ApiCaptureArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus: str
    path: str
    document_count: int = Field(ge=0)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    first_doc_id: str | None
    last_doc_id: str | None

    @field_validator("corpus")
    @classmethod
    def safe_corpus(cls, value: str) -> str:
        return _validate_corpus(value)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError("capture artifact paths must be safe relative paths")
        return path.as_posix()


class ApiCaptureManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    base_url: str
    captured_at: str
    select_fields: list[str]
    order: str
    catalog_path: str
    catalog_size_bytes: int = Field(ge=0)
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    counts_before: dict[str, int]
    counts_after: dict[str, int]
    artifacts: list[ApiCaptureArtifact]
    complete: bool

    @field_validator("catalog_path")
    @classmethod
    def safe_catalog_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError("capture catalog path must be a safe relative path")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_collections(self) -> ApiCaptureManifest:
        corpora = [artifact.corpus for artifact in self.artifacts]
        if len(corpora) != len(set(corpora)):
            raise ValueError("capture artifact corpora must be unique")
        expected = set(corpora)
        if set(self.counts_before) != expected or set(self.counts_after) != expected:
            raise ValueError("capture counts must cover every unique artifact corpus")
        return self

    def artifact_for(self, corpus: str) -> ApiCaptureArtifact:
        for artifact in self.artifacts:
            if artifact.corpus == corpus:
                return artifact
        raise KeyError(corpus)

    def manifest_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class CorpusCoverageReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    capture_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    exclusion_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    frus_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ready: bool
    reasons: list[str]
    archive_document_count: int = Field(ge=0)
    corpus_counts: dict[str, int]
    post_cutoff_frus_volumes: list[str]
    unmatched_frus_document_count: int = Field(ge=0)
    unmatched_frus_document_samples: list[str]

    def manifest_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class _Checkpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus: str
    expected_count: int = Field(ge=0)
    committed_bytes: int = Field(ge=0)
    document_count: int = Field(ge=0)
    first_doc_id: str | None
    last_doc_id: str | None


def capture_historylab_corpus(
    base_url: str,
    output_root: Path,
    *,
    client: httpx.Client | None = None,
    page_size: int = 1_000,
    corpora: Sequence[str] | None = None,
    captured_at: str | None = None,
    resume: bool = True,
) -> ApiCaptureManifest:
    if page_size < 1 or page_size > 10_000:
        raise ValueError("page_size must be between 1 and 10000")
    normalized_base_url = base_url.rstrip("/")
    if not normalized_base_url.startswith(
        ("https://", "http://127.0.0.1", "http://localhost")
    ):
        raise ValueError("History Lab API URL must use HTTPS or loopback HTTP")
    requested_corpora = None
    if corpora is not None:
        requested_corpora = [_validate_corpus(item) for item in corpora]
        if len(requested_corpora) != len(set(requested_corpora)):
            raise ValueError("requested corpus identifiers must be unique")

    output_root.mkdir(parents=True, exist_ok=True)
    collections_root = output_root / "collections"
    collections_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "capture-manifest.json"
    if manifest_path.exists():
        existing = ApiCaptureManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        expected = set(
            requested_corpora or [item.corpus for item in existing.artifacts]
        )
        if (
            existing.base_url != normalized_base_url
            or {item.corpus for item in existing.artifacts} != expected
        ):
            raise CorpusCaptureError(
                "A complete capture already exists with a different source or corpus set"
            )
        verify_api_capture(existing, output_root)
        return existing

    owns_client = client is None
    http_client = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=20.0))
    marker_path = output_root / "capture.incomplete.json"
    try:
        catalog = _fetch_catalog(http_client, normalized_base_url)
        catalog_by_name = {item.corpus: item for item in catalog}
        selected_names = requested_corpora or sorted(catalog_by_name)
        missing = sorted(set(selected_names) - set(catalog_by_name))
        if missing:
            raise CorpusCaptureError(
                f"Requested corpora are absent from catalog: {missing}"
            )
        selected_catalog = [catalog_by_name[name] for name in selected_names]
        capture_time = _capture_time(marker_path, captured_at, resume)
        catalog_payload = [item.model_dump(mode="json") for item in selected_catalog]
        catalog_bytes = _canonical_json_bytes(catalog_payload)
        catalog_file_bytes = catalog_bytes + b"\n"
        _write_bytes_atomic(output_root / "catalog.json", catalog_file_bytes)
        _write_json_atomic(
            marker_path,
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "base_url": normalized_base_url,
                "captured_at": capture_time,
                "corpora": selected_names,
                "complete": False,
            },
        )

        counts_before: dict[str, int] = {}
        counts_after: dict[str, int] = {}
        artifacts: list[ApiCaptureArtifact] = []
        for entry in selected_catalog:
            before = _exact_count(http_client, normalized_base_url, entry.corpus)
            if before != entry.doc_cnt:
                raise CorpusCaptureError(
                    f"Catalog count for {entry.corpus} is {entry.doc_cnt}, "
                    f"but the exact API count is {before}"
                )
            counts_before[entry.corpus] = before
            artifact = _capture_collection(
                http_client,
                normalized_base_url,
                output_root,
                entry.corpus,
                before,
                page_size,
                resume=resume,
            )
            after = _exact_count(http_client, normalized_base_url, entry.corpus)
            counts_after[entry.corpus] = after
            if after != before:
                raise CorpusCaptureError(
                    f"Exact count for {entry.corpus} changed during capture: "
                    f"{before} -> {after}"
                )
            if artifact.document_count != before:
                raise CorpusCaptureError(
                    f"Collection {entry.corpus} captured {artifact.document_count} "
                    f"documents but expected {before}"
                )
            artifacts.append(artifact)

        manifest = ApiCaptureManifest(
            schema_version=CAPTURE_SCHEMA_VERSION,
            base_url=normalized_base_url,
            captured_at=capture_time,
            select_fields=list(CAPTURE_SELECT_FIELDS),
            order="doc_id.asc",
            catalog_path="catalog.json",
            catalog_size_bytes=len(catalog_file_bytes),
            catalog_sha256=hashlib.sha256(catalog_file_bytes).hexdigest(),
            counts_before=counts_before,
            counts_after=counts_after,
            artifacts=artifacts,
            complete=True,
        )
        verify_api_capture(manifest, output_root)
        _write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
        marker_path.unlink(missing_ok=True)
        return manifest
    except CorpusCaptureError:
        raise
    except (httpx.HTTPError, OSError, ValueError) as error:
        raise CorpusCaptureError(str(error)) from error
    finally:
        if owns_client:
            http_client.close()


def verify_api_capture(manifest: ApiCaptureManifest, root: Path) -> None:
    if not manifest.complete:
        raise CorpusCaptureError("History Lab API capture is incomplete")
    if manifest.schema_version != CAPTURE_SCHEMA_VERSION:
        raise CorpusCaptureError("History Lab API capture schema is unsupported")
    if tuple(manifest.select_fields) != CAPTURE_SELECT_FIELDS:
        raise CorpusCaptureError("History Lab API capture projection is incomplete")
    if manifest.order != "doc_id.asc":
        raise CorpusCaptureError("History Lab API capture order is not deterministic")
    catalog_path = root.joinpath(*PurePosixPath(manifest.catalog_path).parts)
    if not catalog_path.is_file():
        raise CorpusCaptureError("Capture catalog artifact is missing")
    if catalog_path.stat().st_size != manifest.catalog_size_bytes:
        raise CorpusCaptureError("Capture catalog artifact size mismatch")
    if _sha256_file(catalog_path) != manifest.catalog_sha256:
        raise CorpusCaptureError("Capture catalog artifact checksum mismatch")
    try:
        catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog = [CatalogEntry.model_validate(item) for item in catalog_payload]
    except (OSError, TypeError, ValueError) as error:
        raise CorpusCaptureError(
            "Capture catalog artifact has an invalid schema"
        ) from error
    catalog_counts = {entry.corpus: entry.doc_cnt for entry in catalog}
    if len(catalog_counts) != len(catalog):
        raise CorpusCaptureError("Capture catalog contains duplicate corpora")
    if catalog_counts != manifest.counts_before:
        raise CorpusCaptureError("Capture catalog counts do not match the manifest")
    for artifact in manifest.artifacts:
        path = root.joinpath(*PurePosixPath(artifact.path).parts)
        if not path.is_file():
            raise CorpusCaptureError(f"Capture artifact is missing: {artifact.path}")
        if path.stat().st_size != artifact.size_bytes:
            raise CorpusCaptureError(f"Capture artifact size mismatch: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            raise CorpusCaptureError(
                f"Capture artifact checksum mismatch: {artifact.path}"
            )


def validate_capture_for_paper(
    capture: ApiCaptureManifest,
    capture_root: Path,
    exclusions: Any,
    frus: Any,
) -> CorpusCoverageReport:
    from redacted_backend.corpus.exclusions import ExclusionManifest
    from redacted_backend.corpus.frus_metadata import (
        FrusMetadataError,
        FrusPublicationManifest,
        match_frus_volume,
    )

    exclusion_manifest = ExclusionManifest.model_validate(exclusions)
    frus_manifest = FrusPublicationManifest.model_validate(frus)
    verify_api_capture(capture, capture_root)
    reasons: list[str] = []
    corpus_counts = {
        artifact.corpus: artifact.document_count for artifact in capture.artifacts
    }
    archive_document_count = sum(corpus_counts.values())
    present_corpora = set(corpus_counts)
    expected_count = exclusion_manifest.expected_archive_document_count
    count_tolerance = exclusion_manifest.archive_document_count_tolerance
    if abs(archive_document_count - expected_count) > count_tolerance:
        reasons.append(
            f"archive_document_count:{archive_document_count}:"
            f"expected:{expected_count}:tolerance:{count_tolerance}"
        )
    for corpus in sorted(exclusion_manifest.required_corpora):
        if corpus_counts.get(corpus, 0) == 0:
            reasons.append(f"required_corpus_missing:{corpus}")
    for corpus in sorted(present_corpora - exclusion_manifest.known_corpora):
        reasons.append(f"unknown_corpus:{corpus}")
    if not exclusion_manifest.cia_coverage_complete:
        reasons.append("cia_collection_mapping_unverified")
    elif exclusion_manifest.cia_reading_room_corpus not in present_corpora:
        reasons.append("cia_reading_room_corpus_missing")
    if not exclusion_manifest.frus_coverage_complete:
        reasons.append("frus_coverage_unverified")
    if exclusion_manifest.frus_metadata_sha256 != frus_manifest.source_sha256:
        reasons.append("frus_metadata_checksum_mismatch")
    if (
        exclusion_manifest.frus_publication_years
        != frus_manifest.known_publication_years
    ):
        reasons.append("frus_publication_manifest_mismatch")

    post_cutoff_volumes: set[str] = set()
    unmatched_count = 0
    unmatched_samples: list[str] = []
    unknown_year_count = 0
    try:
        frus_artifact = capture.artifact_for("frus")
    except KeyError:
        reasons.append("required_corpus_missing:frus")
    else:
        path = capture_root.joinpath(*PurePosixPath(frus_artifact.path).parts)
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                document = NormalizedCorpusDocument.model_validate_json(line)
                try:
                    volume_id = match_frus_volume(
                        document.doc_id,
                        frus_manifest.publication_years,
                    )
                except FrusMetadataError:
                    unmatched_count += 1
                    if len(unmatched_samples) < 20:
                        unmatched_samples.append(document.doc_id)
                    continue
                year = frus_manifest.publication_years[volume_id]
                if year is None:
                    unknown_year_count += 1
                    continue
                if 2019 <= year <= 2024:
                    post_cutoff_volumes.add(volume_id)
        if unmatched_count:
            reasons.append(f"unmatched_frus_document_ids:{unmatched_count}")
        if unknown_year_count:
            reasons.append(f"frus_publication_year_unknown:{unknown_year_count}")
        observed = len(post_cutoff_volumes)
        expected = exclusion_manifest.expected_post_cutoff_frus_volumes
        if observed != expected:
            reasons.append(
                f"post_cutoff_frus_volume_count:{observed}:expected:{expected}"
            )

    unique_reasons = sorted(set(reasons))
    return CorpusCoverageReport(
        capture_manifest_hash=capture.manifest_hash(),
        exclusion_manifest_hash=exclusion_manifest.manifest_hash(),
        frus_manifest_hash=frus_manifest.manifest_hash(),
        ready=not unique_reasons,
        reasons=unique_reasons,
        archive_document_count=archive_document_count,
        corpus_counts=dict(sorted(corpus_counts.items())),
        post_cutoff_frus_volumes=sorted(post_cutoff_volumes),
        unmatched_frus_document_count=unmatched_count,
        unmatched_frus_document_samples=unmatched_samples,
    )


def _fetch_catalog(client: httpx.Client, base_url: str) -> list[CatalogEntry]:
    response = client.get(
        f"{base_url}/corpora",
        params={"order": "corpus.asc"},
        headers={"Accept": "application/json"},
    )
    _raise_for_status(response, "catalog")
    try:
        payload = response.json()
    except ValueError as error:
        raise CorpusCaptureError(
            "History Lab corpus catalog has an invalid schema"
        ) from error
    if not isinstance(payload, list):
        raise CorpusCaptureError("History Lab corpus catalog has an invalid schema")
    entries: list[CatalogEntry] = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("corpus"), str):
            raise CorpusCaptureError("History Lab corpus catalog has an invalid schema")
        _validate_corpus(item["corpus"])
        try:
            entries.append(CatalogEntry.model_validate(item))
        except ValueError as error:
            raise CorpusCaptureError(
                "History Lab corpus catalog has an invalid schema"
            ) from error
    names = [item.corpus for item in entries]
    if not entries or len(names) != len(set(names)):
        raise CorpusCaptureError(
            "History Lab corpus catalog must contain unique corpora"
        )
    return entries


def _exact_count(client: httpx.Client, base_url: str, corpus: str) -> int:
    response = client.get(
        f"{base_url}/docs",
        params={
            "select": "doc_id",
            "corpus": f"eq.{corpus}",
            "limit": "1",
        },
        headers={
            "Accept": "application/json",
            "Prefer": "count=exact",
            "Range": "0-0",
        },
    )
    _raise_for_status(response, f"exact count for {corpus}")
    content_range = response.headers.get("Content-Range", "")
    match = re.search(r"/(\d+)$", content_range)
    if match is None:
        raise CorpusCaptureError(
            f"Exact count response for {corpus} has no usable Content-Range"
        )
    return int(match.group(1))


def _capture_collection(
    client: httpx.Client,
    base_url: str,
    output_root: Path,
    corpus: str,
    expected_count: int,
    page_size: int,
    *,
    resume: bool,
) -> ApiCaptureArtifact:
    collections = output_root / "collections"
    final_path = collections / f"{corpus}.jsonl"
    partial_path = collections / f"{corpus}.jsonl.part"
    checkpoint_path = collections / f"{corpus}.checkpoint.json"
    relative_path = f"collections/{corpus}.jsonl"
    if final_path.exists():
        artifact = _artifact_from_file(final_path, corpus, relative_path)
        if artifact.document_count != expected_count:
            raise CorpusCaptureError(
                f"Completed collection {corpus} has {artifact.document_count} documents, "
                f"expected {expected_count}"
            )
        return artifact

    checkpoint = _load_or_create_checkpoint(
        partial_path,
        checkpoint_path,
        corpus,
        expected_count,
        resume=resume,
    )
    while checkpoint.document_count < expected_count:
        params = {
            "select": ",".join(CAPTURE_SELECT_FIELDS),
            "corpus": f"eq.{corpus}",
            "order": "doc_id.asc",
            "limit": str(page_size),
        }
        if checkpoint.last_doc_id is not None:
            params["doc_id"] = f"gt.{checkpoint.last_doc_id}"
        response = client.get(
            f"{base_url}/docs",
            params=params,
            headers={"Accept": "application/json"},
        )
        _raise_for_status(response, f"document page for {corpus}")
        try:
            raw_rows = response.json()
        except ValueError as error:
            raise CorpusCaptureError(
                f"Document page for {corpus} is not valid JSON"
            ) from error
        if not isinstance(raw_rows, list):
            raise CorpusCaptureError(f"Document page for {corpus} is not a JSON array")
        if not raw_rows:
            break
        encoded_rows: list[bytes] = []
        last_doc_id = checkpoint.last_doc_id
        first_doc_id = checkpoint.first_doc_id
        for raw_row in raw_rows:
            try:
                document = NormalizedCorpusDocument.model_validate(raw_row)
            except ValueError as error:
                raise CorpusCaptureError(
                    f"Document page for {corpus} has an invalid schema"
                ) from error
            if document.corpus != corpus:
                raise CorpusCaptureError(
                    f"Document {document.doc_id} was returned under the wrong corpus"
                )
            if last_doc_id is not None and document.doc_id <= last_doc_id:
                raise CorpusCaptureError(
                    f"Document IDs for {corpus} are not strictly increasing"
                )
            first_doc_id = first_doc_id or document.doc_id
            last_doc_id = document.doc_id
            encoded_rows.append(
                _canonical_json_bytes(document.model_dump(mode="json")) + b"\n"
            )
        new_count = checkpoint.document_count + len(encoded_rows)
        if new_count > expected_count:
            raise CorpusCaptureError(
                f"Collection {corpus} returned more documents than its exact count"
            )
        with partial_path.open("ab") as destination:
            for encoded in encoded_rows:
                destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        checkpoint = _Checkpoint(
            corpus=corpus,
            expected_count=expected_count,
            committed_bytes=partial_path.stat().st_size,
            document_count=new_count,
            first_doc_id=first_doc_id,
            last_doc_id=last_doc_id,
        )
        _write_json_atomic(checkpoint_path, checkpoint.model_dump(mode="json"))

    if checkpoint.document_count != expected_count:
        raise CorpusCaptureError(
            f"Collection {corpus} captured {checkpoint.document_count} documents "
            f"but expected {expected_count}"
        )
    os.replace(partial_path, final_path)
    checkpoint_path.unlink(missing_ok=True)
    return _artifact_from_file(final_path, corpus, relative_path)


def _load_or_create_checkpoint(
    partial_path: Path,
    checkpoint_path: Path,
    corpus: str,
    expected_count: int,
    *,
    resume: bool,
) -> _Checkpoint:
    if partial_path.exists() or checkpoint_path.exists():
        if not resume:
            raise CorpusCaptureError(
                f"Partial capture for {corpus} exists and resume is disabled"
            )
        if not partial_path.is_file() or not checkpoint_path.is_file():
            raise CorpusCaptureError(
                f"Partial capture state for {corpus} is incomplete"
            )
        checkpoint = _Checkpoint.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
        if checkpoint.corpus != corpus or checkpoint.expected_count != expected_count:
            raise CorpusCaptureError(
                f"Partial capture state for {corpus} does not match the current API"
            )
        size = partial_path.stat().st_size
        if size < checkpoint.committed_bytes:
            raise CorpusCaptureError(f"Partial capture for {corpus} was truncated")
        if size > checkpoint.committed_bytes:
            with partial_path.open("r+b") as destination:
                destination.truncate(checkpoint.committed_bytes)
                destination.flush()
                os.fsync(destination.fileno())
        _validate_checkpoint_file(partial_path, checkpoint)
        return checkpoint

    partial_path.touch(exist_ok=False)
    checkpoint = _Checkpoint(
        corpus=corpus,
        expected_count=expected_count,
        committed_bytes=0,
        document_count=0,
        first_doc_id=None,
        last_doc_id=None,
    )
    _write_json_atomic(checkpoint_path, checkpoint.model_dump(mode="json"))
    return checkpoint


def _validate_checkpoint_file(path: Path, checkpoint: _Checkpoint) -> None:
    count = 0
    first_doc_id: str | None = None
    last_doc_id: str | None = None
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.endswith("\n"):
                raise CorpusCaptureError(
                    f"Checkpointed capture for {checkpoint.corpus} has a partial line"
                )
            document = NormalizedCorpusDocument.model_validate_json(line)
            if last_doc_id is not None and document.doc_id <= last_doc_id:
                raise CorpusCaptureError(
                    f"Checkpointed IDs for {checkpoint.corpus} are not strictly increasing"
                )
            first_doc_id = first_doc_id or document.doc_id
            last_doc_id = document.doc_id
            count += 1
    if (
        count != checkpoint.document_count
        or first_doc_id != checkpoint.first_doc_id
        or last_doc_id != checkpoint.last_doc_id
    ):
        raise CorpusCaptureError(
            f"Checkpoint metadata for {checkpoint.corpus} does not match partial bytes"
        )


def _artifact_from_file(
    path: Path,
    corpus: str,
    relative_path: str,
) -> ApiCaptureArtifact:
    count = 0
    first_doc_id: str | None = None
    last_doc_id: str | None = None
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            document = NormalizedCorpusDocument.model_validate_json(line)
            if document.corpus != corpus:
                raise CorpusCaptureError(
                    f"Capture artifact {relative_path} contains another corpus"
                )
            if last_doc_id is not None and document.doc_id <= last_doc_id:
                raise CorpusCaptureError(
                    f"Capture artifact {relative_path} is not strictly ordered"
                )
            first_doc_id = first_doc_id or document.doc_id
            last_doc_id = document.doc_id
            count += 1
    return ApiCaptureArtifact(
        corpus=corpus,
        path=relative_path,
        document_count=count,
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
        first_doc_id=first_doc_id,
        last_doc_id=last_doc_id,
    )


def _capture_time(marker_path: Path, requested: str | None, resume: bool) -> str:
    if marker_path.exists() and resume:
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
            existing = payload.get("captured_at")
            if isinstance(existing, str) and existing:
                return existing
        except (OSError, ValueError):
            pass
    return requested or datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_corpus(value: str) -> str:
    normalized = value.strip().lower()
    if not _CORPUS_PATTERN.fullmatch(normalized):
        raise CorpusCaptureError(f"unsafe corpus identifier: {value!r}")
    return normalized


def _raise_for_status(response: httpx.Response, operation: str) -> None:
    if response.is_success:
        return
    raise CorpusCaptureError(
        f"History Lab API {operation} failed with HTTP {response.status_code}"
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    _write_bytes_atomic(
        path, json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    )


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            destination.write(value)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
