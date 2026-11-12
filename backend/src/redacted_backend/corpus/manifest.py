from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from redacted_backend.corpus.chunker import CorpusDocument, chunk_document
from redacted_backend.corpus.exclusions import ExclusionManifest, should_include
from redacted_backend.corpus.historylab_export import (
    ApiCaptureManifest,
    CorpusCoverageReport,
    verify_api_capture,
)
from redacted_backend.scientific_config import ScientificConfig


class CorpusBuildError(ValueError):
    pass


class CorpusManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    exclusion_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunker_version: str
    chunk_words: int
    chunk_overlap: int
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    corpus_counts: dict[str, int]
    exclusion_counts: dict[str, int]
    coverage_ready: bool
    chunks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def manifest_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def build_chunk_file(
    capture_root: Path,
    output_path: Path,
    *,
    capture: ApiCaptureManifest,
    coverage: CorpusCoverageReport,
    exclusions: ExclusionManifest,
    scientific: ScientificConfig,
) -> CorpusManifest:
    capture_hash = capture.manifest_hash()
    exclusion_hash = exclusions.manifest_hash()
    if not coverage.ready:
        raise CorpusBuildError("Paper corpus coverage report is not ready")
    if coverage.capture_manifest_hash != capture_hash:
        raise CorpusBuildError("Coverage report does not match the API capture")
    if coverage.exclusion_manifest_hash != exclusion_hash:
        raise CorpusBuildError("Coverage report does not match the exclusion manifest")
    verify_api_capture(capture, capture_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    document_count = 0
    chunk_count = 0
    digest = hashlib.sha256()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            for artifact in sorted(capture.artifacts, key=lambda item: item.corpus):
                source_path = capture_root.joinpath(*PurePosixPath(artifact.path).parts)
                with source_path.open("r", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        try:
                            document = CorpusDocument.model_validate_json(line)
                        except (ValidationError, ValueError) as error:
                            raise CorpusBuildError(
                                "Record does not match the normalized History Lab schema at "
                                f"{artifact.path}:{line_number}"
                            ) from error
                        decision = should_include(
                            document.model_dump(mode="python"),
                            exclusions,
                        )
                        if not decision.ready:
                            raise CorpusBuildError(
                                f"Unresolved exclusion for {document.doc_id}: "
                                f"{decision.reason}"
                            )
                        if not decision.included:
                            exclusion_counts[decision.reason] += 1
                            continue
                        chunks = chunk_document(
                            document,
                            max_words=scientific.chunk_words,
                        )
                        document_count += 1
                        chunk_count += len(chunks)
                        corpus_counts[document.corpus] += 1
                        for chunk in chunks:
                            serialized = (
                                json.dumps(
                                    chunk.model_dump(mode="json"),
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                                + b"\n"
                            )
                            destination.write(serialized)
                            digest.update(serialized)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return CorpusManifest(
        schema_version="historylab-normalized-v1",
        source_manifest_hash=capture_hash,
        coverage_report_hash=coverage.manifest_hash(),
        exclusion_hash=exclusion_hash,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=document_count,
        chunk_count=chunk_count,
        corpus_counts=dict(sorted(corpus_counts.items())),
        exclusion_counts=dict(sorted(exclusion_counts.items())),
        coverage_ready=True,
        chunks_sha256=digest.hexdigest(),
    )
