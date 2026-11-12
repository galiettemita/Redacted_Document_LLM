from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from redacted_backend.corpus.chunker import CorpusDocument, chunk_document
from redacted_backend.corpus.exclusions import ExclusionManifest, should_include
from redacted_backend.corpus.snapshot import SnapshotLock, verify_snapshot_inputs
from redacted_backend.scientific_config import ScientificConfig


class CorpusBuildError(ValueError):
    pass


class CorpusManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    exclusion_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunker_version: str
    chunk_words: int
    chunk_overlap: int
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    source_counts: dict[str, int]
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
    source_paths: Sequence[Path],
    output_path: Path,
    *,
    snapshot: SnapshotLock,
    exclusions: ExclusionManifest,
    scientific: ScientificConfig,
) -> CorpusManifest:
    if not source_paths:
        raise ValueError("At least one source JSONL file is required")
    verify_snapshot_inputs(snapshot, source_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    document_count = 0
    chunk_count = 0
    decisions_ready = True
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
            for source_path in source_paths:
                with _open_json_lines(source_path) as source:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        try:
                            document = CorpusDocument.model_validate_json(line)
                        except (ValidationError, ValueError) as error:
                            raise CorpusBuildError(
                                "Record does not match the nine-field schema at "
                                f"{source_path}:{line_number}"
                            ) from error
                        decision = should_include(
                            document.model_dump(mode="python"),
                            exclusions,
                        )
                        decisions_ready = decisions_ready and decision.ready
                        if not decision.included:
                            exclusion_counts[decision.reason] += 1
                            continue
                        chunks = chunk_document(
                            document,
                            max_words=scientific.chunk_words,
                        )
                        document_count += 1
                        chunk_count += len(chunks)
                        source_counts[document.source] += 1
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
        schema_version="foiarchive-nine-fields-v1",
        snapshot_hash=snapshot.manifest_hash(),
        exclusion_hash=exclusions.manifest_hash(),
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=document_count,
        chunk_count=chunk_count,
        source_counts=dict(sorted(source_counts.items())),
        exclusion_counts=dict(sorted(exclusion_counts.items())),
        coverage_ready=exclusions.coverage_ready and decisions_ready,
        chunks_sha256=digest.hexdigest(),
    )


def _open_json_lines(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")
