from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.retrieval.bm25 import Bm25Index, OpenSearchHttpTransport
from redacted_backend.retrieval.types import Bm25IndexManifest
from redacted_backend.scientific_config import ScientificConfig


class Bm25BuildReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    elapsed_seconds: float = Field(ge=0)
    index_manifest: Bm25IndexManifest


def build_bm25_from_files(
    chunks_path: Path,
    corpus_manifest_path: Path,
    output_manifest_path: Path,
    *,
    opensearch_url: str,
    scientific: ScientificConfig | None = None,
) -> Bm25BuildReport:
    config = scientific or ScientificConfig()
    corpus_manifest = CorpusManifest.model_validate_json(
        corpus_manifest_path.read_text(encoding="utf-8")
    )
    chunks_sha256 = _sha256_file(chunks_path)
    if chunks_sha256 != corpus_manifest.chunks_sha256:
        raise ValueError("Corpus chunk checksum does not match its manifest")
    started = time.monotonic()
    index = Bm25Index(
        OpenSearchHttpTransport(opensearch_url),
        scientific=config,
    )
    index_manifest = index.build(_read_chunks(chunks_path), corpus_manifest)
    _write_json_atomic(
        output_manifest_path,
        index_manifest.model_dump(mode="json"),
    )
    return Bm25BuildReport(
        elapsed_seconds=time.monotonic() - started,
        index_manifest=index_manifest,
    )


def _read_chunks(path: Path) -> Iterator[CorpusChunk]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield CorpusChunk.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"Invalid corpus chunk at line {line_number}"
                ) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
