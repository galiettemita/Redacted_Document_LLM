from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from redacted_backend.corpus.best_available import (
    BEST_AVAILABLE_BM25_INDEX_NAME,
    BestAvailableCorpusManifest,
)
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.retrieval.bm25 import (
    Bm25Index,
    OpenSearchHttpTransport,
    SearchTransport,
    _index_document,
    build_index_definition,
    index_definition_hash,
)
from redacted_backend.retrieval.types import Bm25IndexManifest
from redacted_backend.scientific_config import ScientificConfig


class Bm25BuildReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    elapsed_seconds: float = Field(ge=0)
    index_manifest: Bm25IndexManifest


class _BestAvailableBm25Checkpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["best-available-bm25-checkpoint-v1"] = (
        "best-available-bm25-checkpoint-v1"
    )
    corpus_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_definition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_name: Literal["foiarchive-best-available-bm25-v1"]
    indexed_chunk_count: int = Field(ge=0)
    last_chunk_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


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


def build_best_available_bm25(
    chunks_path: Path,
    corpus_manifest_path: Path,
    output_manifest_path: Path,
    checkpoint_path: Path,
    *,
    opensearch_url: str,
    transport: SearchTransport | None = None,
    scientific: ScientificConfig | None = None,
    batch_size: int | None = None,
) -> Bm25BuildReport:
    config = scientific or ScientificConfig()
    effective_batch_size = batch_size or config.bm25_bulk_batch_size
    if effective_batch_size < 1:
        raise ValueError("BM25 batch size must be positive")
    corpus = BestAvailableCorpusManifest.model_validate_json(
        corpus_manifest_path.read_text(encoding="utf-8")
    )
    chunks_sha256 = _sha256_file(chunks_path)
    if chunks_sha256 != corpus.chunks_sha256:
        raise ValueError("Corpus chunk checksum does not match its manifest")
    if corpus.scientific_hash != config.manifest_hash():
        raise ValueError("Corpus scientific manifest does not match BM25 configuration")

    corpus_hash = corpus.manifest_hash()
    scientific_hash = config.manifest_hash()
    definition_hash = index_definition_hash(config)
    search = transport or OpenSearchHttpTransport(opensearch_url)
    checkpoint = _load_best_available_checkpoint(
        checkpoint_path,
        corpus_manifest_hash=corpus_hash,
        scientific_hash=scientific_hash,
        definition_hash=definition_hash,
    )
    if checkpoint is None:
        search.replace_index(
            BEST_AVAILABLE_BM25_INDEX_NAME,
            build_index_definition(config),
        )
        indexed_count = 0
        last_chunk_id: str | None = None
    else:
        indexed_count = checkpoint.indexed_chunk_count
        last_chunk_id = checkpoint.last_chunk_id
        live_count = search.count(BEST_AVAILABLE_BM25_INDEX_NAME)
        if live_count < indexed_count:
            raise ValueError("BM25 live index count is behind checkpoint")
        if live_count > corpus.chunk_count:
            raise ValueError("BM25 live index count exceeds corpus manifest")

    started = time.monotonic()
    batch: list[dict[str, object]] = []
    observed_count = 0
    observed_checkpoint_id: str | None = None
    for chunk in _read_chunks(chunks_path):
        observed_count += 1
        if observed_count <= indexed_count:
            if observed_count == indexed_count:
                observed_checkpoint_id = chunk.id
            continue
        batch.append(_index_document(chunk))
        if len(batch) >= effective_batch_size:
            search.bulk_index(BEST_AVAILABLE_BM25_INDEX_NAME, batch)
            indexed_count += len(batch)
            last_chunk_id = str(batch[-1]["chunk_id"])
            _write_best_available_checkpoint(
                checkpoint_path,
                corpus_manifest_hash=corpus_hash,
                scientific_hash=scientific_hash,
                definition_hash=definition_hash,
                indexed_chunk_count=indexed_count,
                last_chunk_id=last_chunk_id,
            )
            batch = []
    if checkpoint is not None and observed_checkpoint_id != checkpoint.last_chunk_id:
        raise ValueError("BM25 checkpoint does not match chunk ordering")
    if batch:
        search.bulk_index(BEST_AVAILABLE_BM25_INDEX_NAME, batch)
        indexed_count += len(batch)
        last_chunk_id = str(batch[-1]["chunk_id"])
        _write_best_available_checkpoint(
            checkpoint_path,
            corpus_manifest_hash=corpus_hash,
            scientific_hash=scientific_hash,
            definition_hash=definition_hash,
            indexed_chunk_count=indexed_count,
            last_chunk_id=last_chunk_id,
        )
    if indexed_count != corpus.chunk_count or observed_count != corpus.chunk_count:
        raise ValueError("BM25 indexed chunk count does not match corpus manifest")

    search.refresh(BEST_AVAILABLE_BM25_INDEX_NAME)
    live_count = search.count(BEST_AVAILABLE_BM25_INDEX_NAME)
    if live_count != corpus.chunk_count:
        raise ValueError("BM25 live index count does not match corpus manifest")
    manifest = Bm25IndexManifest(
        index_name=BEST_AVAILABLE_BM25_INDEX_NAME,
        retrieval_profile="best_available_foiarchive_v1",
        opensearch_version=config.opensearch_version,
        corpus_manifest_hash=corpus_hash,
        scientific_hash=scientific_hash,
        index_definition_hash=definition_hash,
        stopwords_hash=config.context_stopwords_hash(),
        document_count=corpus.document_count,
        chunk_count=corpus.chunk_count,
    )
    _write_json_atomic(
        output_manifest_path,
        manifest.model_dump(mode="json"),
    )
    checkpoint_path.unlink(missing_ok=True)
    return Bm25BuildReport(
        elapsed_seconds=time.monotonic() - started,
        index_manifest=manifest,
    )


def _load_best_available_checkpoint(
    path: Path,
    *,
    corpus_manifest_hash: str,
    scientific_hash: str,
    definition_hash: str,
) -> _BestAvailableBm25Checkpoint | None:
    if not path.exists():
        return None
    try:
        checkpoint = _BestAvailableBm25Checkpoint.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise ValueError("BM25 checkpoint is invalid") from error
    if (
        checkpoint.corpus_manifest_hash != corpus_manifest_hash
        or checkpoint.scientific_hash != scientific_hash
        or checkpoint.index_definition_hash != definition_hash
        or checkpoint.index_name != BEST_AVAILABLE_BM25_INDEX_NAME
    ):
        raise ValueError("BM25 checkpoint is incompatible")
    return checkpoint


def _write_best_available_checkpoint(
    path: Path,
    *,
    corpus_manifest_hash: str,
    scientific_hash: str,
    definition_hash: str,
    indexed_chunk_count: int,
    last_chunk_id: str | None,
) -> None:
    checkpoint = _BestAvailableBm25Checkpoint(
        corpus_manifest_hash=corpus_manifest_hash,
        scientific_hash=scientific_hash,
        index_definition_hash=definition_hash,
        index_name=BEST_AVAILABLE_BM25_INDEX_NAME,
        indexed_chunk_count=indexed_chunk_count,
        last_chunk_id=last_chunk_id,
    )
    _write_checkpoint_atomic(path, checkpoint.model_dump(mode="json"))


def _write_checkpoint_atomic(path: Path, value: object) -> None:
    _write_json_atomic(path, value)


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
