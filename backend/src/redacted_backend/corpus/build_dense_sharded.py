from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.corpus.best_available import BestAvailableCorpusManifest
from redacted_backend.corpus.build_dense import FaissHnswIndex
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.dense import BgeEncoder, require_dense_recall
from redacted_backend.retrieval.dense_sharded import (
    CandidateLoader,
    certify_sharded_dense_recall,
    select_certification_queries,
    verify_dense_shard_artifacts,
    verify_sharded_dense_artifacts,
)
from redacted_backend.retrieval.types import (
    DenseShardManifest,
    ShardedDenseIndexManifest,
)
from redacted_backend.scientific_config import ScientificConfig


MAX_DENSE_SHARD_CHUNKS = 1_000_000


class DocumentEncoder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def encode_query(self, text: str) -> NDArray[np.float32]: ...


class CandidateIndexBuilder(Protocol):
    def add(self, vectors: NDArray[np.float32]) -> None: ...

    def search(
        self,
        query: NDArray[np.float32],
        candidates: int,
    ) -> NDArray[np.int64]: ...

    def save(self, path: Path) -> None: ...


CandidateFactory = Callable[[ScientificConfig], CandidateIndexBuilder]


class ShardedDenseBuildReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    elapsed_seconds: float = Field(ge=0)
    corpus_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_manifest: ShardedDenseIndexManifest


def build_sharded_dense_from_files(
    chunks_path: Path,
    corpus_manifest_path: Path,
    output_directory: Path,
    output_manifest_path: Path,
    *,
    application_queries_path: Path,
    certification_queries_path: Path,
    encoder: DocumentEncoder | None = None,
    candidate_factory: CandidateFactory | None = None,
    candidate_loader: CandidateLoader | None = None,
    scientific: ScientificConfig | None = None,
    batch_size: int = 64,
    shard_size: int = MAX_DENSE_SHARD_CHUNKS,
    application_query_count: int = 50,
    corpus_query_count: int = 50,
    recall_block_rows: int = 16_384,
) -> ShardedDenseBuildReport:
    if batch_size < 1:
        raise ValueError("Dense encoding batch size must be positive")
    if shard_size < 1 or shard_size > MAX_DENSE_SHARD_CHUNKS:
        raise ValueError(
            f"Dense shard size must be between 1 and {MAX_DENSE_SHARD_CHUNKS}"
        )
    config = scientific or ScientificConfig()
    corpus = BestAvailableCorpusManifest.model_validate_json(
        corpus_manifest_path.read_text(encoding="utf-8")
    )
    chunks_sha256 = _sha256_file(chunks_path)
    if chunks_sha256 != corpus.chunks_sha256:
        raise ValueError("Corpus chunk checksum does not match its manifest")
    if corpus.scientific_hash != config.manifest_hash():
        raise ValueError("Corpus scientific manifest does not match dense configuration")
    if corpus.paper_coverage_ready:
        raise ValueError("Best-available corpus must not claim exact-paper coverage")
    if corpus.chunk_count < 1:
        raise ValueError("Dense index cannot be built from an empty corpus")

    output_directory.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    queries = select_certification_queries(
        chunks_path,
        application_queries_path,
        certification_queries_path,
        application_query_count=application_query_count,
        corpus_query_count=corpus_query_count,
    )
    queries_sha256 = _sha256_file(certification_queries_path)
    corpus_hash = corpus.manifest_hash()
    existing = _load_completed_manifest(
        output_manifest_path,
        output_directory,
        corpus_hash=corpus_hash,
        chunks_sha256=chunks_sha256,
        queries_sha256=queries_sha256,
        query_count=len(queries),
        config=config,
    )
    if existing is not None:
        return ShardedDenseBuildReport(
            elapsed_seconds=time.monotonic() - started,
            corpus_manifest_hash=corpus_hash,
            index_manifest=existing,
        )

    embedding_encoder = encoder or BgeEncoder(scientific=config)
    make_candidate = candidate_factory or FaissHnswIndex.create
    load_candidate = candidate_loader or FaissHnswIndex.load
    chunks = _read_chunks(chunks_path)
    shards: list[DenseShardManifest] = []
    start_row = 0
    shard_id = 0
    while start_row < corpus.chunk_count:
        chunk_count = min(shard_size, corpus.chunk_count - start_row)
        directory_name = f"shard-{shard_id:05d}"
        final_directory = output_directory / directory_name
        existing_shard = _load_completed_shard(
            output_directory,
            final_directory,
            expected_id=shard_id,
            expected_start=start_row,
            expected_count=chunk_count,
            embedding_dimension=config.embedding_dimension,
        )
        if existing_shard is not None:
            _consume_chunks(chunks, chunk_count)
            shard = existing_shard
        else:
            shard = _build_shard(
                chunks,
                output_directory,
                directory_name=directory_name,
                shard_id=shard_id,
                start_row=start_row,
                chunk_count=chunk_count,
                encoder=embedding_encoder,
                candidate=make_candidate(config),
                config=config,
                batch_size=batch_size,
            )
        shards.append(shard)
        start_row = shard.end_row
        shard_id += 1
    try:
        extra = next(chunks)
    except StopIteration:
        extra = None
    if extra is not None:
        raise ValueError("Dense chunk count exceeds the corpus manifest")

    preliminary = _manifest(
        corpus_hash=corpus_hash,
        chunks_sha256=chunks_sha256,
        chunk_count=corpus.chunk_count,
        shard_size=shard_size,
        shards=shards,
        queries_sha256=queries_sha256,
        query_count=len(queries),
        recall=1.0,
        config=config,
    )
    recall = certify_sharded_dense_recall(
        output_directory,
        preliminary,
        queries,
        encoder=embedding_encoder,
        candidate_loader=load_candidate,
        scientific=config,
        limit=config.top_chunks_per_list,
        block_rows=recall_block_rows,
    )
    require_dense_recall(recall, minimum=config.dense_recall_minimum)
    manifest = preliminary.model_copy(update={"recall_at_300": recall})
    _write_json_atomic(output_manifest_path, manifest.model_dump(mode="json"))
    return ShardedDenseBuildReport(
        elapsed_seconds=time.monotonic() - started,
        corpus_manifest_hash=corpus_hash,
        index_manifest=manifest,
    )


def _build_shard(
    chunks: Iterator[CorpusChunk],
    output_root: Path,
    *,
    directory_name: str,
    shard_id: int,
    start_row: int,
    chunk_count: int,
    encoder: DocumentEncoder,
    candidate: CandidateIndexBuilder,
    config: ScientificConfig,
    batch_size: int,
) -> DenseShardManifest:
    temporary = Path(
        tempfile.mkdtemp(
            dir=output_root,
            prefix=f".{directory_name}.",
        )
    )
    final_directory = output_root / directory_name
    try:
        embeddings_path = temporary / "embeddings.f16"
        metadata_path = temporary / "metadata.jsonl"
        offsets_path = temporary / "metadata.offsets.u64"
        candidate_path = temporary / "candidate.faiss"
        embeddings = np.memmap(
            embeddings_path,
            dtype=np.float16,
            mode="w+",
            shape=(chunk_count, config.embedding_dimension),
        )
        offsets = np.memmap(
            offsets_path,
            dtype="<u8",
            mode="w+",
            shape=(chunk_count,),
        )
        row = 0
        with metadata_path.open("wb") as metadata:
            while row < chunk_count:
                count = min(batch_size, chunk_count - row)
                batch = _take_chunks(chunks, count)
                vectors = np.asarray(
                    encoder.encode_documents([chunk.text for chunk in batch]),
                    dtype=np.float32,
                )
                expected_shape = (len(batch), config.embedding_dimension)
                if vectors.shape != expected_shape:
                    raise ValueError(
                        f"Dense encoder returned shape {vectors.shape}; "
                        f"expected {expected_shape}"
                    )
                norms = np.linalg.norm(vectors, axis=1)
                if np.any(~np.isfinite(vectors)) or not np.allclose(
                    norms,
                    1.0,
                    atol=1e-4,
                ):
                    raise ValueError(
                        "Dense document embeddings must be finite and L2-normalized"
                    )
                embeddings[row : row + len(batch)] = vectors.astype(np.float16)
                candidate.add(vectors)
                for chunk in batch:
                    offsets[row] = metadata.tell()
                    metadata.write(
                        json.dumps(
                            chunk.model_dump(mode="json"),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        + b"\n"
                    )
                    row += 1
            metadata.flush()
            os.fsync(metadata.fileno())
        embeddings.flush()
        offsets.flush()
        del embeddings
        del offsets
        _fsync_path(embeddings_path)
        _fsync_path(offsets_path)
        candidate.save(candidate_path)
        _fsync_path(candidate_path)

        manifest = DenseShardManifest(
            shard_id=shard_id,
            path=directory_name,
            start_row=start_row,
            end_row=start_row + chunk_count,
            chunk_count=chunk_count,
            embeddings_sha256=_sha256_file(embeddings_path),
            metadata_sha256=_sha256_file(metadata_path),
            offsets_sha256=_sha256_file(offsets_path),
            candidate_index_sha256=_sha256_file(candidate_path),
        )
        _write_json_atomic(
            temporary / "manifest.json",
            manifest.model_dump(mode="json"),
        )
        if final_directory.exists():
            raise ValueError(
                f"Incomplete dense shard directory already exists: {final_directory}"
            )
        os.replace(temporary, final_directory)
        _fsync_directory(output_root)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_completed_shard(
    output_root: Path,
    final_directory: Path,
    *,
    expected_id: int,
    expected_start: int,
    expected_count: int,
    embedding_dimension: int,
) -> DenseShardManifest | None:
    if not final_directory.exists():
        return None
    manifest_path = final_directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"Incomplete dense shard directory already exists: {final_directory}"
        )
    manifest = DenseShardManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if (
        manifest.shard_id != expected_id
        or manifest.path != final_directory.name
        or manifest.start_row != expected_start
        or manifest.chunk_count != expected_count
    ):
        raise ValueError("Completed dense shard does not match the expected range")
    verify_dense_shard_artifacts(output_root, manifest, embedding_dimension)
    return manifest


def _load_completed_manifest(
    path: Path,
    output_root: Path,
    *,
    corpus_hash: str,
    chunks_sha256: str,
    queries_sha256: str,
    query_count: int,
    config: ScientificConfig,
) -> ShardedDenseIndexManifest | None:
    if not path.exists():
        return None
    manifest = ShardedDenseIndexManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if (
        manifest.corpus_manifest_hash != corpus_hash
        or manifest.chunks_sha256 != chunks_sha256
        or manifest.recall_queries_sha256 != queries_sha256
        or manifest.recall_query_count != query_count
        or manifest.model_name != config.embedding_model
        or manifest.model_revision != config.embedding_model_revision
        or manifest.scientific_hash != config.manifest_hash()
        or manifest.embedding_dimension != config.embedding_dimension
        or manifest.candidate_count != config.dense_candidate_count
    ):
        raise ValueError("Completed sharded dense manifest is incompatible")
    verify_sharded_dense_artifacts(output_root, manifest)
    require_dense_recall(
        manifest.recall_at_300,
        minimum=config.dense_recall_minimum,
    )
    return manifest


def _manifest(
    *,
    corpus_hash: str,
    chunks_sha256: str,
    chunk_count: int,
    shard_size: int,
    shards: list[DenseShardManifest],
    queries_sha256: str,
    query_count: int,
    recall: float,
    config: ScientificConfig,
) -> ShardedDenseIndexManifest:
    return ShardedDenseIndexManifest(
        model_name=config.embedding_model,
        model_revision=config.embedding_model_revision,
        embedding_dimension=config.embedding_dimension,
        corpus_manifest_hash=corpus_hash,
        scientific_hash=config.manifest_hash(),
        chunks_sha256=chunks_sha256,
        chunk_count=chunk_count,
        shard_size_limit=shard_size,
        shards=shards,
        dtype="float16",
        metric="inner_product",
        candidate_index=config.dense_index_version,
        hnsw_m=config.dense_hnsw_m,
        ef_construction=config.dense_ef_construction,
        ef_search=config.dense_ef_search,
        candidate_count=config.dense_candidate_count,
        recall_at_300=recall,
        recall_queries_sha256=queries_sha256,
        recall_query_count=query_count,
    )


def _read_chunks(path: Path) -> Iterator[CorpusChunk]:
    with path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield CorpusChunk.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"Invalid corpus chunk at line {line_number}"
                ) from error


def _take_chunks(chunks: Iterator[CorpusChunk], count: int) -> list[CorpusChunk]:
    result: list[CorpusChunk] = []
    for _ in range(count):
        try:
            result.append(next(chunks))
        except StopIteration as error:
            raise ValueError(
                "Dense chunk count is smaller than the corpus manifest"
            ) from error
    return result


def _consume_chunks(chunks: Iterator[CorpusChunk], count: int) -> None:
    for _ in range(count):
        try:
            next(chunks)
        except StopIteration as error:
            raise ValueError(
                "Dense chunk count is smaller than the corpus manifest"
            ) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(value, destination, sort_keys=True, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_path(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
