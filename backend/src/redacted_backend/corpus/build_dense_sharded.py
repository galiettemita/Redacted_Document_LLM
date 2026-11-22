from __future__ import annotations

import hashlib
import heapq
import fcntl
import json
import math
import os
import resource
import shutil
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.corpus.best_available import (
    BestAvailableCorpusManifest,
    BestAvailableRetrievalContract,
)
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.chunk_offsets import (
    ChunkOffsetIndex,
    ChunkOffsetsReport,
    build_chunk_offsets,
)
from redacted_backend.retrieval.dense import BgeEncoder, require_dense_recall
from redacted_backend.retrieval.dense_profile import (
    DenseArtifactProfile,
    dense_semantics_hash,
    dense_training_selection_hash,
    faiss_layout_hash,
)
from redacted_backend.retrieval.dense_sharded import (
    _cuda_runtime_visible,
    _matrix_scores,
    select_certification_queries,
)
from redacted_backend.retrieval.faiss_ivfpq import (
    FaissIvfPqIndex,
    train_ivfpq_template,
)
from redacted_backend.retrieval.int8_embeddings import (
    dequantize_directions,
    quantize_directions,
)
from redacted_backend.retrieval.types import (
    CompressedDenseShardManifest,
    CompressedShardedDenseIndexManifest,
    DenseRecallReport,
    ShardedDenseIndexManifest,
)
from redacted_backend.scientific_config import ScientificConfig


MAX_DENSE_SHARD_CHUNKS = 1_000_000
_CHECKPOINT_ROOT = "embedding-checkpoints"
_OFFSETS_NAME = "chunks.offsets.u64"
_OFFSETS_REPORT_NAME = "chunks.offsets-report.json"
_TRAINING_ROWS_NAME = "training-rows.u64"
_TRAINING_ROWS_MANIFEST_NAME = "training-rows-manifest.json"
_TEMPLATE_NAME = "candidate-template.faiss"
_TEMPLATE_MANIFEST_NAME = "candidate-template-manifest.json"
_RECALL_REPORT_NAME = "recall-report.json"


class StorageBudgetError(RuntimeError):
    pass


class DocumentEncoder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def encode_query(self, text: str) -> NDArray[np.float32]: ...


class EmbeddingShardCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["embedding-shard-checkpoint-v2"] = (
        "embedding-shard-checkpoint-v2"
    )
    dense_semantics_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_id: int = Field(ge=0)
    path: str = Field(min_length=1)
    start_row: int = Field(ge=0)
    end_row: int = Field(gt=0)
    chunk_count: int = Field(ge=1)
    embeddings_size_bytes: int = Field(ge=1)
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrainingRowsManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["dense-training-rows-v2"] = "dense-training-rows-v2"
    training_selection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_count: int = Field(ge=1)
    sample_count: int = Field(ge=1)
    selection_version: Literal["stable-hash-row-v1"]
    rows_size_bytes: int = Field(ge=8)
    rows_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrainedTemplateManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["trained-ivfpq-template-v2"] = (
        "trained-ivfpq-template-v2"
    )
    faiss_layout_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_rows_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: Literal["candidate-template.faiss"] = _TEMPLATE_NAME
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DenseBuildBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["dense-build-benchmark-v2"] = (
        "dense-build-benchmark-v2"
    )
    sample_rows: int = Field(ge=1)
    production_chunk_count: int = Field(ge=1)
    elapsed_seconds: float = Field(gt=0)
    setup_seconds: float = Field(ge=0)
    encoding_seconds: float = Field(gt=0)
    training_seconds: float = Field(ge=0)
    candidate_add_seconds: float = Field(ge=0)
    certification_probe_seconds: float = Field(ge=0)
    certification_probe_queries: int = Field(ge=1)
    rows_per_second: float = Field(gt=0)
    peak_rss_bytes: int | None = Field(default=None, ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    embedding_bytes_per_row: float = Field(gt=0)
    candidate_bytes_per_row: float = Field(gt=0)
    candidate_fixed_bytes_per_shard: int = Field(ge=1)
    projected_shard_count: int = Field(ge=1)
    offset_bytes_per_row: float = Field(gt=0)
    projected_full_artifact_bytes: int = Field(ge=1)
    projected_full_seconds: float = Field(ge=0)
    projected_gpu_hours: float = Field(ge=0)
    publishable: Literal[False] = False
    training_mode: Literal["benchmark-only-all-sample-rows"] = (
        "benchmark-only-all-sample-rows"
    )


DenseRecallBuildReport = DenseRecallReport


class ShardedDenseBuildReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    elapsed_seconds: float = Field(ge=0)
    corpus_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_manifest: (
        CompressedShardedDenseIndexManifest | ShardedDenseIndexManifest
    )


def build_sharded_dense_from_files(
    chunks_path: Path,
    corpus_manifest_path: Path,
    output_directory: Path,
    output_manifest_path: Path,
    *,
    application_queries_path: Path,
    certification_queries_path: Path,
    encoder: DocumentEncoder | None = None,
    scientific: ScientificConfig | None = None,
    profile: DenseArtifactProfile | None = None,
    batch_size: int = 64,
    shard_size: int = MAX_DENSE_SHARD_CHUNKS,
    application_query_count: int = 50,
    corpus_query_count: int = 50,
    recall_block_rows: int = 16_384,
    contract: BestAvailableRetrievalContract | None = None,
) -> ShardedDenseBuildReport:
    output_directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_build_lock(output_directory):
        return _build_sharded_dense_from_files_unlocked(
            chunks_path,
            corpus_manifest_path,
            output_directory,
            output_manifest_path,
            application_queries_path=application_queries_path,
            certification_queries_path=certification_queries_path,
            encoder=encoder,
            scientific=scientific,
            profile=profile,
            batch_size=batch_size,
            shard_size=shard_size,
            application_query_count=application_query_count,
            corpus_query_count=corpus_query_count,
            recall_block_rows=recall_block_rows,
            contract=contract,
        )


def _build_sharded_dense_from_files_unlocked(
    chunks_path: Path,
    corpus_manifest_path: Path,
    output_directory: Path,
    output_manifest_path: Path,
    *,
    application_queries_path: Path,
    certification_queries_path: Path,
    encoder: DocumentEncoder | None = None,
    scientific: ScientificConfig | None = None,
    profile: DenseArtifactProfile | None = None,
    batch_size: int = 64,
    shard_size: int = MAX_DENSE_SHARD_CHUNKS,
    application_query_count: int = 50,
    corpus_query_count: int = 50,
    recall_block_rows: int = 16_384,
    contract: BestAvailableRetrievalContract | None = None,
) -> ShardedDenseBuildReport:
    if batch_size < 1:
        raise ValueError("Dense encoding batch size must be positive")
    if shard_size < 1 or shard_size > MAX_DENSE_SHARD_CHUNKS:
        raise ValueError(
            f"Dense shard size must be between 1 and {MAX_DENSE_SHARD_CHUNKS}"
        )
    if recall_block_rows < 1:
        raise ValueError("Dense recall block size must be positive")
    config = scientific or ScientificConfig()
    artifact_profile = profile or DenseArtifactProfile(
        embedding_dimension=config.embedding_dimension,
    )
    corpus = _load_verified_corpus(chunks_path, corpus_manifest_path, config)
    if contract is not None:
        _require_retrieval_contract(corpus, contract)
    _validate_profile(artifact_profile, config, shard_size, corpus.chunk_count)

    output_directory.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_incomplete_directories(output_directory)
    started = time.monotonic()
    if contract is not None and (
        _sha256_file(application_queries_path)
        != contract.application_queries_sha256
    ):
        raise ValueError("Dense application queries do not match production identity")
    queries = select_certification_queries(
        chunks_path,
        application_queries_path,
        certification_queries_path,
        application_query_count=application_query_count,
        corpus_query_count=corpus_query_count,
    )
    queries_sha256 = _sha256_file(certification_queries_path)
    if contract is not None and (
        queries_sha256 != contract.recall_queries_sha256
        or len(queries) != contract.recall_query_count
    ):
        raise ValueError("Dense certification queries do not match production identity")
    corpus_hash = corpus.manifest_hash()
    existing = _load_completed_manifest(
        output_manifest_path,
        output_directory,
        corpus_hash=corpus_hash,
        chunks_sha256=corpus.chunks_sha256,
        queries_sha256=queries_sha256,
        query_count=len(queries),
        config=config,
        profile=artifact_profile,
    )
    if existing is not None:
        return ShardedDenseBuildReport(
            elapsed_seconds=time.monotonic() - started,
            corpus_manifest_hash=corpus_hash,
            index_manifest=existing,
        )

    offsets, training_rows = _prepare_offsets_and_training_rows(
        chunks_path,
        output_directory,
        chunk_count=corpus.chunk_count,
        chunks_sha256=corpus.chunks_sha256,
        profile=artifact_profile,
    )
    embedding_encoder = encoder or BgeEncoder(scientific=config)
    checkpoints = _build_embedding_shards(
        chunks_path,
        output_directory,
        chunk_count=corpus.chunk_count,
        chunks_sha256=corpus.chunks_sha256,
        encoder=embedding_encoder,
        profile=artifact_profile,
        semantics_hash=dense_semantics_hash(config),
        shard_size=shard_size,
        batch_size=batch_size,
    )
    template = _train_candidate_template(
        output_directory,
        checkpoints=checkpoints,
        training_rows=training_rows,
        profile=artifact_profile,
    )
    shards = _build_candidate_shards(
        output_directory,
        checkpoints=checkpoints,
        template=template,
        profile=artifact_profile,
        batch_size=batch_size,
    )
    manifest = _publish_certified_manifest(
        output_manifest_path,
        output_directory,
        corpus_hash=corpus_hash,
        chunks_sha256=corpus.chunks_sha256,
        chunk_count=corpus.chunk_count,
        queries=queries,
        queries_sha256=queries_sha256,
        encoder=embedding_encoder,
        offsets=offsets,
        template=template,
        shards=shards,
        profile=artifact_profile,
        config=config,
        recall_block_rows=recall_block_rows,
    )
    shutil.rmtree(output_directory / _CHECKPOINT_ROOT, ignore_errors=True)
    return ShardedDenseBuildReport(
        elapsed_seconds=time.monotonic() - started,
        corpus_manifest_hash=corpus_hash,
        index_manifest=manifest,
    )


def benchmark_sharded_dense_build(
    chunks_path: Path,
    corpus_manifest_path: Path,
    output_directory: Path,
    *,
    encoder: DocumentEncoder | None = None,
    scientific: ScientificConfig | None = None,
    profile: DenseArtifactProfile | None = None,
    sample_rows: int = 100_000,
    batch_size: int = 64,
    contract: BestAvailableRetrievalContract | None = None,
) -> DenseBuildBenchmarkReport:
    started = time.monotonic()
    setup_started = started
    if sample_rows < 1:
        raise ValueError("Dense benchmark sample rows must be positive")
    if batch_size < 1:
        raise ValueError("Dense benchmark batch size must be positive")
    config = scientific or ScientificConfig()
    base_profile = profile or DenseArtifactProfile(
        embedding_dimension=config.embedding_dimension
    )
    corpus = _load_verified_corpus(chunks_path, corpus_manifest_path, config)
    if contract is not None:
        _require_retrieval_contract(corpus, contract)
    if sample_rows > corpus.chunk_count:
        raise ValueError("Dense benchmark sample exceeds the corpus chunk count")
    benchmark_profile = base_profile.model_copy(
        update={
            "training_sample_count": sample_rows,
            "shard_size_limit": max(base_profile.shard_size_limit, sample_rows),
        }
    )
    benchmark_profile = DenseArtifactProfile.model_validate(
        benchmark_profile.model_dump(mode="json")
    )
    if benchmark_profile.embedding_dimension != config.embedding_dimension:
        raise ValueError("Dense benchmark profile dimension mismatch")
    output_directory.mkdir(parents=True, exist_ok=True)
    _reset_peak_gpu_memory()
    offsets_path = output_directory / _OFFSETS_NAME
    offsets = build_chunk_offsets(
        chunks_path,
        offsets_path,
        expected_count=corpus.chunk_count,
        expected_chunks_sha256=corpus.chunks_sha256,
    )
    selected_rows = _select_training_rows(chunks_path, sample_rows)
    embeddings_path = output_directory / "sample-embeddings.i8"
    _require_available_bytes(
        output_directory,
        sample_rows * config.embedding_dimension,
    )
    embeddings = np.memmap(
        embeddings_path,
        dtype=np.int8,
        mode="w+",
        shape=(sample_rows, config.embedding_dimension),
    )
    effective_encoder = encoder or BgeEncoder(scientific=config)
    with ChunkOffsetIndex.open(
        chunks_path,
        offsets_path,
        expected_count=corpus.chunk_count,
        expected_chunks_sha256=corpus.chunks_sha256,
        expected_offsets_sha256=offsets.offsets_sha256,
    ) as chunk_index:
        setup_seconds = time.monotonic() - setup_started
        encoding_started = time.monotonic()
        for start in range(0, sample_rows, batch_size):
            rows = selected_rows[start : start + batch_size]
            chunks = [chunk_index.read(row) for row in rows]
            vectors = _encode_documents(effective_encoder, chunks, config)
            embeddings[start : start + len(vectors)] = quantize_directions(vectors)
        encoding_seconds = max(time.monotonic() - encoding_started, 1e-9)
    embeddings.flush()
    del embeddings
    _fsync_path(embeddings_path)
    training_started = time.monotonic()
    training_vectors = _dequantized_float32_embeddings(
        embeddings_path,
        rows=sample_rows,
        dimension=config.embedding_dimension,
    )
    template_path = output_directory / _TEMPLATE_NAME
    train_ivfpq_template(
        training_vectors,
        template_path,
        dimension=config.embedding_dimension,
        profile=benchmark_profile,
    )
    training_seconds = time.monotonic() - training_started
    candidate = FaissIvfPqIndex.create(template_path, profile=benchmark_profile)
    empty_candidate_path = output_directory / "empty-candidate.faiss"
    candidate.save(empty_candidate_path)
    candidate_add_started = time.monotonic()
    for start in range(0, sample_rows, batch_size):
        end = min(start + batch_size, sample_rows)
        candidate.add(training_vectors[start:end])
    candidate_path = output_directory / "candidate.faiss"
    candidate.save(candidate_path)
    candidate_add_seconds = time.monotonic() - candidate_add_started
    probe_query_count = min(10, sample_rows)
    certification_started = time.monotonic()
    probe_queries = np.ascontiguousarray(
        training_vectors[:probe_query_count],
        dtype=np.float32,
    )
    top_k = min(config.top_chunks_per_list, sample_rows)
    exact_probe: list[list[tuple[float, int]]] = [
        [] for _ in range(probe_query_count)
    ]
    benchmark_embeddings = np.memmap(
        embeddings_path,
        dtype=np.int8,
        mode="r",
        shape=(sample_rows, config.embedding_dimension),
    )
    for start in range(0, sample_rows, 16_384):
        end = min(start + 16_384, sample_rows)
        probe_scores = _matrix_scores(
            dequantize_directions(
                np.asarray(benchmark_embeddings[start:end], dtype=np.int8)
            ),
            probe_queries,
            "auto",
        )
        for query_index in range(probe_query_count):
            values = [
                (float(score), start + row)
                for row, score in enumerate(probe_scores[:, query_index])
            ]
            exact_probe[query_index] = _best_scores(
                (*exact_probe[query_index], *values),
                top_k,
            )
    del benchmark_embeddings
    for query_vector in probe_queries:
        candidate.search(
            query_vector,
            min(benchmark_profile.candidate_count_per_shard, sample_rows),
        )
    certification_probe_seconds = time.monotonic() - certification_started
    elapsed = max(time.monotonic() - started, 1e-9)
    embedding_bytes_per_row = embeddings_path.stat().st_size / sample_rows
    candidate_fixed_bytes = empty_candidate_path.stat().st_size
    candidate_bytes_per_row = max(
        (candidate_path.stat().st_size - candidate_fixed_bytes) / sample_rows,
        1e-9,
    )
    projected_shard_count = math.ceil(
        corpus.chunk_count / base_profile.shard_size_limit
    )
    projected_bytes = math.ceil(
        corpus.chunk_count
        * (embedding_bytes_per_row + candidate_bytes_per_row + 8)
        + candidate_fixed_bytes * projected_shard_count
        + template_path.stat().st_size
    )
    row_scale = corpus.chunk_count / sample_rows
    training_scale = max(base_profile.training_sample_count / sample_rows, 1.0)
    certification_scale = row_scale * (100 / probe_query_count)
    projected_seconds = (
        setup_seconds
        + encoding_seconds * row_scale
        + training_seconds * training_scale
        + candidate_add_seconds * row_scale
        + certification_probe_seconds * certification_scale
    )
    return DenseBuildBenchmarkReport(
        sample_rows=sample_rows,
        production_chunk_count=corpus.chunk_count,
        elapsed_seconds=elapsed,
        setup_seconds=setup_seconds,
        encoding_seconds=encoding_seconds,
        training_seconds=training_seconds,
        candidate_add_seconds=candidate_add_seconds,
        certification_probe_seconds=certification_probe_seconds,
        certification_probe_queries=probe_query_count,
        rows_per_second=sample_rows / encoding_seconds,
        peak_rss_bytes=_peak_rss_bytes(),
        peak_gpu_memory_bytes=_peak_gpu_memory_bytes(),
        embedding_bytes_per_row=embedding_bytes_per_row,
        candidate_bytes_per_row=candidate_bytes_per_row,
        candidate_fixed_bytes_per_shard=candidate_fixed_bytes,
        projected_shard_count=projected_shard_count,
        offset_bytes_per_row=8,
        projected_full_artifact_bytes=projected_bytes,
        projected_full_seconds=projected_seconds,
        projected_gpu_hours=projected_seconds / 3600,
    )


def _prepare_offsets_and_training_rows(
    chunks_path: Path,
    output_directory: Path,
    *,
    chunk_count: int,
    chunks_sha256: str,
    profile: DenseArtifactProfile,
) -> tuple[ChunkOffsetsReport, TrainingRowsManifest]:
    offsets_path = output_directory / _OFFSETS_NAME
    offsets_report_path = output_directory / _OFFSETS_REPORT_NAME
    if offsets_path.exists() != offsets_report_path.exists():
        offsets_path.unlink(missing_ok=True)
        offsets_report_path.unlink(missing_ok=True)
    if offsets_path.exists() and offsets_report_path.exists():
        if not offsets_path.is_file() or not offsets_report_path.is_file():
            raise ValueError("Invalid shared chunk offset artifacts")
        offsets = ChunkOffsetsReport.model_validate_json(
            offsets_report_path.read_text(encoding="utf-8")
        )
        if (
            offsets.chunk_count != chunk_count
            or offsets.chunks_sha256 != chunks_sha256
            or offsets.offsets_size_bytes != chunk_count * 8
        ):
            raise ValueError("Shared chunk offset report is incompatible")
        with ChunkOffsetIndex.open(
            chunks_path,
            offsets_path,
            expected_count=chunk_count,
            expected_chunks_sha256=chunks_sha256,
            expected_offsets_sha256=offsets.offsets_sha256,
        ):
            pass
    else:
        _require_available_bytes(output_directory, chunk_count * 8)
        offsets = build_chunk_offsets(
            chunks_path,
            offsets_path,
            expected_count=chunk_count,
            expected_chunks_sha256=chunks_sha256,
        )
        _write_json_atomic(
            offsets_report_path,
            offsets.model_dump(mode="json"),
        )

    rows_path = output_directory / _TRAINING_ROWS_NAME
    rows_manifest_path = output_directory / _TRAINING_ROWS_MANIFEST_NAME
    if rows_path.exists() != rows_manifest_path.exists():
        rows_path.unlink(missing_ok=True)
        rows_manifest_path.unlink(missing_ok=True)
    if rows_path.exists() and rows_manifest_path.exists():
        if not rows_path.is_file() or not rows_manifest_path.is_file():
            raise ValueError("Invalid dense training row artifacts")
        training = TrainingRowsManifest.model_validate_json(
            rows_manifest_path.read_text(encoding="utf-8")
        )
        _verify_training_rows(
            rows_path,
            training,
            chunk_count=chunk_count,
            chunks_sha256=chunks_sha256,
            profile=profile,
        )
    else:
        rows = _select_training_rows(chunks_path, profile.training_sample_count)
        encoded = np.asarray(rows, dtype="<u8").tobytes()
        _write_bytes_atomic(rows_path, encoded)
        training = TrainingRowsManifest(
            training_selection_hash=dense_training_selection_hash(profile),
            chunks_sha256=chunks_sha256,
            chunk_count=chunk_count,
            sample_count=len(rows),
            selection_version=profile.training_selection_version,
            rows_size_bytes=len(encoded),
            rows_sha256=hashlib.sha256(encoded).hexdigest(),
        )
        _write_json_atomic(
            rows_manifest_path,
            training.model_dump(mode="json"),
        )
    return offsets, training


def _build_embedding_shards(
    chunks_path: Path,
    output_directory: Path,
    *,
    chunk_count: int,
    chunks_sha256: str,
    encoder: DocumentEncoder,
    profile: DenseArtifactProfile,
    semantics_hash: str,
    shard_size: int,
    batch_size: int,
) -> list[EmbeddingShardCheckpoint]:
    checkpoint_root = output_directory / _CHECKPOINT_ROOT
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    chunks = _read_chunks(chunks_path)
    checkpoints: list[EmbeddingShardCheckpoint] = []
    start_row = 0
    shard_id = 0
    while start_row < chunk_count:
        count = min(shard_size, chunk_count - start_row)
        name = f"shard-{shard_id:05d}"
        checkpoint_directory = checkpoint_root / name
        checkpoint = _load_embedding_checkpoint(
            checkpoint_directory,
            expected_id=shard_id,
            expected_start=start_row,
            expected_count=count,
            chunks_sha256=chunks_sha256,
            profile=profile,
            semantics_hash=semantics_hash,
        )
        if checkpoint is None:
            checkpoint = _checkpoint_from_final_shard(
                output_directory,
                checkpoint_directory,
                shard_id=shard_id,
                start_row=start_row,
                chunk_count=count,
                chunks_sha256=chunks_sha256,
                profile=profile,
                semantics_hash=semantics_hash,
            )
        if checkpoint is not None:
            _consume_chunks(chunks, count)
        else:
            checkpoint = _encode_embedding_checkpoint(
                chunks,
                checkpoint_root,
                name=name,
                shard_id=shard_id,
                start_row=start_row,
                chunk_count=count,
                chunks_sha256=chunks_sha256,
                encoder=encoder,
                profile=profile,
                semantics_hash=semantics_hash,
                batch_size=batch_size,
            )
        checkpoints.append(checkpoint)
        start_row = checkpoint.end_row
        shard_id += 1
    try:
        next(chunks)
    except StopIteration:
        pass
    else:
        raise ValueError("Dense chunk count exceeds the corpus manifest")
    return checkpoints


def _train_candidate_template(
    output_directory: Path,
    *,
    checkpoints: list[EmbeddingShardCheckpoint],
    training_rows: TrainingRowsManifest,
    profile: DenseArtifactProfile,
) -> TrainedTemplateManifest:
    template_path = output_directory / _TEMPLATE_NAME
    manifest_path = output_directory / _TEMPLATE_MANIFEST_NAME
    if template_path.exists() != manifest_path.exists():
        template_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    if template_path.exists() and manifest_path.exists():
        if not template_path.is_file() or not manifest_path.is_file():
            raise ValueError("Invalid trained candidate template artifacts")
        manifest = TrainedTemplateManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            manifest.faiss_layout_hash != faiss_layout_hash(profile)
            or manifest.training_rows_sha256 != training_rows.rows_sha256
            or manifest.size_bytes != template_path.stat().st_size
            or manifest.sha256 != _sha256_file(template_path)
        ):
            template_path.unlink()
            manifest_path.unlink()
        else:
            FaissIvfPqIndex.create(template_path, profile=profile)
            return manifest

    rows = np.fromfile(
        output_directory / _TRAINING_ROWS_NAME,
        dtype="<u8",
    ).astype(np.int64)
    training_vectors = np.empty(
        (len(rows), profile.embedding_dimension),
        dtype=np.float32,
    )
    output_start = 0
    for checkpoint in checkpoints:
        output_end = int(np.searchsorted(rows, checkpoint.end_row, side="left"))
        if output_end == output_start:
            continue
        embeddings = np.memmap(
            output_directory
            / _CHECKPOINT_ROOT
            / checkpoint.path
            / profile.embedding_filename,
            dtype=np.int8,
            mode="r",
            shape=(checkpoint.chunk_count, profile.embedding_dimension),
        )
        local_rows = rows[output_start:output_end] - checkpoint.start_row
        training_vectors[output_start:output_end] = dequantize_directions(
            np.asarray(embeddings[local_rows], dtype=np.int8)
        )
        del embeddings
        output_start = output_end
    if output_start != len(rows):
        raise ValueError("Dense training rows fall outside embedding checkpoints")
    _require_available_bytes(output_directory, max(training_vectors.nbytes // 16, 1))
    train_ivfpq_template(
        training_vectors,
        template_path,
        dimension=profile.embedding_dimension,
        profile=profile,
    )
    manifest = TrainedTemplateManifest(
        faiss_layout_hash=faiss_layout_hash(profile),
        training_rows_sha256=training_rows.rows_sha256,
        size_bytes=template_path.stat().st_size,
        sha256=_sha256_file(template_path),
    )
    _write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    return manifest


def _build_candidate_shards(
    output_directory: Path,
    *,
    checkpoints: list[EmbeddingShardCheckpoint],
    template: TrainedTemplateManifest,
    profile: DenseArtifactProfile,
    batch_size: int,
) -> list[CompressedDenseShardManifest]:
    result: list[CompressedDenseShardManifest] = []
    for checkpoint in checkpoints:
        final_directory = output_directory / f"shard-{checkpoint.shard_id:05d}"
        existing = _load_completed_candidate_shard(
            final_directory,
            checkpoint=checkpoint,
            template=template,
            profile=profile,
        )
        if existing is not None:
            result.append(existing)
            continue
        candidate_estimate = max(
            template.size_bytes,
            checkpoint.chunk_count
            * (math.ceil(profile.pq_subvectors * profile.pq_bits / 8) + 8),
        )
        _require_available_bytes(output_directory, candidate_estimate)
        temporary = Path(
            tempfile.mkdtemp(
                dir=output_directory,
                prefix=f".shard-{checkpoint.shard_id:05d}.",
            )
        )
        try:
            source_embeddings = (
                output_directory
                / _CHECKPOINT_ROOT
                / checkpoint.path
                / profile.embedding_filename
            )
            embeddings_path = temporary / profile.embedding_filename
            os.link(source_embeddings, embeddings_path)
            candidate = FaissIvfPqIndex.create(
                output_directory / template.path,
                profile=profile,
            )
            embeddings = np.memmap(
                embeddings_path,
                dtype=np.int8,
                mode="r",
                shape=(checkpoint.chunk_count, profile.embedding_dimension),
            )
            for start in range(0, checkpoint.chunk_count, batch_size):
                end = min(start + batch_size, checkpoint.chunk_count)
                candidate.add(
                    dequantize_directions(
                        np.asarray(embeddings[start:end], dtype=np.int8)
                    )
                )
            del embeddings
            candidate_path = temporary / "candidate.faiss"
            candidate.save(candidate_path)
            manifest = CompressedDenseShardManifest(
                artifact_profile_hash=profile.manifest_hash(),
                candidate_template_sha256=template.sha256,
                shard_id=checkpoint.shard_id,
                path=final_directory.name,
                start_row=checkpoint.start_row,
                end_row=checkpoint.end_row,
                chunk_count=checkpoint.chunk_count,
                embeddings_size_bytes=checkpoint.embeddings_size_bytes,
                embeddings_sha256=checkpoint.embeddings_sha256,
                candidate_index_size_bytes=candidate_path.stat().st_size,
                candidate_index_sha256=_sha256_file(candidate_path),
            )
            _write_json_atomic(
                temporary / "manifest.json",
                manifest.model_dump(mode="json"),
            )
            _replace_directory(temporary, final_directory, output_directory)
            result.append(manifest)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return result


def _publish_certified_manifest(
    output_manifest_path: Path,
    output_directory: Path,
    *,
    corpus_hash: str,
    chunks_sha256: str,
    chunk_count: int,
    queries: list[str],
    queries_sha256: str,
    encoder: DocumentEncoder,
    offsets: ChunkOffsetsReport,
    template: TrainedTemplateManifest,
    shards: list[CompressedDenseShardManifest],
    profile: DenseArtifactProfile,
    config: ScientificConfig,
    recall_block_rows: int,
) -> CompressedShardedDenseIndexManifest:
    per_query_recall = _certify_candidate_recall(
        output_directory,
        shards,
        queries,
        encoder=encoder,
        profile=profile,
        config=config,
        block_rows=recall_block_rows,
    )
    recall = float(math.fsum(per_query_recall) / len(per_query_recall))
    require_dense_recall(recall, minimum=config.dense_recall_minimum)
    recall_report = DenseRecallBuildReport(
        artifact_profile_hash=profile.manifest_hash(),
        corpus_manifest_hash=corpus_hash,
        recall_queries_sha256=queries_sha256,
        recall_query_count=len(queries),
        recall_at_300=recall,
        per_query_recall=per_query_recall,
    )
    recall_path = output_directory / _RECALL_REPORT_NAME
    _write_json_atomic(recall_path, recall_report.model_dump(mode="json"))
    total_bytes = (
        offsets.offsets_size_bytes
        + template.size_bytes
        + recall_path.stat().st_size
        + sum(
            shard.embeddings_size_bytes + shard.candidate_index_size_bytes
            for shard in shards
        )
    )
    if total_bytes > profile.max_artifact_bytes:
        raise StorageBudgetError(
            "Dense final artifacts exceed the profile storage ceiling: "
            f"required={total_bytes} available={profile.max_artifact_bytes} "
            f"path={output_directory}"
        )
    manifest = CompressedShardedDenseIndexManifest(
        model_name=config.embedding_model,
        model_revision=config.embedding_model_revision,
        embedding_dimension=config.embedding_dimension,
        corpus_manifest_hash=corpus_hash,
        dense_semantics_hash=dense_semantics_hash(config),
        chunks_sha256=chunks_sha256,
        chunk_count=chunk_count,
        artifact_profile=profile,
        artifact_profile_hash=profile.manifest_hash(),
        chunk_offsets_path=_OFFSETS_NAME,
        chunk_offsets_format=profile.metadata_lookup_version,
        chunk_offsets_size_bytes=offsets.offsets_size_bytes,
        chunk_offsets_sha256=offsets.offsets_sha256,
        candidate_template_path=template.path,
        candidate_template_size_bytes=template.size_bytes,
        candidate_template_sha256=template.sha256,
        shard_size_limit=profile.shard_size_limit,
        shards=shards,
        embedding_file=profile.embedding_filename,
        dtype=profile.embedding_dtype,
        metric=profile.metric,
        candidate_index=profile.candidate_index,
        recall_report_path=_RECALL_REPORT_NAME,
        recall_report_size_bytes=recall_path.stat().st_size,
        recall_report_sha256=_sha256_file(recall_path),
        recall_at_300=recall,
        recall_queries_sha256=queries_sha256,
        recall_query_count=len(queries),
        total_artifact_bytes=total_bytes,
    )
    _write_json_atomic(output_manifest_path, manifest.model_dump(mode="json"))
    return manifest


def _certify_candidate_recall(
    output_directory: Path,
    shards: list[CompressedDenseShardManifest],
    queries: list[str],
    *,
    encoder: DocumentEncoder,
    profile: DenseArtifactProfile,
    config: ScientificConfig,
    block_rows: int,
) -> list[float]:
    query_vectors = np.asarray(
        [_encode_query(encoder, query, config) for query in queries],
        dtype=np.float32,
    )
    limit = min(config.top_chunks_per_list, sum(shard.chunk_count for shard in shards))
    exact: list[list[tuple[float, int]]] = [[] for _ in queries]
    approximate: list[list[tuple[float, int]]] = [[] for _ in queries]
    for shard in shards:
        directory = output_directory / shard.path
        embeddings = np.memmap(
            directory / profile.embedding_filename,
            dtype=np.int8,
            mode="r",
            shape=(shard.chunk_count, profile.embedding_dimension),
        )
        for start in range(0, shard.chunk_count, block_rows):
            end = min(start + block_rows, shard.chunk_count)
            block = dequantize_directions(
                np.asarray(embeddings[start:end], dtype=np.int8)
            )
            scores = _matrix_scores(block, query_vectors, "auto")
            for query_index in range(len(queries)):
                values = [
                    (float(score), shard.start_row + start + row)
                    for row, score in enumerate(scores[:, query_index])
                ]
                exact[query_index] = _best_scores(
                    (*exact[query_index], *values),
                    limit,
                )
        candidate = FaissIvfPqIndex.load(
            directory / "candidate.faiss",
            profile=profile,
        )
        for query_index, query_vector in enumerate(query_vectors):
            rows = candidate.search(
                query_vector,
                min(profile.candidate_count_per_shard, shard.chunk_count),
            )
            scores = dequantize_directions(
                np.asarray(embeddings[rows], dtype=np.int8)
            ) @ query_vector
            values = [
                (float(score), shard.start_row + int(row))
                for row, score in zip(rows, scores, strict=True)
            ]
            approximate[query_index] = _best_scores(
                (*approximate[query_index], *values),
                limit,
            )
        del embeddings
    recalls: list[float] = []
    for exact_values, approximate_values in zip(exact, approximate, strict=True):
        if len(approximate_values) != limit:
            raise ValueError(
                "Compressed dense recall did not return a complete result list"
            )
        expected = {row for _, row in exact_values}
        returned = {row for _, row in approximate_values}
        recalls.append(len(expected & returned) / len(expected))
    return recalls


def _load_verified_corpus(
    chunks_path: Path,
    corpus_manifest_path: Path,
    config: ScientificConfig,
) -> BestAvailableCorpusManifest:
    corpus = BestAvailableCorpusManifest.model_validate_json(
        corpus_manifest_path.read_text(encoding="utf-8")
    )
    if _sha256_file(chunks_path) != corpus.chunks_sha256:
        raise ValueError("Corpus chunk checksum does not match its manifest")
    if corpus.scientific_hash != config.manifest_hash():
        raise ValueError("Corpus scientific manifest does not match dense configuration")
    if corpus.paper_coverage_ready:
        raise ValueError("Best-available corpus must not claim exact-paper coverage")
    if corpus.chunk_count < 1:
        raise ValueError("Dense index cannot be built from an empty corpus")
    return corpus


def _require_retrieval_contract(
    corpus: BestAvailableCorpusManifest,
    contract: BestAvailableRetrievalContract,
) -> None:
    if (
        corpus.manifest_hash() != contract.corpus_manifest_hash
        or corpus.chunks_sha256 != contract.chunks_sha256
        or corpus.parsed_row_count != contract.parsed_row_count
        or corpus.document_count != contract.document_count
        or corpus.chunk_count != contract.chunk_count
    ):
        raise ValueError("Best-available production corpus identity mismatch")


def _validate_profile(
    profile: DenseArtifactProfile,
    config: ScientificConfig,
    shard_size: int,
    chunk_count: int,
) -> None:
    if profile.embedding_dimension != config.embedding_dimension:
        raise ValueError("Dense artifact profile dimension mismatch")
    if shard_size > profile.shard_size_limit:
        raise ValueError("Dense shard size exceeds the artifact profile maximum")
    if profile.training_sample_count > chunk_count:
        raise ValueError("Dense training sample exceeds the corpus chunk count")


def _load_embedding_checkpoint(
    directory: Path,
    *,
    expected_id: int,
    expected_start: int,
    expected_count: int,
    chunks_sha256: str,
    profile: DenseArtifactProfile,
    semantics_hash: str,
) -> EmbeddingShardCheckpoint | None:
    if not directory.exists():
        return None
    manifest_path = directory / "manifest.json"
    embeddings_path = directory / profile.embedding_filename
    if not manifest_path.is_file() or not embeddings_path.is_file():
        raise ValueError(f"Incomplete embedding checkpoint: {directory}")
    checkpoint = EmbeddingShardCheckpoint.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if (
        checkpoint.dense_semantics_hash != semantics_hash
        or checkpoint.chunks_sha256 != chunks_sha256
        or checkpoint.shard_id != expected_id
        or checkpoint.path != directory.name
        or checkpoint.start_row != expected_start
        or checkpoint.chunk_count != expected_count
        or checkpoint.end_row != expected_start + expected_count
        or checkpoint.embeddings_size_bytes
        != expected_count * profile.embedding_dimension
    ):
        raise ValueError("Completed embedding checkpoint is incompatible")
    if (
        embeddings_path.stat().st_size != checkpoint.embeddings_size_bytes
        or _sha256_file(embeddings_path) != checkpoint.embeddings_sha256
    ):
        raise ValueError("embedding checkpoint checksum mismatch")
    return checkpoint


def _checkpoint_from_final_shard(
    output_directory: Path,
    checkpoint_directory: Path,
    *,
    shard_id: int,
    start_row: int,
    chunk_count: int,
    chunks_sha256: str,
    profile: DenseArtifactProfile,
    semantics_hash: str,
) -> EmbeddingShardCheckpoint | None:
    final_directory = output_directory / f"shard-{shard_id:05d}"
    if not final_directory.exists():
        return None
    manifest_path = final_directory / "manifest.json"
    embeddings_path = final_directory / profile.embedding_filename
    if not manifest_path.is_file() or not embeddings_path.is_file():
        raise ValueError(f"Incomplete dense shard directory: {final_directory}")
    shard = CompressedDenseShardManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if (
        shard.shard_id != shard_id
        or shard.start_row != start_row
        or shard.chunk_count != chunk_count
        or shard.embeddings_size_bytes != chunk_count * profile.embedding_dimension
        or embeddings_path.stat().st_size != shard.embeddings_size_bytes
        or _sha256_file(embeddings_path) != shard.embeddings_sha256
    ):
        raise ValueError("Completed dense shard embedding checksum mismatch")
    temporary = Path(
        tempfile.mkdtemp(
            dir=checkpoint_directory.parent,
            prefix=f".{checkpoint_directory.name}.",
        )
    )
    try:
        os.link(embeddings_path, temporary / profile.embedding_filename)
        checkpoint = EmbeddingShardCheckpoint(
            dense_semantics_hash=semantics_hash,
            chunks_sha256=chunks_sha256,
            shard_id=shard_id,
            path=checkpoint_directory.name,
            start_row=start_row,
            end_row=start_row + chunk_count,
            chunk_count=chunk_count,
            embeddings_size_bytes=shard.embeddings_size_bytes,
            embeddings_sha256=shard.embeddings_sha256,
        )
        _write_json_atomic(
            temporary / "manifest.json",
            checkpoint.model_dump(mode="json"),
        )
        os.replace(temporary, checkpoint_directory)
        _fsync_directory(checkpoint_directory.parent)
        return checkpoint
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _encode_embedding_checkpoint(
    chunks: Iterator[CorpusChunk],
    checkpoint_root: Path,
    *,
    name: str,
    shard_id: int,
    start_row: int,
    chunk_count: int,
    chunks_sha256: str,
    encoder: DocumentEncoder,
    profile: DenseArtifactProfile,
    semantics_hash: str,
    batch_size: int,
) -> EmbeddingShardCheckpoint:
    required_bytes = chunk_count * profile.embedding_dimension
    _require_available_bytes(checkpoint_root, required_bytes)
    temporary = Path(
        tempfile.mkdtemp(
            dir=checkpoint_root,
            prefix=f".{name}.",
        )
    )
    final_directory = checkpoint_root / name
    try:
        embeddings_path = temporary / profile.embedding_filename
        embeddings = np.memmap(
            embeddings_path,
            dtype=np.int8,
            mode="w+",
            shape=(chunk_count, profile.embedding_dimension),
        )
        row = 0
        while row < chunk_count:
            count = min(batch_size, chunk_count - row)
            batch = _take_chunks(chunks, count)
            vectors = _encode_documents(
                encoder,
                batch,
                ScientificConfig(embedding_dimension=profile.embedding_dimension),
            )
            embeddings[row : row + len(batch)] = quantize_directions(vectors)
            row += len(batch)
        embeddings.flush()
        del embeddings
        _fsync_path(embeddings_path)
        checkpoint = EmbeddingShardCheckpoint(
            dense_semantics_hash=semantics_hash,
            chunks_sha256=chunks_sha256,
            shard_id=shard_id,
            path=name,
            start_row=start_row,
            end_row=start_row + chunk_count,
            chunk_count=chunk_count,
            embeddings_size_bytes=embeddings_path.stat().st_size,
            embeddings_sha256=_sha256_file(embeddings_path),
        )
        _write_json_atomic(
            temporary / "manifest.json",
            checkpoint.model_dump(mode="json"),
        )
        os.replace(temporary, final_directory)
        _fsync_directory(checkpoint_root)
        return checkpoint
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_completed_candidate_shard(
    directory: Path,
    *,
    checkpoint: EmbeddingShardCheckpoint,
    template: TrainedTemplateManifest,
    profile: DenseArtifactProfile,
) -> CompressedDenseShardManifest | None:
    if not directory.exists():
        return None
    manifest_path = directory / "manifest.json"
    embeddings_path = directory / profile.embedding_filename
    candidate_path = directory / "candidate.faiss"
    if (
        not manifest_path.is_file()
        or not embeddings_path.is_file()
        or not candidate_path.is_file()
    ):
        return None
    shard = CompressedDenseShardManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if (
        shard.artifact_profile_hash != profile.manifest_hash()
        or shard.candidate_template_sha256 != template.sha256
    ):
        return None
    if (
        shard.shard_id != checkpoint.shard_id
        or shard.path != directory.name
        or shard.start_row != checkpoint.start_row
        or shard.chunk_count != checkpoint.chunk_count
        or shard.embeddings_sha256 != checkpoint.embeddings_sha256
        or shard.embeddings_size_bytes != checkpoint.embeddings_size_bytes
    ):
        raise ValueError("Completed compressed dense shard is incompatible")
    if (
        candidate_path.stat().st_size != shard.candidate_index_size_bytes
        or _sha256_file(candidate_path) != shard.candidate_index_sha256
    ):
        return None
    candidate = FaissIvfPqIndex.load(candidate_path, profile=profile)
    if candidate.ntotal != checkpoint.chunk_count:
        return None
    return shard


def _load_completed_manifest(
    path: Path,
    output_directory: Path,
    *,
    corpus_hash: str,
    chunks_sha256: str,
    queries_sha256: str,
    query_count: int,
    config: ScientificConfig,
    profile: DenseArtifactProfile,
) -> CompressedShardedDenseIndexManifest | None:
    if not path.exists():
        return None
    manifest = CompressedShardedDenseIndexManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if manifest.artifact_profile_hash != profile.manifest_hash():
        return None
    if (
        manifest.corpus_manifest_hash != corpus_hash
        or manifest.chunks_sha256 != chunks_sha256
        or manifest.recall_queries_sha256 != queries_sha256
        or manifest.recall_query_count != query_count
        or manifest.model_name != config.embedding_model
        or manifest.model_revision != config.embedding_model_revision
        or manifest.dense_semantics_hash != dense_semantics_hash(config)
    ):
        raise ValueError("Completed compressed dense manifest is incompatible")
    _repair_missing_shard_manifests(output_directory, manifest)
    try:
        _verify_published_artifacts(output_directory, manifest)
        require_dense_recall(
            manifest.recall_at_300,
            minimum=config.dense_recall_minimum,
        )
    except (OSError, ValueError):
        return None
    return manifest


def _repair_missing_shard_manifests(
    root: Path,
    manifest: CompressedShardedDenseIndexManifest,
) -> None:
    for shard in manifest.shards:
        manifest_path = root / shard.path / "manifest.json"
        if manifest_path.exists():
            continue
        embeddings_path = root / shard.path / manifest.embedding_file
        if (
            not embeddings_path.is_file()
            or embeddings_path.stat().st_size != shard.embeddings_size_bytes
            or _sha256_file(embeddings_path) != shard.embeddings_sha256
        ):
            return
        _write_json_atomic(manifest_path, shard.model_dump(mode="json"))


def _verify_published_artifacts(
    root: Path,
    manifest: CompressedShardedDenseIndexManifest,
) -> None:
    files = [
        (
            root / manifest.chunk_offsets_path,
            manifest.chunk_offsets_size_bytes,
            manifest.chunk_offsets_sha256,
        ),
        (
            root / manifest.candidate_template_path,
            manifest.candidate_template_size_bytes,
            manifest.candidate_template_sha256,
        ),
        (
            root / manifest.recall_report_path,
            manifest.recall_report_size_bytes,
            manifest.recall_report_sha256,
        ),
    ]
    for shard in manifest.shards:
        shard_manifest_path = root / shard.path / "manifest.json"
        if not shard_manifest_path.is_file():
            raise ValueError(
                f"Published dense shard manifest is missing: {shard_manifest_path}"
            )
        try:
            on_disk_shard = CompressedDenseShardManifest.model_validate_json(
                shard_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Published dense shard manifest is invalid: {shard_manifest_path}"
            ) from error
        if on_disk_shard != shard:
            raise ValueError(
                f"Published dense shard manifest mismatch: {shard_manifest_path}"
            )
        files.extend(
            [
                (
                    root / shard.path / manifest.embedding_file,
                    shard.embeddings_size_bytes,
                    shard.embeddings_sha256,
                ),
                (
                    root / shard.path / "candidate.faiss",
                    shard.candidate_index_size_bytes,
                    shard.candidate_index_sha256,
                ),
            ]
        )
    for artifact, expected_size, expected_hash in files:
        if not artifact.is_file():
            raise ValueError(f"Published dense artifact is missing: {artifact}")
        if artifact.stat().st_size != expected_size:
            raise ValueError(f"Published dense artifact size mismatch: {artifact}")
        if _sha256_file(artifact) != expected_hash:
            raise ValueError(f"Published dense artifact checksum mismatch: {artifact}")


def _verify_training_rows(
    path: Path,
    manifest: TrainingRowsManifest,
    *,
    chunk_count: int,
    chunks_sha256: str,
    profile: DenseArtifactProfile,
) -> None:
    if (
        manifest.training_selection_hash != dense_training_selection_hash(profile)
        or manifest.chunks_sha256 != chunks_sha256
        or manifest.chunk_count != chunk_count
        or manifest.sample_count != profile.training_sample_count
        or manifest.selection_version != profile.training_selection_version
        or manifest.rows_size_bytes != manifest.sample_count * 8
        or path.stat().st_size != manifest.rows_size_bytes
        or _sha256_file(path) != manifest.rows_sha256
    ):
        raise ValueError("Dense training row checksum or profile mismatch")
    rows = np.fromfile(path, dtype="<u8")
    if (
        len(rows) != manifest.sample_count
        or np.any(rows >= chunk_count)
        or np.any(rows[1:] <= rows[:-1])
    ):
        raise ValueError("Dense training row values are invalid")


def _select_training_rows(chunks_path: Path, count: int) -> list[int]:
    selected = heapq.nsmallest(
        count,
        (
            (
                hashlib.sha256(f"{chunk.id}:{row}".encode("utf-8")).digest(),
                row,
            )
            for row, chunk in enumerate(_read_chunks(chunks_path))
        ),
        key=lambda value: value,
    )
    if len(selected) != count:
        raise ValueError("Dense training sample exceeds available chunks")
    return sorted(row for _digest, row in selected)


def _encode_documents(
    encoder: DocumentEncoder,
    chunks: Sequence[CorpusChunk],
    config: ScientificConfig,
) -> NDArray[np.float32]:
    vectors = np.asarray(
        encoder.encode_documents([chunk.text for chunk in chunks]),
        dtype=np.float32,
    )
    expected_shape = (len(chunks), config.embedding_dimension)
    if vectors.shape != expected_shape:
        raise ValueError(
            f"Dense encoder returned shape {vectors.shape}; expected {expected_shape}"
        )
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(vectors)) or not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Dense document embeddings must be finite and L2-normalized")
    return vectors


def _encode_query(
    encoder: DocumentEncoder,
    query: str,
    config: ScientificConfig,
) -> NDArray[np.float32]:
    vector = np.asarray(encoder.encode_query(query), dtype=np.float32)
    if vector.shape != (config.embedding_dimension,):
        raise ValueError("Dense query embedding dimension mismatch")
    if np.any(~np.isfinite(vector)) or not np.isclose(
        np.linalg.norm(vector),
        1.0,
        atol=1e-4,
    ):
        raise ValueError("Dense query embedding must be finite and L2-normalized")
    return vector


def _dequantized_float32_embeddings(
    path: Path,
    *,
    rows: int,
    dimension: int,
) -> NDArray[np.float32]:
    embeddings = np.memmap(
        path,
        dtype=np.int8,
        mode="r",
        shape=(rows, dimension),
    )
    values = np.asarray(embeddings, dtype=np.int8)
    del embeddings
    return dequantize_directions(values)


def _best_scores(
    values: Sequence[tuple[float, int]],
    limit: int,
) -> list[tuple[float, int]]:
    return sorted(values, key=lambda value: (-value[0], value[1]))[:limit]


def _replace_directory(
    temporary: Path,
    final: Path,
    root: Path,
) -> None:
    if not final.exists():
        os.replace(temporary, final)
        _fsync_directory(root)
        return
    backup = root / f".replaced-{final.name}"
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(final, backup)
    try:
        os.replace(temporary, final)
        _fsync_directory(root)
    except BaseException:
        if not final.exists() and backup.exists():
            os.replace(backup, final)
        raise
    shutil.rmtree(backup)


def _cleanup_incomplete_directories(root: Path) -> None:
    paths = list(root.iterdir())
    checkpoint_root = root / _CHECKPOINT_ROOT
    if checkpoint_root.is_dir():
        paths.extend(checkpoint_root.iterdir())
    for path in paths:
        if path.is_dir() and (
            path.name.startswith(".shard-")
            or path.name.startswith(".replaced-shard-")
            or path.name.startswith(".embedding-")
        ):
            shutil.rmtree(path)


def _read_chunks(path: Path) -> Iterator[CorpusChunk]:
    with path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"Blank corpus chunk at line {line_number}")
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


@contextmanager
def _exclusive_build_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".compressed-dense-build.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another compressed dense build owns {root}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _require_available_bytes(path: Path, required: int) -> None:
    available = _available_bytes(path)
    if available < required:
        raise StorageBudgetError(
            "Insufficient dense storage: "
            f"required={required} available={available} path={path}"
        )


def _available_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _peak_rss_bytes() -> int | None:
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):
        return None
    if value < 0:
        return None
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def _reset_peak_gpu_memory() -> None:
    if not _cuda_runtime_visible():
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_memory_bytes() -> int | None:
    if not _cuda_runtime_visible():
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    _write_bytes_atomic(
        path,
        (
            json.dumps(value, sort_keys=True, indent=2, separators=(",", ": "))
            + "\n"
        ).encode("utf-8"),
    )


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            destination.write(value)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
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
