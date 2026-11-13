from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.retrieval.dense import BgeEncoder, require_dense_recall
from redacted_backend.retrieval.types import DenseIndexManifest
from redacted_backend.scientific_config import ScientificConfig


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


class DenseBuildReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    elapsed_seconds: float = Field(ge=0)
    corpus_manifest_hash: str
    index_manifest: DenseIndexManifest


def build_dense_from_files(
    chunks_path: Path,
    corpus_manifest_path: Path,
    output_directory: Path,
    output_manifest_path: Path,
    *,
    recall_queries: Sequence[str],
    encoder: DocumentEncoder | None = None,
    candidate_index: CandidateIndexBuilder | None = None,
    scientific: ScientificConfig | None = None,
    batch_size: int = 64,
) -> DenseBuildReport:
    if batch_size < 1:
        raise ValueError("Dense encoding batch size must be positive")
    if not recall_queries or any(not query.strip() for query in recall_queries):
        raise ValueError("Dense recall queries must contain visible, non-empty text")
    config = scientific or ScientificConfig()
    corpus = CorpusManifest.model_validate_json(
        corpus_manifest_path.read_text(encoding="utf-8")
    )
    chunks_sha256 = _sha256_file(chunks_path)
    if chunks_sha256 != corpus.chunks_sha256:
        raise ValueError("Corpus chunk checksum does not match its manifest")
    if not corpus.coverage_ready:
        raise ValueError("Corpus coverage is not ready for dense indexing")
    if corpus.scientific_hash != config.manifest_hash():
        raise ValueError("Corpus scientific manifest does not match dense configuration")

    output_directory.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "embeddings": output_directory / "embeddings.f16",
        "metadata": output_directory / "metadata.jsonl",
        "candidate": output_directory / "candidate.faiss",
    }
    embedding_encoder = encoder or BgeEncoder(scientific=config)
    index = candidate_index or FaissHnswIndex.create(config)
    started = time.monotonic()

    temporary_embeddings = _temporary_path(paths["embeddings"])
    temporary_metadata = _temporary_path(paths["metadata"])
    temporary_candidate = _temporary_path(paths["candidate"])
    temporary_paths = {
        temporary_embeddings,
        temporary_metadata,
        temporary_candidate,
    }
    chunk_ids: list[str] = []
    try:
        matrix = np.memmap(
            temporary_embeddings,
            dtype=np.float16,
            mode="w+",
            shape=(corpus.chunk_count, config.embedding_dimension),
        )
        row = 0
        with temporary_metadata.open("wb") as metadata:
            batch: list[CorpusChunk] = []
            for chunk in _read_chunks(chunks_path):
                batch.append(chunk)
                if len(batch) >= batch_size:
                    row = _encode_batch(
                        batch,
                        row,
                        matrix,
                        metadata,
                        chunk_ids,
                        embedding_encoder,
                        index,
                        config,
                    )
                    batch = []
            if batch:
                row = _encode_batch(
                    batch,
                    row,
                    matrix,
                    metadata,
                    chunk_ids,
                    embedding_encoder,
                    index,
                    config,
                )
            metadata.flush()
            os.fsync(metadata.fileno())
        if row != corpus.chunk_count:
            raise ValueError("Dense chunk count does not match the corpus manifest")
        matrix.flush()
        del matrix
        index.save(temporary_candidate)

        embeddings = np.memmap(
            temporary_embeddings,
            dtype=np.float16,
            mode="r",
            shape=(corpus.chunk_count, config.embedding_dimension),
        )
        recall = _measure_index_recall(
            recall_queries,
            embedding_encoder,
            index,
            embeddings,
            chunk_ids,
            config,
        )
        require_dense_recall(recall, minimum=config.dense_recall_minimum)
        del embeddings

        for temporary, destination in (
            (temporary_embeddings, paths["embeddings"]),
            (temporary_metadata, paths["metadata"]),
            (temporary_candidate, paths["candidate"]),
        ):
            os.replace(temporary, destination)
            temporary_paths.discard(temporary)

        manifest = DenseIndexManifest(
            model_name=config.embedding_model,
            model_revision=config.embedding_model_revision,
            embedding_dimension=config.embedding_dimension,
            corpus_manifest_hash=corpus.manifest_hash(),
            scientific_hash=config.manifest_hash(),
            chunks_sha256=chunks_sha256,
            embeddings_sha256=_sha256_file(paths["embeddings"]),
            metadata_sha256=_sha256_file(paths["metadata"]),
            candidate_index_sha256=_sha256_file(paths["candidate"]),
            chunk_count=corpus.chunk_count,
            dtype="float16",
            metric="inner_product",
            candidate_index=config.dense_index_version,
            hnsw_m=config.dense_hnsw_m,
            ef_construction=config.dense_ef_construction,
            ef_search=config.dense_ef_search,
            candidate_count=config.dense_candidate_count,
            recall_at_300=recall,
            recall_sample_sha256=_canonical_hash(list(recall_queries)),
        )
        _write_json_atomic(output_manifest_path, manifest.model_dump(mode="json"))
        return DenseBuildReport(
            elapsed_seconds=time.monotonic() - started,
            corpus_manifest_hash=corpus.manifest_hash(),
            index_manifest=manifest,
        )
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


class FaissHnswIndex:
    def __init__(self, index: object) -> None:
        self._index = index

    @classmethod
    def create(cls, scientific: ScientificConfig) -> FaissHnswIndex:
        faiss = _import_faiss()
        index = faiss.IndexHNSWFlat(
            scientific.embedding_dimension,
            scientific.dense_hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = scientific.dense_ef_construction
        index.hnsw.efSearch = scientific.dense_ef_search
        return cls(index)

    @classmethod
    def load(cls, path: Path, *, ef_search: int) -> FaissHnswIndex:
        faiss = _import_faiss()
        index = faiss.read_index(str(path))
        index.hnsw.efSearch = ef_search
        return cls(index)

    def add(self, vectors: NDArray[np.float32]) -> None:
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(
        self,
        query: NDArray[np.float32],
        candidates: int,
    ) -> NDArray[np.int64]:
        total = int(self._index.ntotal)
        if total == 0:
            return np.empty(0, dtype=np.int64)
        count = min(candidates, total)
        _, rows = self._index.search(
            np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32),
            count,
        )
        return np.asarray(rows[0][rows[0] >= 0], dtype=np.int64)

    def save(self, path: Path) -> None:
        _import_faiss().write_index(self._index, str(path))


def _encode_batch(
    batch: Sequence[CorpusChunk],
    row: int,
    matrix: np.memmap,
    metadata: object,
    chunk_ids: list[str],
    encoder: DocumentEncoder,
    index: CandidateIndexBuilder,
    config: ScientificConfig,
) -> int:
    vectors = np.asarray(
        encoder.encode_documents([chunk.text for chunk in batch]),
        dtype=np.float32,
    )
    expected_shape = (len(batch), config.embedding_dimension)
    if vectors.shape != expected_shape:
        raise ValueError(
            f"Dense encoder returned shape {vectors.shape}; expected {expected_shape}"
        )
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(vectors)) or not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Dense document embeddings must be finite and L2-normalized")
    next_row = row + len(batch)
    if next_row > matrix.shape[0]:
        raise ValueError("Dense chunk count exceeds the corpus manifest")
    matrix[row:next_row] = vectors.astype(np.float16)
    index.add(vectors)
    for chunk in batch:
        serialized = (
            json.dumps(
                chunk.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        metadata.write(serialized)
        chunk_ids.append(chunk.id)
    return next_row


def _measure_index_recall(
    queries: Sequence[str],
    encoder: DocumentEncoder,
    index: CandidateIndexBuilder,
    embeddings: NDArray[np.float16],
    chunk_ids: Sequence[str],
    config: ScientificConfig,
) -> float:
    if len(chunk_ids) == 0:
        raise ValueError("Dense index cannot be built from an empty corpus")
    limit = min(config.top_chunks_per_list, len(chunk_ids))
    candidate_count = min(config.dense_candidate_count, len(chunk_ids))
    recalls: list[float] = []
    float_embeddings = np.asarray(embeddings, dtype=np.float32)
    for query in queries:
        query_vector = np.asarray(encoder.encode_query(query), dtype=np.float32)
        if query_vector.shape != (config.embedding_dimension,):
            raise ValueError("Dense recall query has an incompatible dimension")
        exact_scores = float_embeddings @ query_vector
        exact_rows = _stable_top_rows(exact_scores, range(len(chunk_ids)), chunk_ids, limit)
        candidates = np.asarray(
            index.search(query_vector, candidate_count),
            dtype=np.int64,
        ).reshape(-1)
        if np.any(candidates < 0) or np.any(candidates >= len(chunk_ids)):
            raise ValueError("Dense candidate index returned an invalid row")
        approximate_rows = _stable_top_rows(
            exact_scores,
            np.unique(candidates),
            chunk_ids,
            limit,
        )
        recalls.append(len(set(exact_rows) & set(approximate_rows)) / len(exact_rows))
    return float(sum(recalls) / len(recalls))


def _stable_top_rows(
    scores: NDArray[np.float32],
    rows: Sequence[int],
    chunk_ids: Sequence[str],
    limit: int,
) -> list[int]:
    return sorted(
        (int(row) for row in rows),
        key=lambda row: (-float(scores[row]), chunk_ids[row]),
    )[:limit]


def _read_chunks(path: Path) -> Iterator[CorpusChunk]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield CorpusChunk.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid corpus chunk at line {line_number}") from error


def _temporary_path(destination: Path) -> Path:
    descriptor, value = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
    )
    os.close(descriptor)
    return Path(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(value, destination, sort_keys=True, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _import_faiss():
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError("faiss-cpu is required for real dense indexing") from error
    return faiss
