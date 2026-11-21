from __future__ import annotations

import hashlib
import heapq
import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.dense import (
    IncompatibleDenseIndexError,
    QueryEncoder,
    require_dense_recall,
)
from redacted_backend.retrieval.types import (
    DenseShardManifest,
    RankedChunk,
    ShardedDenseIndexManifest,
)
from redacted_backend.scientific_config import ScientificConfig


class CandidateIndex(Protocol):
    def search(
        self,
        query: NDArray[np.float32],
        candidates: int,
    ) -> NDArray[np.int64]: ...


CandidateLoader = Callable[..., CandidateIndex]


class ShardedDenseIndex:
    def __init__(
        self,
        *,
        root: Path,
        manifest: ShardedDenseIndexManifest,
        encoder: QueryEncoder,
        candidate_loader: CandidateLoader,
        scientific: ScientificConfig | None = None,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.encoder = encoder
        self.candidate_loader = candidate_loader
        self.scientific = scientific or ScientificConfig()
        self._validate()

    def search(
        self,
        query: str,
        candidates: int = 2_000,
        limit: int = 300,
        *,
        query_kind: Literal["context", "own"] = "context",
    ) -> list[RankedChunk]:
        if not query.strip():
            raise ValueError("Dense query must not be empty")
        if limit < 1 or limit > self.scientific.top_chunks_per_list:
            raise ValueError(
                "Dense result limit must be between 1 and "
                f"{self.scientific.top_chunks_per_list}"
            )
        if candidates < 1 or candidates > self.manifest.candidate_count:
            raise ValueError(
                "Dense candidate count must be positive and not exceed the manifest"
            )
        query_vector = _query_vector(
            self.encoder,
            query,
            self.manifest.embedding_dimension,
        )
        best: list[tuple[float, CorpusChunk]] = []
        for shard in self.manifest.shards:
            directory = self.root / shard.path
            index = self.candidate_loader(
                directory / "candidate.faiss",
                ef_search=self.manifest.ef_search,
            )
            local_count = min(candidates, shard.chunk_count)
            rows = np.asarray(
                index.search(query_vector, local_count),
                dtype=np.int64,
            ).reshape(-1)
            rows = np.unique(rows)
            if np.any(rows < 0) or np.any(rows >= shard.chunk_count):
                raise IncompatibleDenseIndexError(
                    "Dense candidate index returned an invalid shard row"
                )
            embeddings = _open_embeddings(
                directory,
                shard.chunk_count,
                self.manifest.embedding_dimension,
            )
            scores = np.asarray(embeddings[rows], dtype=np.float32) @ query_vector
            offsets = _open_offsets(directory, shard.chunk_count)
            with (directory / "metadata.jsonl").open("rb") as metadata:
                local = [
                    (
                        float(score),
                        _read_chunk_at(metadata, int(offsets[int(row)])),
                    )
                    for row, score in zip(rows, scores, strict=True)
                ]
            del embeddings
            del offsets
            best = _best_scored_chunks((*best, *local), limit)

        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        index_hash = self.manifest.manifest_hash()
        return [
            RankedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                ordinal=chunk.ordinal,
                text=chunk.text,
                title=chunk.title,
                corpus=chunk.corpus,
                source_url=chunk.source_url,
                classification=chunk.classification,
                authored=chunk.authored,
                doc_lang=chunk.doc_lang,
                rank=rank,
                score=score,
                retriever="dense",
                query_kind=query_kind,
                query_sha256=query_hash,
                index_manifest_hash=index_hash,
                corpus_manifest_hash=self.manifest.corpus_manifest_hash,
            )
            for rank, (score, chunk) in enumerate(best, start=1)
        ]

    def _validate(self) -> None:
        config = self.scientific
        manifest = self.manifest
        if manifest.model_name != config.embedding_model:
            raise IncompatibleDenseIndexError("Dense model name mismatch")
        if manifest.model_revision != config.embedding_model_revision:
            raise IncompatibleDenseIndexError("Dense model revision mismatch")
        if manifest.scientific_hash != config.manifest_hash():
            raise IncompatibleDenseIndexError("Dense scientific manifest mismatch")
        if manifest.embedding_dimension != config.embedding_dimension:
            raise IncompatibleDenseIndexError("Dense embedding dimension mismatch")
        if (
            manifest.candidate_index != config.dense_index_version
            or manifest.hnsw_m != config.dense_hnsw_m
            or manifest.ef_construction != config.dense_ef_construction
            or manifest.ef_search != config.dense_ef_search
            or manifest.candidate_count != config.dense_candidate_count
        ):
            raise IncompatibleDenseIndexError("Dense index parameters mismatch")
        require_dense_recall(
            manifest.recall_at_300,
            minimum=config.dense_recall_minimum,
        )
        for shard in manifest.shards:
            _verify_shard_files(self.root, shard, manifest.embedding_dimension)


def select_certification_queries(
    chunks_path: Path,
    application_queries_path: Path,
    output_path: Path,
    *,
    application_query_count: int = 50,
    corpus_query_count: int = 50,
) -> list[str]:
    if application_query_count < 1 or corpus_query_count < 1:
        raise ValueError("Dense certification query counts must be positive")
    application_queries = [
        line.strip()
        for line in application_queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(application_queries) != application_query_count:
        raise ValueError(
            f"Expected exactly {application_query_count} application queries"
        )
    if len(set(application_queries)) != len(application_queries):
        raise ValueError("Application certification queries must be unique")

    selected = heapq.nsmallest(
        corpus_query_count,
        (chunk for chunk in _read_chunks(chunks_path) if chunk.ordinal == 0),
        key=lambda chunk: (
            hashlib.sha256(chunk.document_id.encode("utf-8")).hexdigest(),
            chunk.document_id,
            chunk.id,
        ),
    )
    if len(selected) != corpus_query_count:
        raise ValueError(
            f"Expected at least {corpus_query_count} corpus-derived queries"
        )
    corpus_queries = [_corpus_query(chunk) for chunk in selected]
    queries = application_queries + corpus_queries
    if len(set(queries)) != len(queries):
        raise ValueError("Combined dense certification queries must be unique")
    _write_bytes_atomic(
        output_path,
        ("\n".join(queries) + "\n").encode("utf-8"),
    )
    return queries


def certify_sharded_dense_recall(
    root: Path,
    manifest: ShardedDenseIndexManifest,
    queries: Sequence[str],
    *,
    encoder: QueryEncoder,
    candidate_loader: CandidateLoader,
    scientific: ScientificConfig | None = None,
    limit: int = 300,
    block_rows: int = 16_384,
) -> float:
    if not queries or any(not query.strip() for query in queries):
        raise ValueError("Dense recall queries must contain visible, non-empty text")
    if block_rows < 1:
        raise ValueError("Dense recall block size must be positive")
    config = scientific or ScientificConfig()
    if manifest.scientific_hash != config.manifest_hash():
        raise IncompatibleDenseIndexError("Dense scientific manifest mismatch")
    effective_limit = min(limit, manifest.chunk_count)
    verify_sharded_dense_artifacts(root, manifest)
    query_vectors = np.asarray(
        [
            _query_vector(encoder, query, manifest.embedding_dimension)
            for query in queries
        ],
        dtype=np.float32,
    )
    exact: list[list[tuple[float, CorpusChunk]]] = [[] for _ in queries]
    approximate: list[list[tuple[float, CorpusChunk]]] = [[] for _ in queries]
    for shard in manifest.shards:
        directory = root / shard.path
        embeddings = _open_embeddings(
            directory,
            shard.chunk_count,
            manifest.embedding_dimension,
        )
        offsets = _open_offsets(directory, shard.chunk_count)
        with (directory / "metadata.jsonl").open("rb") as metadata:
            for start in range(0, shard.chunk_count, block_rows):
                end = min(start + block_rows, shard.chunk_count)
                block_matrix = np.asarray(embeddings[start:end], dtype=np.float32)
                scores = block_matrix @ query_vectors.T
                block_chunks = [
                    _read_chunk_at(metadata, int(offsets[row]))
                    for row in range(start, end)
                ]
                for query_index in range(len(queries)):
                    scored = [
                        (float(score), chunk)
                        for score, chunk in zip(
                            scores[:, query_index],
                            block_chunks,
                            strict=True,
                        )
                    ]
                    exact[query_index] = _best_scored_chunks(
                        (*exact[query_index], *scored),
                        effective_limit,
                    )

            index = candidate_loader(
                directory / "candidate.faiss",
                ef_search=manifest.ef_search,
            )
            for query_index, query_vector in enumerate(query_vectors):
                rows = np.asarray(
                    index.search(
                        query_vector,
                        min(manifest.candidate_count, shard.chunk_count),
                    ),
                    dtype=np.int64,
                ).reshape(-1)
                rows = np.unique(rows)
                if np.any(rows < 0) or np.any(rows >= shard.chunk_count):
                    raise IncompatibleDenseIndexError(
                        "Dense candidate index returned an invalid shard row"
                    )
                scores = np.asarray(embeddings[rows], dtype=np.float32) @ query_vector
                candidates = [
                    (
                        float(score),
                        _read_chunk_at(metadata, int(offsets[int(row)])),
                    )
                    for row, score in zip(rows, scores, strict=True)
                ]
                approximate[query_index] = _best_scored_chunks(
                    (*approximate[query_index], *candidates),
                    effective_limit,
                )
        del embeddings
        del offsets

    recalls: list[float] = []
    for exact_ranking, approximate_ranking in zip(
        exact,
        approximate,
        strict=True,
    ):
        expected = {chunk.id for _, chunk in exact_ranking}
        returned = {chunk.id for _, chunk in approximate_ranking}
        if not expected:
            raise ValueError("Exact sharded dense ranking returned no results")
        recalls.append(len(expected & returned) / len(expected))
    return float(sum(recalls) / len(recalls))


def verify_sharded_dense_artifacts(
    root: Path,
    manifest: ShardedDenseIndexManifest,
) -> None:
    for shard in manifest.shards:
        _verify_shard_files(root, shard, manifest.embedding_dimension)


def verify_dense_shard_artifacts(
    root: Path,
    shard: DenseShardManifest,
    embedding_dimension: int,
) -> None:
    _verify_shard_files(root, shard, embedding_dimension)


def _best_scored_chunks(
    values: Sequence[tuple[float, CorpusChunk]],
    limit: int,
) -> list[tuple[float, CorpusChunk]]:
    return sorted(values, key=lambda item: (-item[0], item[1].id))[:limit]


def _query_vector(
    encoder: QueryEncoder,
    query: str,
    dimension: int,
) -> NDArray[np.float32]:
    vector = np.asarray(encoder.encode_query(query), dtype=np.float32)
    if vector.shape != (dimension,):
        raise IncompatibleDenseIndexError(
            "Dense query dimension does not match the index manifest"
        )
    if np.any(~np.isfinite(vector)):
        raise IncompatibleDenseIndexError("Dense query vector must be finite")
    return vector


def _open_embeddings(
    directory: Path,
    rows: int,
    dimension: int,
) -> np.memmap:
    return np.memmap(
        directory / "embeddings.f16",
        dtype=np.float16,
        mode="r",
        shape=(rows, dimension),
    )


def _open_offsets(directory: Path, rows: int) -> np.memmap:
    return np.memmap(
        directory / "metadata.offsets.u64",
        dtype="<u8",
        mode="r",
        shape=(rows,),
    )


def _read_chunk_at(metadata: object, offset: int) -> CorpusChunk:
    metadata.seek(offset)
    line = metadata.readline()
    try:
        return CorpusChunk.model_validate_json(line)
    except ValueError as error:
        raise IncompatibleDenseIndexError(
            "Dense candidate metadata is invalid"
        ) from error


def _verify_shard_files(
    root: Path,
    shard: DenseShardManifest,
    dimension: int,
) -> None:
    directory = root / shard.path
    expected_sizes = {
        "embeddings.f16": shard.chunk_count * dimension * 2,
        "metadata.offsets.u64": shard.chunk_count * 8,
    }
    expected_hashes = {
        "embeddings.f16": shard.embeddings_sha256,
        "metadata.jsonl": shard.metadata_sha256,
        "metadata.offsets.u64": shard.offsets_sha256,
        "candidate.faiss": shard.candidate_index_sha256,
    }
    for name, expected_hash in expected_hashes.items():
        path = directory / name
        if not path.is_file():
            raise IncompatibleDenseIndexError(f"Dense shard artifact is missing: {path}")
        expected_size = expected_sizes.get(name)
        if expected_size is not None and path.stat().st_size != expected_size:
            raise IncompatibleDenseIndexError(
                f"Dense shard artifact has an invalid size: {path}"
            )
        if _sha256_file(path) != expected_hash:
            raise IncompatibleDenseIndexError(
                f"Dense shard artifact checksum mismatch: {path}"
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


def _corpus_query(chunk: CorpusChunk) -> str:
    title = " ".join((chunk.title or "").split())
    opening = " ".join(chunk.text.split()[:60])
    value = " ".join(part for part in (title, opening) if part).strip()
    if not value:
        raise ValueError(f"Corpus chunk {chunk.id} cannot produce a recall query")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(value)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
