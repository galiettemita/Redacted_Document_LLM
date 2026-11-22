from __future__ import annotations

import hashlib
import heapq
import os
import platform
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.chunk_offsets import ChunkOffsetIndex
from redacted_backend.retrieval.dense import (
    IncompatibleDenseIndexError,
    QueryEncoder,
    require_dense_recall,
)
from redacted_backend.retrieval.dense_profile import (
    DenseArtifactProfile,
    dense_semantics_hash,
)
from redacted_backend.retrieval.faiss_ivfpq import FaissIvfPqIndex
from redacted_backend.retrieval.types import (
    CompressedDenseShardManifest,
    CompressedShardedDenseIndexManifest,
    DenseLatencyReport,
    DenseShardManifest,
    RankedChunk,
    ShardedDenseIndexManifest,
)
from redacted_backend.scientific_config import ScientificConfig


class CandidateIndex(Protocol):
    ntotal: int

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
        chunks_path: Path,
        manifest: CompressedShardedDenseIndexManifest,
        encoder: QueryEncoder,
        candidate_loader: CandidateLoader = FaissIvfPqIndex.load,
        profile: DenseArtifactProfile | None = None,
        scientific: ScientificConfig | None = None,
    ) -> None:
        self.root = root
        self.chunks_path = chunks_path
        self.manifest = manifest
        self.encoder = encoder
        self.candidate_loader = candidate_loader
        self.scientific = scientific or ScientificConfig()
        self.profile = profile or manifest.artifact_profile
        self._closed = False
        self._candidates: list[CandidateIndex] = []
        self._embeddings: list[np.memmap] = []
        self._chunks: ChunkOffsetIndex | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._validate()
        try:
            self._chunks = ChunkOffsetIndex.open(
                chunks_path,
                root / manifest.chunk_offsets_path,
                expected_count=manifest.chunk_count,
                expected_chunks_sha256=manifest.chunks_sha256,
                expected_offsets_sha256=manifest.chunk_offsets_sha256,
            )
            for shard in manifest.shards:
                directory = root / shard.path
                candidate = candidate_loader(
                    directory / "candidate.faiss",
                    profile=self.profile,
                )
                if int(candidate.ntotal) != shard.chunk_count:
                    raise IncompatibleDenseIndexError(
                        "Dense candidate row count does not match its shard"
                    )
                self._candidates.append(candidate)
                self._embeddings.append(
                    np.memmap(
                        directory / "embeddings.f16",
                        dtype=np.float16,
                        mode="r",
                        shape=(shard.chunk_count, manifest.embedding_dimension),
                    )
                )
            self._executor = ThreadPoolExecutor(
                max_workers=min(
                    self.profile.max_parallel_shards,
                    len(manifest.shards),
                ),
                thread_name_prefix="dense-shard",
            )
        except BaseException:
            if self._chunks is not None:
                self._chunks.close()
            self._embeddings.clear()
            self._candidates.clear()
            raise

    def search(
        self,
        query: str,
        candidates: int | None = None,
        limit: int = 300,
        *,
        query_kind: Literal["context", "own"] = "context",
    ) -> list[RankedChunk]:
        if self._closed:
            raise IncompatibleDenseIndexError("Compressed dense runtime is closed")
        if not query.strip():
            raise ValueError("Dense query must not be empty")
        if limit < 1 or limit > self.scientific.top_chunks_per_list:
            raise ValueError(
                "Dense result limit must be between 1 and "
                f"{self.scientific.top_chunks_per_list}"
            )
        requested_candidates = (
            self.profile.candidate_count_per_shard
            if candidates is None
            else candidates
        )
        if (
            requested_candidates < 1
            or requested_candidates > self.profile.candidate_count_per_shard
        ):
            raise ValueError(
                "Dense candidate count must be positive and not exceed the manifest"
            )
        query_vector = _query_vector(
            self.encoder,
            query,
            self.manifest.embedding_dimension,
        )
        assert self._executor is not None
        futures = [
            self._executor.submit(
                self._search_shard,
                ordinal,
                query_vector,
                requested_candidates,
            )
            for ordinal in range(len(self.manifest.shards))
        ]
        scored = [value for future in futures for value in future.result()]
        ranked = self._materialize(scored, limit)

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
            for rank, (score, chunk) in enumerate(ranked, start=1)
        ]

    def _validate(self) -> None:
        config = self.scientific
        manifest = self.manifest
        if manifest.model_name != config.embedding_model:
            raise IncompatibleDenseIndexError("Dense model name mismatch")
        if manifest.model_revision != config.embedding_model_revision:
            raise IncompatibleDenseIndexError("Dense model revision mismatch")
        if manifest.dense_semantics_hash != dense_semantics_hash(config):
            raise IncompatibleDenseIndexError("Dense semantic manifest mismatch")
        if manifest.embedding_dimension != config.embedding_dimension:
            raise IncompatibleDenseIndexError("Dense embedding dimension mismatch")
        if (
            manifest.artifact_profile_hash != self.profile.manifest_hash()
            or manifest.artifact_profile != self.profile
            or manifest.candidate_index != self.profile.candidate_index
        ):
            raise IncompatibleDenseIndexError("Dense artifact profile mismatch")
        require_dense_recall(
            manifest.recall_at_300,
            minimum=config.dense_recall_minimum,
        )
        _verify_compressed_artifacts(self.root, manifest)

    def _search_shard(
        self,
        ordinal: int,
        query_vector: NDArray[np.float32],
        candidates: int,
    ) -> list[tuple[float, int]]:
        shard = self.manifest.shards[ordinal]
        local_count = min(candidates, shard.chunk_count)
        raw_rows = np.asarray(
            self._candidates[ordinal].search(query_vector, local_count),
            dtype=np.int64,
        ).reshape(-1)
        rows: list[int] = []
        seen: set[int] = set()
        for value in raw_rows:
            row = int(value)
            if row == -1 or row in seen:
                continue
            if row < 0 or row >= shard.chunk_count:
                raise IncompatibleDenseIndexError(
                    "Dense candidate index returned an invalid shard row"
                )
            seen.add(row)
            rows.append(row)
        if not rows:
            return []
        row_array = np.asarray(rows, dtype=np.int64)
        scores = (
            np.asarray(self._embeddings[ordinal][row_array], dtype=np.float32)
            @ query_vector
        )
        return [
            (float(score), shard.start_row + row)
            for row, score in zip(rows, scores, strict=True)
        ]

    def _materialize(
        self,
        scored: Sequence[tuple[float, int]],
        limit: int,
    ) -> list[tuple[float, CorpusChunk]]:
        if not scored:
            return []
        ordered = sorted(scored, key=lambda value: (-value[0], value[1]))
        selected = ordered[:limit]
        cached: dict[int, CorpusChunk] = {}
        if len(ordered) > limit:
            cutoff = selected[-1][0]
            above = [value for value in ordered if value[0] > cutoff]
            tied = [value for value in ordered if value[0] == cutoff]
            remaining = limit - len(above)
            if len(tied) > remaining:
                assert self._chunks is not None
                for _score, row in tied:
                    cached[row] = self._chunks.read(row)
                tied = sorted(tied, key=lambda value: cached[value[1]].id)
            selected = above + tied[:remaining]
        assert self._chunks is not None
        materialized: list[tuple[float, CorpusChunk]] = []
        for score, row in selected:
            chunk = cached.get(row)
            if chunk is None:
                chunk = self._chunks.read(row)
            materialized.append((score, chunk))
        return sorted(materialized, key=lambda value: (-value[0], value[1].id))

    def close(self) -> None:
        if self._closed:
            return
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        if self._chunks is not None:
            self._chunks.close()
        self._embeddings.clear()
        self._candidates.clear()
        self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise IncompatibleDenseIndexError("Compressed dense runtime is closed")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()


class DenseLatencyError(RuntimeError):
    pass


def certify_dense_latency(
    runtime: object,
    queries: Sequence[str],
    output_path: Path,
    *,
    clock: Callable[[], float] = time.monotonic,
    platform_name: Callable[[], str] = platform.platform,
    machine_name: Callable[[], str] = platform.machine,
) -> DenseLatencyReport:
    manifest = runtime.manifest
    if len(queries) != manifest.recall_query_count or any(
        not query.strip() for query in queries
    ):
        raise DenseLatencyError("Dense latency query count or content mismatch")
    query_bytes = ("\n".join(queries) + "\n").encode("utf-8")
    query_hash = hashlib.sha256(query_bytes).hexdigest()
    if query_hash != manifest.recall_queries_sha256:
        raise DenseLatencyError("Dense latency query hash mismatch")
    runtime.search(queries[0])
    timings: list[float] = []
    maximum = manifest.artifact_profile.max_warm_search_seconds
    for query in queries:
        started = clock()
        runtime.search(query)
        elapsed = clock() - started
        if elapsed <= 0:
            raise DenseLatencyError("Dense latency clock returned a nonpositive duration")
        if elapsed > maximum:
            raise DenseLatencyError(
                f"Dense warm search took {elapsed:.2f}s; maximum is {maximum:.1f}s"
            )
        timings.append(elapsed)
    report = DenseLatencyReport(
        dense_manifest_hash=manifest.manifest_hash(),
        recall_queries_sha256=query_hash,
        query_count=len(queries),
        warm_search_seconds=timings,
        max_observed_seconds=max(timings),
        maximum_allowed_seconds=maximum,
        platform=platform_name(),
        machine=machine_name(),
    )
    report.require_compatible(manifest)
    _write_bytes_atomic(
        output_path,
        (report.model_dump_json() + "\n").encode("utf-8"),
    )
    return report


def _verify_compressed_artifacts(
    root: Path,
    manifest: CompressedShardedDenseIndexManifest,
) -> None:
    artifacts: list[tuple[Path, int, str]] = [
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
        artifacts.extend(
            [
                (
                    root / shard.path / "embeddings.f16",
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
    for path, size, checksum in artifacts:
        if not path.is_file():
            raise IncompatibleDenseIndexError(
                f"Compressed dense artifact is missing: {path}"
            )
        if path.stat().st_size != size:
            raise IncompatibleDenseIndexError(
                f"Compressed dense artifact has an invalid size: {path}"
            )
        if _sha256_file(path) != checksum:
            raise IncompatibleDenseIndexError(
                f"Compressed dense artifact checksum mismatch: {path}"
            )


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


def verify_sharded_dense_artifacts_cached(
    root: Path,
    manifest: ShardedDenseIndexManifest,
) -> None:
    signatures: list[tuple[str, int, int]] = []
    for shard in manifest.shards:
        for name in (
            "embeddings.f16",
            "metadata.jsonl",
            "metadata.offsets.u64",
            "candidate.faiss",
        ):
            path = root / shard.path / name
            stat = path.stat()
            signatures.append((f"{shard.path}/{name}", stat.st_size, stat.st_mtime_ns))
    _verify_sharded_dense_artifacts_cached(
        str(root.resolve()),
        manifest.model_dump_json(),
        tuple(signatures),
    )


@lru_cache(maxsize=8)
def _verify_sharded_dense_artifacts_cached(
    root: str,
    manifest_json: str,
    _signatures: tuple[tuple[str, int, int], ...],
) -> None:
    manifest = ShardedDenseIndexManifest.model_validate_json(manifest_json)
    verify_sharded_dense_artifacts(Path(root), manifest)


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
