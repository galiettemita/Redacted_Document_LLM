from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.chunk_offsets import build_chunk_offsets
from redacted_backend.retrieval.dense import IncompatibleDenseIndexError
from redacted_backend.retrieval.dense_profile import (
    DenseArtifactProfile,
    dense_semantics_hash,
)
from redacted_backend.retrieval.dense_sharded import ShardedDenseIndex
from redacted_backend.retrieval.types import (
    CompressedDenseShardManifest,
    CompressedShardedDenseIndexManifest,
)
from redacted_backend.scientific_config import ScientificConfig


class QueryEncoder:
    def encode_query(self, text: str) -> NDArray[np.float32]:
        assert text == "visible query"
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


class RowsCandidateIndex:
    def __init__(self, rows: list[int], *, ntotal: int) -> None:
        self.rows = rows
        self.ntotal = ntotal

    def search(
        self,
        query: NDArray[np.float32],
        candidates: int,
    ) -> NDArray[np.int64]:
        del query
        return np.asarray(self.rows[:candidates], dtype=np.int64)


def _load_template(
    _path: Path,
    *,
    profile: DenseArtifactProfile,
) -> RowsCandidateIndex:
    del profile
    return RowsCandidateIndex([], ntotal=0)


def _profile(**updates: object) -> DenseArtifactProfile:
    values = {
        "embedding_dimension": 3,
        "ivf_nlist": 1,
        "pq_subvectors": 1,
        "pq_bits": 1,
        "nprobe": 1,
        "candidate_count_per_shard": 4,
        "training_sample_count": 2,
        "shard_size_limit": 3,
        "max_artifact_bytes": 1024 * 1024,
    }
    values.update(updates)
    return DenseArtifactProfile(**values)


def _scientific() -> ScientificConfig:
    return ScientificConfig(embedding_dimension=3)


def _chunk(name: str, ordinal: int = 0) -> CorpusChunk:
    text = f"Public archive text {name}"
    return CorpusChunk(
        id=hashlib.sha256(name.encode()).hexdigest(),
        document_id=f"document-{name}",
        ordinal=ordinal,
        text=text,
        word_count=len(text.split()),
        title=f"Title {name}",
        corpus="historylab-unlabeled",
        source_url="https://example.test/source",
        classification="UNCLASSIFIED",
        authored="1975-01-01",
        doc_lang=None,
    )


def _write_runtime_fixture(
    root: Path,
    *,
    vectors_by_shard: list[NDArray[np.float32]],
    chunks: list[CorpusChunk],
    candidate_rows: list[list[int]],
    profile: DenseArtifactProfile | None = None,
) -> tuple[CompressedShardedDenseIndexManifest, Path, dict[str, list[int]]]:
    effective_profile = profile or _profile()
    scientific = _scientific()
    chunks_path = root / "chunks.jsonl"
    chunks_path.write_text(
        "".join(
            json.dumps(
                chunk.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
    )
    chunks_sha256 = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    offsets_path = root / "chunks.offsets.u64"
    offsets = build_chunk_offsets(
        chunks_path,
        offsets_path,
        expected_count=len(chunks),
        expected_chunks_sha256=chunks_sha256,
    )
    template = root / "candidate-template.faiss"
    template.write_bytes(b"template")
    template_sha256 = hashlib.sha256(template.read_bytes()).hexdigest()
    recall = root / "recall-report.json"
    recall.write_bytes(b"{}\n")
    shards: list[CompressedDenseShardManifest] = []
    rows_by_path: dict[str, list[int]] = {}
    start = 0
    for shard_id, (vectors, rows) in enumerate(
        zip(vectors_by_shard, candidate_rows, strict=True)
    ):
        directory = root / f"shard-{shard_id:05d}"
        directory.mkdir()
        embeddings = directory / "embeddings.f16"
        np.asarray(vectors, dtype=np.float16).tofile(embeddings)
        candidate = directory / "candidate.faiss"
        candidate.write_text(json.dumps(rows), encoding="utf-8")
        rows_by_path[directory.name] = rows
        shards.append(
            CompressedDenseShardManifest(
                artifact_profile_hash=effective_profile.manifest_hash(),
                candidate_template_sha256=template_sha256,
                shard_id=shard_id,
                path=directory.name,
                start_row=start,
                end_row=start + len(vectors),
                chunk_count=len(vectors),
                embeddings_size_bytes=embeddings.stat().st_size,
                embeddings_sha256=hashlib.sha256(
                    embeddings.read_bytes()
                ).hexdigest(),
                candidate_index_size_bytes=candidate.stat().st_size,
                candidate_index_sha256=hashlib.sha256(
                    candidate.read_bytes()
                ).hexdigest(),
            )
        )
        start += len(vectors)
    listed_bytes = (
        offsets.offsets_size_bytes
        + template.stat().st_size
        + recall.stat().st_size
        + sum(
            shard.embeddings_size_bytes + shard.candidate_index_size_bytes
            for shard in shards
        )
    )
    manifest = CompressedShardedDenseIndexManifest(
        model_name=scientific.embedding_model,
        model_revision=scientific.embedding_model_revision,
        embedding_dimension=3,
        corpus_manifest_hash="1" * 64,
        dense_semantics_hash=dense_semantics_hash(scientific),
        chunks_sha256=chunks_sha256,
        chunk_count=len(chunks),
        artifact_profile=effective_profile,
        artifact_profile_hash=effective_profile.manifest_hash(),
        chunk_offsets_path=offsets_path.name,
        chunk_offsets_format=effective_profile.metadata_lookup_version,
        chunk_offsets_size_bytes=offsets.offsets_size_bytes,
        chunk_offsets_sha256=offsets.offsets_sha256,
        candidate_template_path=template.name,
        candidate_template_size_bytes=template.stat().st_size,
        candidate_template_sha256=template_sha256,
        shard_size_limit=effective_profile.shard_size_limit,
        shards=shards,
        dtype="float16",
        metric="inner_product",
        candidate_index=effective_profile.candidate_index,
        recall_report_path=recall.name,
        recall_report_size_bytes=recall.stat().st_size,
        recall_report_sha256=hashlib.sha256(recall.read_bytes()).hexdigest(),
        recall_at_300=1.0,
        recall_queries_sha256="4" * 64,
        recall_query_count=4,
        total_artifact_bytes=listed_bytes,
    )
    return manifest, chunks_path, rows_by_path


def test_search_caches_every_shard_and_exactly_rescores_global_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_b = _chunk("b")
    chunk_invalid = _chunk("invalid")
    chunk_c = _chunk("c")
    chunk_z = _chunk("z")
    chunk_a = _chunk("a")
    chunk_d = _chunk("d")
    manifest, chunks_path, rows_by_path = _write_runtime_fixture(
        tmp_path,
        vectors_by_shard=[
            np.asarray(
                [[0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.8, 0.2, 0.0]]
            ),
            np.asarray(
                [[0.95, 0.05, 0.0], [0.9, 0.1, 0.0], [0.7, 0.3, 0.0]]
            ),
        ],
        chunks=[chunk_a, chunk_invalid, chunk_c, chunk_z, chunk_b, chunk_d],
        candidate_rows=[[0, -1, 2, 2], [0, 1, 2]],
    )
    loaded_paths: list[str] = []
    loaded_templates: list[str] = []

    def load_candidate(
        path: Path,
        *,
        profile: DenseArtifactProfile,
    ) -> RowsCandidateIndex:
        assert profile == manifest.artifact_profile
        loaded_paths.append(path.parent.name)
        return RowsCandidateIndex(
            rows_by_path[path.parent.name],
            ntotal=manifest.shards[int(path.parent.name[-5:])].chunk_count,
        )

    read_count = 0
    original_read = __import__(
        "redacted_backend.retrieval.chunk_offsets",
        fromlist=["ChunkOffsetIndex"],
    ).ChunkOffsetIndex.read

    def counted_read(self, row: int, **kwargs):
        nonlocal read_count
        read_count += 1
        return original_read(self, row, **kwargs)

    monkeypatch.setattr(
        "redacted_backend.retrieval.chunk_offsets.ChunkOffsetIndex.read",
        counted_read,
    )

    def load_template(
        path: Path,
        *,
        profile: DenseArtifactProfile,
    ) -> RowsCandidateIndex:
        assert profile == manifest.artifact_profile
        loaded_templates.append(path.name)
        return RowsCandidateIndex([], ntotal=0)

    with ShardedDenseIndex(
        root=tmp_path,
        chunks_path=chunks_path,
        manifest=manifest,
        encoder=QueryEncoder(),
        candidate_loader=load_candidate,
        template_loader=load_template,
        scientific=_scientific(),
    ) as index:
        first = index.search(
            "visible query",
            candidates=4,
            limit=5,
            query_kind="context",
        )
        second = index.search(
            "visible query",
            candidates=4,
            limit=5,
            query_kind="context",
        )

    assert loaded_paths == ["shard-00000", "shard-00001"]
    assert loaded_templates == ["candidate-template.faiss"]
    assert [item.chunk_id for item in first] == [
        chunk_z.id,
        chunk_a.id,
        chunk_b.id,
        chunk_c.id,
        chunk_d.id,
    ]
    assert chunk_b.id < chunk_a.id  # chunk-ID order is deliberately reversed.
    assert second == first
    assert read_count == 10
    assert [item.rank for item in first] == [1, 2, 3, 4, 5]
    assert all(item.retriever == "dense" for item in first)
    assert all(item.query_kind == "context" for item in first)
    assert all(item.corpus_manifest_hash == "1" * 64 for item in first)
    assert all(item.index_manifest_hash == manifest.manifest_hash() for item in first)


def test_search_rejects_out_of_range_candidate_rows(tmp_path: Path) -> None:
    manifest, chunks_path, rows_by_path = _write_runtime_fixture(
        tmp_path,
        vectors_by_shard=[np.asarray([[1.0, 0.0, 0.0]])],
        chunks=[_chunk("only")],
        candidate_rows=[[1]],
        profile=_profile(
            training_sample_count=1,
            shard_size_limit=1,
            candidate_count_per_shard=1,
        ),
    )

    def loader(path: Path, *, profile: DenseArtifactProfile) -> RowsCandidateIndex:
        del profile
        return RowsCandidateIndex(rows_by_path[path.parent.name], ntotal=1)

    with ShardedDenseIndex(
        root=tmp_path,
        chunks_path=chunks_path,
        manifest=manifest,
        encoder=QueryEncoder(),
        candidate_loader=loader,
        template_loader=_load_template,
        scientific=_scientific(),
    ) as index:
        with pytest.raises(IncompatibleDenseIndexError, match="invalid.*row"):
            index.search("visible query", candidates=1, limit=1)


def test_runtime_refuses_a_checksum_consistent_non_faiss_template(
    tmp_path: Path,
) -> None:
    manifest, chunks_path, rows_by_path = _write_runtime_fixture(
        tmp_path,
        vectors_by_shard=[np.asarray([[1.0, 0.0, 0.0]])],
        chunks=[_chunk("only")],
        candidate_rows=[[0]],
        profile=_profile(
            training_sample_count=1,
            shard_size_limit=1,
            candidate_count_per_shard=1,
        ),
    )

    def candidate_loader(
        path: Path,
        *,
        profile: DenseArtifactProfile,
    ) -> RowsCandidateIndex:
        del profile
        return RowsCandidateIndex(rows_by_path[path.parent.name], ntotal=1)

    with pytest.raises(IncompatibleDenseIndexError, match="artifact format"):
        ShardedDenseIndex(
            root=tmp_path,
            chunks_path=chunks_path,
            manifest=manifest,
            encoder=QueryEncoder(),
            candidate_loader=candidate_loader,
            scientific=_scientific(),
        )


class _ConcurrencyTracker:
    def __init__(self, barrier: threading.Barrier) -> None:
        self.barrier = barrier
        self.lock = threading.Lock()
        self.active = 0
        self.maximum = 0

    def enter(self, *, wait: bool) -> None:
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        if wait:
            self.barrier.wait(timeout=2)

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


class _ConcurrentCandidate(RowsCandidateIndex):
    def __init__(
        self,
        tracker: _ConcurrencyTracker,
        *,
        wait: bool,
    ) -> None:
        super().__init__([0], ntotal=1)
        self.tracker = tracker
        self.wait = wait

    def search(
        self,
        query: NDArray[np.float32],
        candidates: int,
    ) -> NDArray[np.int64]:
        self.tracker.enter(wait=self.wait)
        try:
            return super().search(query, candidates)
        finally:
            self.tracker.leave()


def test_search_uses_no_more_than_four_parallel_shards(tmp_path: Path) -> None:
    profile = _profile(
        training_sample_count=6,
        shard_size_limit=1,
        candidate_count_per_shard=1,
        max_parallel_shards=4,
    )
    manifest, chunks_path, _ = _write_runtime_fixture(
        tmp_path,
        vectors_by_shard=[np.asarray([[1.0, 0.0, 0.0]]) for _ in range(6)],
        chunks=[_chunk(f"chunk-{index}") for index in range(6)],
        candidate_rows=[[0] for _ in range(6)],
        profile=profile,
    )
    tracker = _ConcurrencyTracker(threading.Barrier(4))
    loaded = 0

    def loader(path: Path, *, profile: DenseArtifactProfile) -> _ConcurrentCandidate:
        nonlocal loaded
        del path, profile
        ordinal = loaded
        loaded += 1
        return _ConcurrentCandidate(tracker, wait=ordinal < 4)

    with ShardedDenseIndex(
        root=tmp_path,
        chunks_path=chunks_path,
        manifest=manifest,
        encoder=QueryEncoder(),
        candidate_loader=loader,
        template_loader=_load_template,
        scientific=_scientific(),
    ) as index:
        results = index.search("visible query", candidates=1, limit=6)

    assert len(results) == 6
    assert tracker.maximum == 4
