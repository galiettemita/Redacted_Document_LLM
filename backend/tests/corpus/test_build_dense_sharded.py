from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from redacted_backend.corpus.best_available import BestAvailableCorpusManifest
from redacted_backend.corpus.build_dense_sharded import build_sharded_dense_from_files
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.dense import DenseRecallError
from redacted_backend.retrieval.types import ShardedDenseIndexManifest
from redacted_backend.scientific_config import ScientificConfig


class DeterministicEncoder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.document_batches: list[list[str]] = []

    def encode_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.document_batches.append(list(texts))
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            index = int(text.rsplit(" ", 1)[-1]) % self.dimension
            vectors[row, index] = 1.0
        return vectors

    def encode_query(self, text: str) -> NDArray[np.float32]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        suffix = text.rsplit(" ", 1)[-1]
        index = int(suffix) % self.dimension if suffix.isdigit() else 0
        vector[index] = 1.0
        return vector


class ArrayCandidateIndex:
    def __init__(self, *, candidate_limit: int | None = None) -> None:
        self.vectors: list[NDArray[np.float32]] = []
        self.candidate_limit = candidate_limit

    def add(self, vectors: NDArray[np.float32]) -> None:
        self.vectors.extend(np.asarray(vectors, dtype=np.float32))

    def search(
        self,
        query: NDArray[np.float32],
        candidates: int,
    ) -> NDArray[np.int64]:
        matrix = np.asarray(self.vectors, dtype=np.float32)
        scores = matrix @ query
        count = min(candidates, len(matrix))
        if self.candidate_limit is not None:
            count = min(count, self.candidate_limit)
        rows = sorted(range(len(matrix)), key=lambda row: (-scores[row], row))[:count]
        return np.asarray(rows, dtype=np.int64)

    def save(self, path: Path) -> None:
        with path.open("wb") as destination:
            np.save(destination, np.asarray(self.vectors, dtype=np.float32))

    @classmethod
    def load(cls, path: Path, *, ef_search: int) -> ArrayCandidateIndex:
        del ef_search
        index = cls()
        with path.open("rb") as source:
            index.vectors = list(np.load(source).astype(np.float32))
        return index


def _chunk(index: int) -> CorpusChunk:
    text = f"archive context {index}"
    return CorpusChunk(
        id=hashlib.sha256(f"chunk-{index}".encode()).hexdigest(),
        document_id=f"document-{index}",
        ordinal=0,
        text=text,
        word_count=3,
        title=f"Public title {index}",
        corpus="historylab-unlabeled",
        source_url="https://example.test/document.pdf",
        classification="UNCLASSIFIED",
        authored="1975-01-01",
        doc_lang=None,
    )


def _fixture(
    root: Path,
    count: int,
) -> tuple[Path, Path, Path, list[CorpusChunk]]:
    scientific = ScientificConfig()
    chunks = [_chunk(index) for index in range(count)]
    chunks_path = root / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
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
    manifest = BestAvailableCorpusManifest(
        source_profile_hash="1" * 64,
        source_sha256="2" * 64,
        source_size_bytes=100,
        audit_manifest_hash="3" * 64,
        limitations_sha256="4" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        parsed_row_count=count,
        included_row_count=count,
        empty_body_count=0,
        malformed_row_count=0,
        document_count=count,
        chunk_count=count,
        corpus_counts={"historylab-unlabeled": count},
        chunks_sha256=hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
    )
    manifest_path = root / "corpus.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    application_queries = root / "application-queries.txt"
    application_queries.write_text("public query 0\npublic query 1\n", encoding="utf-8")
    return chunks_path, manifest_path, application_queries, chunks


def _build(
    root: Path,
    *,
    count: int = 5,
    encoder: DeterministicEncoder | None = None,
    factory=None,
    loader=None,
) -> tuple[ShardedDenseIndexManifest, Path, DeterministicEncoder]:
    chunks, corpus, queries, _ = _fixture(root, count)
    output = root / "dense"
    effective_encoder = encoder or DeterministicEncoder(768)
    report = build_sharded_dense_from_files(
        chunks,
        corpus,
        output,
        output / "manifest.json",
        application_queries_path=queries,
        certification_queries_path=output / "recall-queries.txt",
        encoder=effective_encoder,
        candidate_factory=factory or (lambda _config: ArrayCandidateIndex()),
        candidate_loader=loader or ArrayCandidateIndex.load,
        batch_size=2,
        shard_size=2,
        application_query_count=2,
        corpus_query_count=2,
    )
    return report.index_manifest, output, effective_encoder


def test_sharded_dense_build_publishes_consecutive_bounded_verified_shards(
    tmp_path: Path,
) -> None:
    manifest, output, _ = _build(tmp_path)

    assert manifest.schema_version == "sharded-dense-index-manifest-v1"
    assert manifest.retrieval_profile == "best_available_foiarchive_v1"
    assert manifest.chunk_count == 5
    assert manifest.shard_size_limit == 2
    assert [(item.start_row, item.end_row) for item in manifest.shards] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    for shard in manifest.shards:
        directory = output / shard.path
        assert (directory / "embeddings.f16").stat().st_size == (
            shard.chunk_count * 768 * 2
        )
        assert (directory / "metadata.offsets.u64").stat().st_size == (
            shard.chunk_count * 8
        )
        for name, expected in {
            "embeddings.f16": shard.embeddings_sha256,
            "metadata.jsonl": shard.metadata_sha256,
            "metadata.offsets.u64": shard.offsets_sha256,
            "candidate.faiss": shard.candidate_index_sha256,
        }.items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    query_bytes = (output / "recall-queries.txt").read_bytes()
    assert hashlib.sha256(query_bytes).hexdigest() == manifest.recall_queries_sha256
    assert len(query_bytes.decode("utf-8").splitlines()) == 4
    assert manifest.recall_at_300 == 1.0


def test_sharded_dense_build_resumes_completed_shards_after_failure(
    tmp_path: Path,
) -> None:
    chunks, corpus, queries, _ = _fixture(tmp_path, 5)
    output = tmp_path / "dense"
    first_encoder = DeterministicEncoder(768)
    creations = 0

    def failing_factory(_config: ScientificConfig) -> ArrayCandidateIndex:
        nonlocal creations
        creations += 1
        if creations == 2:
            raise OSError("simulated shard failure")
        return ArrayCandidateIndex()

    with pytest.raises(OSError, match="simulated shard failure"):
        build_sharded_dense_from_files(
            chunks,
            corpus,
            output,
            output / "manifest.json",
            application_queries_path=queries,
            certification_queries_path=output / "recall-queries.txt",
            encoder=first_encoder,
            candidate_factory=failing_factory,
            candidate_loader=ArrayCandidateIndex.load,
            batch_size=2,
            shard_size=2,
            application_query_count=2,
            corpus_query_count=2,
        )
    assert (output / "shard-00000/manifest.json").is_file()
    assert not (output / "manifest.json").exists()

    second_encoder = DeterministicEncoder(768)
    report = build_sharded_dense_from_files(
        chunks,
        corpus,
        output,
        output / "manifest.json",
        application_queries_path=queries,
        certification_queries_path=output / "recall-queries.txt",
        encoder=second_encoder,
        candidate_factory=lambda _config: ArrayCandidateIndex(),
        candidate_loader=ArrayCandidateIndex.load,
        batch_size=2,
        shard_size=2,
        application_query_count=2,
        corpus_query_count=2,
    )

    assert report.index_manifest.chunk_count == 5
    assert sum(len(batch) for batch in first_encoder.document_batches) == 2
    assert sum(len(batch) for batch in second_encoder.document_batches) == 3


def test_sharded_dense_resume_rejects_corrupt_completed_artifact(
    tmp_path: Path,
) -> None:
    manifest, output, _ = _build(tmp_path)
    (output / "manifest.json").unlink()
    first = output / manifest.shards[0].path / "metadata.jsonl"
    first.write_bytes(first.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="checksum"):
        _build(tmp_path)

    assert not (output / "manifest.json").exists()


def test_sharded_dense_does_not_publish_top_manifest_when_recall_fails(
    tmp_path: Path,
) -> None:
    def limited_loader(path: Path, *, ef_search: int) -> ArrayCandidateIndex:
        index = ArrayCandidateIndex.load(path, ef_search=ef_search)
        index.candidate_limit = 1
        return index

    with pytest.raises(DenseRecallError, match="recall@300"):
        _build(
            tmp_path,
            loader=limited_loader,
        )

    assert not (tmp_path / "dense/manifest.json").exists()


def test_sharded_dense_rejects_more_than_one_million_chunks_per_shard(
    tmp_path: Path,
) -> None:
    chunks, corpus, queries, _ = _fixture(tmp_path, 1)

    with pytest.raises(ValueError, match="1000000"):
        build_sharded_dense_from_files(
            chunks,
            corpus,
            tmp_path / "dense",
            tmp_path / "dense/manifest.json",
            application_queries_path=queries,
            certification_queries_path=tmp_path / "dense/recall-queries.txt",
            encoder=DeterministicEncoder(768),
            candidate_factory=lambda _config: ArrayCandidateIndex(),
            candidate_loader=ArrayCandidateIndex.load,
            shard_size=1_000_001,
            application_query_count=2,
            corpus_query_count=2,
        )
