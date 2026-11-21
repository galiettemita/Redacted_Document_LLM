from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.dense import IncompatibleDenseIndexError
from redacted_backend.retrieval.dense_sharded import ShardedDenseIndex
from redacted_backend.retrieval.types import (
    DenseShardManifest,
    ShardedDenseIndexManifest,
)
from redacted_backend.scientific_config import ScientificConfig


class QueryEncoder:
    def encode_query(self, text: str) -> NDArray[np.float32]:
        assert text == "visible query"
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


class RowsCandidateIndex:
    def __init__(self, rows: list[int]) -> None:
        self.rows = rows

    def search(
        self,
        query: NDArray[np.float32],
        candidates: int,
    ) -> NDArray[np.int64]:
        del query
        return np.asarray(self.rows[:candidates], dtype=np.int64)


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


def _write_shard(
    root: Path,
    ordinal: int,
    start: int,
    vectors: NDArray[np.float32],
    chunks: list[CorpusChunk | None],
    candidate_rows: list[int],
) -> DenseShardManifest:
    directory = root / f"shard-{ordinal:05d}"
    directory.mkdir(parents=True)
    embeddings = directory / "embeddings.f16"
    np.asarray(vectors, dtype=np.float16).tofile(embeddings)
    metadata = directory / "metadata.jsonl"
    offsets: list[int] = []
    with metadata.open("wb") as destination:
        for chunk in chunks:
            offsets.append(destination.tell())
            if chunk is None:
                destination.write(b"{not-json}\n")
            else:
                destination.write(
                    json.dumps(
                        chunk.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
    offsets_path = directory / "metadata.offsets.u64"
    np.asarray(offsets, dtype="<u8").tofile(offsets_path)
    candidate = directory / "candidate.faiss"
    candidate.write_text(json.dumps(candidate_rows), encoding="utf-8")
    return DenseShardManifest(
        shard_id=ordinal,
        path=directory.name,
        start_row=start,
        end_row=start + len(chunks),
        chunk_count=len(chunks),
        embeddings_sha256=hashlib.sha256(embeddings.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        offsets_sha256=hashlib.sha256(offsets_path.read_bytes()).hexdigest(),
        candidate_index_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    )


def _manifest(
    shards: list[DenseShardManifest],
    scientific: ScientificConfig,
) -> ShardedDenseIndexManifest:
    return ShardedDenseIndexManifest(
        model_name="BAAI/bge-base-en-v1.5",
        model_revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        embedding_dimension=3,
        corpus_manifest_hash="1" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunks_sha256="3" * 64,
        chunk_count=sum(item.chunk_count for item in shards),
        shard_size_limit=3,
        shards=shards,
        dtype="float16",
        metric="inner_product",
        candidate_index="faiss-hnsw-flat-v1",
        hnsw_m=32,
        ef_construction=200,
        ef_search=512,
        candidate_count=3,
        recall_at_300=1.0,
        recall_queries_sha256="4" * 64,
        recall_query_count=4,
    )


def test_sharded_dense_search_uses_every_shard_and_globally_rescores_candidates(
    tmp_path: Path,
) -> None:
    chunk_b = _chunk("b")
    chunk_c = _chunk("c")
    chunk_z = _chunk("z")
    chunk_a = _chunk("a")
    chunk_d = _chunk("d")
    first = _write_shard(
        tmp_path,
        0,
        0,
        np.asarray([[0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.8, 0.2, 0.0]]),
        [chunk_b, None, chunk_c],
        [0, 2],
    )
    second = _write_shard(
        tmp_path,
        1,
        3,
        np.asarray([[0.95, 0.05, 0.0], [0.9, 0.1, 0.0], [0.7, 0.3, 0.0]]),
        [chunk_z, chunk_a, chunk_d],
        [0, 1, 2],
    )
    loaded_paths: list[str] = []
    scientific = ScientificConfig().model_copy(
        update={
            "embedding_dimension": 3,
            "dense_candidate_count": 3,
        }
    )

    def load_candidate(path: Path, *, ef_search: int) -> RowsCandidateIndex:
        assert ef_search == 512
        loaded_paths.append(path.parent.name)
        return RowsCandidateIndex(json.loads(path.read_text(encoding="utf-8")))

    index = ShardedDenseIndex(
        root=tmp_path,
        manifest=_manifest([first, second], scientific),
        encoder=QueryEncoder(),
        candidate_loader=load_candidate,
        scientific=scientific,
    )

    results = index.search(
        "visible query",
        candidates=3,
        limit=5,
        query_kind="context",
    )

    assert loaded_paths == ["shard-00000", "shard-00001"]
    assert [item.chunk_id for item in results] == [
        chunk_z.id,
        min(chunk_a.id, chunk_b.id),
        max(chunk_a.id, chunk_b.id),
        chunk_c.id,
        chunk_d.id,
    ]
    assert [item.rank for item in results] == [1, 2, 3, 4, 5]
    assert all(item.retriever == "dense" for item in results)
    assert all(item.corpus_manifest_hash == "1" * 64 for item in results)
    assert all(item.index_manifest_hash == index.manifest.manifest_hash() for item in results)


def test_sharded_manifest_rejects_missing_overlapping_or_reordered_ranges(
    tmp_path: Path,
) -> None:
    first = _write_shard(
        tmp_path,
        0,
        0,
        np.asarray([[1.0, 0.0, 0.0]]),
        [_chunk("first")],
        [0],
    )
    second = _write_shard(
        tmp_path,
        1,
        1,
        np.asarray([[1.0, 0.0, 0.0]]),
        [_chunk("second")],
        [0],
    )
    scientific = ScientificConfig().model_copy(
        update={"embedding_dimension": 3, "dense_candidate_count": 3}
    )
    valid = _manifest([first, second], scientific)

    for shards in ([second, first], [first, second.model_copy(update={"start_row": 0})]):
        with pytest.raises(ValidationError, match="consecutive|ordered"):
            ShardedDenseIndexManifest.model_validate(
                valid.model_dump(mode="json") | {
                    "shards": [item.model_dump(mode="json") for item in shards]
                }
            )


def test_sharded_dense_runtime_rejects_a_corrupt_completed_artifact(
    tmp_path: Path,
) -> None:
    shard = _write_shard(
        tmp_path,
        0,
        0,
        np.asarray([[1.0, 0.0, 0.0]]),
        [_chunk("first")],
        [0],
    )
    scientific = ScientificConfig().model_copy(
        update={"embedding_dimension": 3, "dense_candidate_count": 3}
    )
    (tmp_path / shard.path / "candidate.faiss").write_text(
        "tampered",
        encoding="utf-8",
    )

    with pytest.raises(IncompatibleDenseIndexError, match="checksum"):
        ShardedDenseIndex(
            root=tmp_path,
            manifest=_manifest([shard], scientific),
            encoder=QueryEncoder(),
            candidate_loader=lambda *_args, **_kwargs: RowsCandidateIndex([0]),
            scientific=scientific,
        )
