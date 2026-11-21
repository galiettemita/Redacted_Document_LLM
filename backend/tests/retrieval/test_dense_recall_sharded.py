from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.dense_sharded import (
    certify_sharded_dense_recall,
    select_certification_queries,
)
from redacted_backend.retrieval.types import (
    DenseShardManifest,
    ShardedDenseIndexManifest,
)
from redacted_backend.scientific_config import ScientificConfig


class Encoder:
    def encode_query(self, text: str) -> NDArray[np.float32]:
        del text
        return np.asarray([1.0, 0.0], dtype=np.float32)


class Candidate:
    def __init__(self, rows: list[int]) -> None:
        self.rows = rows

    def search(
        self,
        query: NDArray[np.float32],
        candidates: int,
    ) -> NDArray[np.int64]:
        del query
        return np.asarray(self.rows[:candidates], dtype=np.int64)


def _chunk(index: int) -> CorpusChunk:
    text = f"opening archive words {index}"
    return CorpusChunk(
        id=hashlib.sha256(f"chunk-{index}".encode()).hexdigest(),
        document_id=f"document-{index}",
        ordinal=0,
        text=text,
        word_count=4,
        title=f"Public title {index}",
        corpus="historylab-unlabeled",
        source_url=None,
        classification=None,
        authored=None,
        doc_lang=None,
    )


def test_certification_query_selection_is_stable_and_hashes_published_bytes(
    tmp_path: Path,
) -> None:
    chunks = [_chunk(index) for index in range(5)]
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    application = tmp_path / "application.txt"
    application.write_text("visible public query A\nvisible public query B\n", encoding="utf-8")
    output = tmp_path / "combined.txt"

    queries = select_certification_queries(
        chunks_path,
        application,
        output,
        application_query_count=2,
        corpus_query_count=2,
    )

    assert queries == [
        "visible public query A",
        "visible public query B",
        "Public title 1 opening archive words 1",
        "Public title 0 opening archive words 0",
    ]
    assert output.read_text(encoding="utf-8") == "\n".join(queries) + "\n"
    assert hashlib.sha256(output.read_bytes()).hexdigest() == (
        "7eb395ce98f7149e54fdcd20e9a173fd168418b3489442815cd214729ce99c41"
    )


def _write_recall_shard(
    root: Path,
    shard_id: int,
    start: int,
    scores: list[float],
    candidate_rows: list[int],
) -> DenseShardManifest:
    directory = root / f"shard-{shard_id:05d}"
    directory.mkdir(parents=True)
    vectors = np.asarray([[score, 1.0 - score] for score in scores], dtype=np.float16)
    embeddings = directory / "embeddings.f16"
    vectors.tofile(embeddings)
    metadata = directory / "metadata.jsonl"
    offsets: list[int] = []
    with metadata.open("wb") as destination:
        for index in range(len(scores)):
            offsets.append(destination.tell())
            destination.write(_chunk(start + index).model_dump_json().encode() + b"\n")
    offset_path = directory / "metadata.offsets.u64"
    np.asarray(offsets, dtype="<u8").tofile(offset_path)
    candidate = directory / "candidate.faiss"
    candidate.write_text(json.dumps(candidate_rows), encoding="utf-8")
    return DenseShardManifest(
        shard_id=shard_id,
        path=directory.name,
        start_row=start,
        end_row=start + len(scores),
        chunk_count=len(scores),
        embeddings_sha256=hashlib.sha256(embeddings.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        offsets_sha256=hashlib.sha256(offset_path.read_bytes()).hexdigest(),
        candidate_index_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    )


def test_sharded_recall_compares_candidate_union_with_streamed_global_exact_top_k(
    tmp_path: Path,
) -> None:
    scientific = ScientificConfig().model_copy(
        update={"embedding_dimension": 2, "dense_candidate_count": 2}
    )
    first = _write_recall_shard(tmp_path, 0, 0, [0.9, 0.1], [0])
    second = _write_recall_shard(tmp_path, 1, 2, [0.8, 0.7], [0, 1])
    manifest = ShardedDenseIndexManifest(
        model_name="BAAI/bge-base-en-v1.5",
        model_revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        embedding_dimension=2,
        corpus_manifest_hash="1" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunks_sha256="3" * 64,
        chunk_count=4,
        shard_size_limit=2,
        shards=[first, second],
        dtype="float16",
        metric="inner_product",
        candidate_index="faiss-hnsw-flat-v1",
        hnsw_m=32,
        ef_construction=200,
        ef_search=512,
        candidate_count=2,
        recall_at_300=1.0,
        recall_queries_sha256="4" * 64,
        recall_query_count=1,
    )

    def loader(path: Path, *, ef_search: int) -> Candidate:
        del ef_search
        return Candidate(json.loads(path.read_text(encoding="utf-8")))

    recall = certify_sharded_dense_recall(
        tmp_path,
        manifest,
        ["query"],
        encoder=Encoder(),
        candidate_loader=loader,
        scientific=scientific,
        limit=2,
        block_rows=1,
    )

    assert recall == 1.0
