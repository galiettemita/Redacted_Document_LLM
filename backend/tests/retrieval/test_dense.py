from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.dense import (
    BgeEncoder,
    DenseIndex,
    IncompatibleDenseIndexError,
)
from redacted_backend.retrieval.types import DenseIndexManifest
from redacted_backend.scientific_config import ScientificConfig


class RecordingSentenceModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, sentences: Sequence[str], **_kwargs: Any) -> np.ndarray:
        self.calls.append(list(sentences))
        return np.asarray(
            [[3.0, 4.0] if "first" in sentence else [0.0, 2.0] for sentence in sentences],
            dtype=np.float32,
        )


class RecordingModelFactory:
    def __init__(self, model: RecordingSentenceModel) -> None:
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def __call__(self, model_name: str, revision: str) -> RecordingSentenceModel:
        self.calls.append((model_name, revision))
        return self.model


class FixedQueryEncoder:
    def __init__(self, vector: Sequence[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode_query(self, _text: str) -> np.ndarray:
        return self.vector


class FixedCandidateIndex:
    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.requests: list[int] = []

    def search(self, _query: np.ndarray, candidates: int) -> np.ndarray:
        self.requests.append(candidates)
        return self.indices[:candidates]


def test_bge_encoder_pins_revision_distinguishes_queries_and_normalizes() -> None:
    scientific = ScientificConfig(embedding_dimension=2)
    model = RecordingSentenceModel()
    factory = RecordingModelFactory(model)
    encoder = BgeEncoder(scientific=scientific, model_factory=factory)

    documents = encoder.encode_documents(["first document", "second document"])
    query = encoder.encode_query("first query")

    assert factory.calls == [
        (scientific.embedding_model, scientific.embedding_model_revision)
    ]
    assert model.calls == [
        ["first document", "second document"],
        [f"{scientific.embedding_query_instruction}first query"],
    ]
    np.testing.assert_allclose(np.linalg.norm(documents, axis=1), [1.0, 1.0])
    np.testing.assert_allclose(np.linalg.norm(query), 1.0)
    np.testing.assert_allclose(documents[0], [0.6, 0.8])


def test_dense_search_exactly_rescores_candidates_and_uses_stable_ties() -> None:
    scientific = ScientificConfig(embedding_dimension=2)
    chunks = [_chunk(index) for index in range(4)]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float16,
    )
    candidate_index = FixedCandidateIndex([2, 1, 3, 0])
    manifest = _dense_manifest(scientific, len(chunks), dimension=2)
    index = DenseIndex(
        chunks=chunks,
        embeddings=embeddings,
        candidate_index=candidate_index,
        encoder=FixedQueryEncoder([1.0, 0.0]),
        manifest=manifest,
        corpus_manifest_hash="1" * 64,
        scientific=scientific,
    )

    results = index.search("visible query", candidates=4, limit=3)

    tied = sorted((chunks[0].id, chunks[3].id))
    assert [result.chunk_id for result in results] == [*tied, chunks[1].id]
    assert [result.score for result in results] == pytest.approx(
        [1.0, 1.0, 0.8],
        abs=1e-3,
    )
    assert [result.rank for result in results] == [1, 2, 3]
    assert all(result.retriever == "dense" for result in results)
    assert all(result.query_kind == "context" for result in results)
    assert all(result.index_manifest_hash == manifest.manifest_hash() for result in results)
    assert candidate_index.requests == [4]


def test_dense_search_rejects_an_incompatible_model_revision() -> None:
    scientific = ScientificConfig(embedding_dimension=2)
    incompatible = _dense_manifest(scientific, 1, dimension=2).model_copy(
        update={"model_revision": "0" * 40}
    )
    index = DenseIndex(
        chunks=[_chunk(0)],
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float16),
        candidate_index=FixedCandidateIndex([0]),
        encoder=FixedQueryEncoder([1.0, 0.0]),
        manifest=incompatible,
        corpus_manifest_hash="1" * 64,
        scientific=scientific,
    )

    with pytest.raises(IncompatibleDenseIndexError, match="model revision"):
        index.search("visible query", candidates=1, limit=1)


def _chunk(index: int) -> CorpusChunk:
    text = f"Visible dense context {index}"
    return CorpusChunk(
        id=hashlib.sha256(f"chunk-{index}".encode()).hexdigest(),
        document_id=f"document-{index}",
        ordinal=index,
        text=text,
        word_count=len(text.split()),
        title=f"Title {index}",
        corpus="cfpf",
        source_url="https://example.test/source",
        classification="UNCLASSIFIED",
        authored="Visible author",
        doc_lang="en",
    )


def _dense_manifest(
    scientific: ScientificConfig,
    chunk_count: int,
    *,
    dimension: int,
) -> DenseIndexManifest:
    return DenseIndexManifest(
        model_name=scientific.embedding_model,
        model_revision=scientific.embedding_model_revision,
        embedding_dimension=dimension,
        corpus_manifest_hash="1" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunks_sha256="2" * 64,
        embeddings_sha256="3" * 64,
        metadata_sha256="4" * 64,
        candidate_index_sha256="5" * 64,
        chunk_count=chunk_count,
        dtype="float16",
        metric="inner_product",
        candidate_index="faiss-hnsw-flat-v1",
        hnsw_m=32,
        ef_construction=200,
        ef_search=512,
        candidate_count=2_000,
        recall_at_300=0.99,
        recall_sample_sha256="6" * 64,
    )
