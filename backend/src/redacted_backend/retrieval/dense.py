from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.types import DenseIndexManifest, RankedChunk
from redacted_backend.scientific_config import ScientificConfig


class DenseRecallError(ValueError):
    pass


class IncompatibleDenseIndexError(ValueError):
    pass


class SentenceModel(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        **kwargs: Any,
    ) -> NDArray[np.floating[Any]]: ...


class CandidateIndex(Protocol):
    def search(
        self,
        query: NDArray[np.float32],
        candidates: int,
    ) -> NDArray[np.int64]: ...


class QueryEncoder(Protocol):
    def encode_query(self, text: str) -> NDArray[np.float32]: ...


ModelFactory = Callable[[str, str], SentenceModel]
Ranker = Callable[[str], Sequence[str]]


class BgeEncoder:
    def __init__(
        self,
        *,
        scientific: ScientificConfig | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.scientific = scientific or ScientificConfig()
        self._model_factory = model_factory or _load_sentence_transformer
        self._model: SentenceModel | None = None

    def encode_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.scientific.embedding_dimension), dtype=np.float32)
        return _normalize_rows(self._encode(list(texts)))

    def encode_query(self, text: str) -> NDArray[np.float32]:
        if not text.strip():
            raise ValueError("Dense query must not be empty")
        instructed = f"{self.scientific.embedding_query_instruction}{text}"
        return _normalize_rows(self._encode([instructed]))[0]

    def _encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        model = self._model
        if model is None:
            model = self._model_factory(
                self.scientific.embedding_model,
                self.scientific.embedding_model_revision,
            )
            self._model = model
        encoded = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        values = np.asarray(encoded, dtype=np.float32)
        if values.shape != (len(texts), self.scientific.embedding_dimension):
            raise ValueError("BGE encoder returned an unexpected matrix shape")
        return values


class DenseIndex:
    def __init__(
        self,
        *,
        chunks: Sequence[CorpusChunk],
        embeddings: NDArray[np.floating[Any]],
        candidate_index: CandidateIndex,
        encoder: QueryEncoder,
        manifest: DenseIndexManifest,
        corpus_manifest_hash: str,
        scientific: ScientificConfig | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.embeddings = embeddings
        self.candidate_index = candidate_index
        self.encoder = encoder
        self.manifest = manifest
        self.corpus_manifest_hash = corpus_manifest_hash
        self.scientific = scientific or ScientificConfig()

    def search(
        self,
        query: str,
        candidates: int = 2_000,
        limit: int = 300,
        *,
        query_kind: Literal["context", "own"] = "context",
    ) -> list[RankedChunk]:
        self._validate()
        if limit < 1 or limit > self.scientific.top_chunks_per_list:
            raise ValueError(
                f"Dense result limit must be between 1 and {self.scientific.top_chunks_per_list}"
            )
        if candidates < limit or candidates > self.manifest.candidate_count:
            raise ValueError(
                "Dense candidate count must cover the result limit and not exceed the manifest"
            )
        query_vector = np.asarray(self.encoder.encode_query(query), dtype=np.float32)
        if query_vector.shape != (self.manifest.embedding_dimension,):
            raise IncompatibleDenseIndexError(
                "Dense query dimension does not match the index manifest"
            )
        candidate_rows = np.asarray(
            self.candidate_index.search(query_vector, candidates),
            dtype=np.int64,
        ).reshape(-1)
        candidate_rows = np.unique(candidate_rows)
        if np.any(candidate_rows < 0) or np.any(candidate_rows >= len(self.chunks)):
            raise IncompatibleDenseIndexError("Dense candidate index returned an invalid row")
        exact_vectors = np.asarray(self.embeddings[candidate_rows], dtype=np.float32)
        scores = exact_vectors @ query_vector
        rescored = sorted(
            (
                (float(score), self.chunks[int(row)])
                for row, score in zip(candidate_rows, scores, strict=True)
            ),
            key=lambda item: (-item[0], item[1].id),
        )[:limit]
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
                corpus_manifest_hash=self.corpus_manifest_hash,
            )
            for rank, (score, chunk) in enumerate(rescored, start=1)
        ]

    def _validate(self) -> None:
        if self.manifest.model_name != self.scientific.embedding_model:
            raise IncompatibleDenseIndexError("Dense model name mismatch")
        if self.manifest.model_revision != self.scientific.embedding_model_revision:
            raise IncompatibleDenseIndexError("Dense model revision mismatch")
        if self.manifest.scientific_hash != self.scientific.manifest_hash():
            raise IncompatibleDenseIndexError("Dense scientific manifest mismatch")
        if self.manifest.corpus_manifest_hash != self.corpus_manifest_hash:
            raise IncompatibleDenseIndexError("Dense corpus manifest mismatch")
        if self.manifest.chunk_count != len(self.chunks):
            raise IncompatibleDenseIndexError("Dense chunk count mismatch")
        if self.embeddings.shape != (
            len(self.chunks),
            self.manifest.embedding_dimension,
        ):
            raise IncompatibleDenseIndexError("Dense embedding matrix shape mismatch")
        if self.manifest.embedding_dimension != self.scientific.embedding_dimension:
            raise IncompatibleDenseIndexError("Dense embedding dimension mismatch")
        if (
            self.manifest.candidate_index != self.scientific.dense_index_version
            or self.manifest.hnsw_m != self.scientific.dense_hnsw_m
            or self.manifest.ef_construction
            != self.scientific.dense_ef_construction
            or self.manifest.ef_search != self.scientific.dense_ef_search
            or self.manifest.candidate_count
            != self.scientific.dense_candidate_count
        ):
            raise IncompatibleDenseIndexError("Dense index parameters mismatch")
        require_dense_recall(
            self.manifest.recall_at_300,
            minimum=self.scientific.dense_recall_minimum,
        )


def measure_recall_at_300(
    queries: Sequence[str],
    approximate: Ranker,
    exact: Ranker,
    *,
    limit: int = 300,
) -> float:
    if not queries:
        raise ValueError("Dense recall sample must contain at least one query")
    if limit < 1:
        raise ValueError("Dense recall limit must be positive")
    recalls: list[float] = []
    for query in queries:
        expected = set(exact(query)[:limit])
        if not expected:
            raise ValueError("Exact dense ranking returned no reference results")
        returned = set(approximate(query)[:limit])
        recalls.append(len(returned & expected) / len(expected))
    return sum(recalls) / len(recalls)


def require_dense_recall(recall: float, *, minimum: float = 0.98) -> None:
    if recall < minimum:
        raise DenseRecallError(
            f"Dense recall@300 must be at least {minimum:.4f}; got {recall:.4f}"
        )


def _normalize_rows(values: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("BGE encoder returned a non-finite or zero vector")
    return np.asarray(values / norms, dtype=np.float32)


def _load_sentence_transformer(model_name: str, revision: str) -> SentenceModel:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is required for real BGE encoding"
        ) from error
    return SentenceTransformer(
        model_name,
        revision=revision,
        trust_remote_code=False,
    )
