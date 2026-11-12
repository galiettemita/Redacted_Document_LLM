from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import pytest

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.retrieval.bm25 import (
    Bm25Index,
    IncompatibleIndexManifestError,
)
from redacted_backend.scientific_config import ScientificConfig


class RecordingTransport:
    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.index_definition: dict[str, Any] | None = None
        self.indexed_documents: list[dict[str, Any]] = []
        self.search_request: dict[str, Any] | None = None
        self.refreshed_index_name: str | None = None
        self.hits = hits or []

    def replace_index(self, index_name: str, definition: dict[str, Any]) -> None:
        self.index_definition = definition

    def bulk_index(
        self,
        index_name: str,
        documents: Sequence[dict[str, Any]],
    ) -> None:
        self.indexed_documents.extend(documents)

    def refresh(self, index_name: str) -> None:
        self.refreshed_index_name = index_name

    def search(self, index_name: str, request: dict[str, Any]) -> list[dict[str, Any]]:
        self.search_request = request
        return self.hits


def chunk(ordinal: int = 0) -> CorpusChunk:
    text = f"Visible archive context {ordinal}"
    return CorpusChunk(
        id=hashlib.sha256(f"chunk-{ordinal}".encode()).hexdigest(),
        document_id=f"document-{ordinal}",
        ordinal=ordinal,
        text=text,
        word_count=len(text.split()),
        title=f"Title {ordinal}",
        corpus="cfpf",
        source_url="https://example.test/source",
        classification="UNCLASSIFIED",
        authored="Visible author",
        doc_lang="en",
    )


def corpus_manifest(scientific: ScientificConfig, chunk_count: int) -> CorpusManifest:
    return CorpusManifest(
        schema_version="historylab-normalized-v1",
        source_manifest_hash="1" * 64,
        coverage_report_hash="4" * 64,
        exclusion_hash="2" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=chunk_count,
        chunk_count=chunk_count,
        corpus_counts={"cfpf": chunk_count},
        exclusion_counts={},
        coverage_ready=True,
        chunks_sha256="3" * 64,
    )


def search_hit(item: CorpusChunk, score: float = 1.0) -> dict[str, Any]:
    return {
        "_id": item.id,
        "_score": score,
        "_source": {
            "chunk_id": item.id,
            "document_id": item.document_id,
            "ordinal": item.ordinal,
            "text": item.text,
            "title": item.title,
            "corpus": item.corpus,
            "source_url": item.source_url,
            "classification": item.classification,
            "authored": item.authored,
            "doc_lang": item.doc_lang,
        },
    }


def test_build_pins_standard_unicode_analysis_stopwords_and_bm25() -> None:
    scientific = ScientificConfig()
    transport = RecordingTransport()
    item = chunk()
    index = Bm25Index(transport, scientific=scientific)

    manifest = index.build([item], corpus_manifest(scientific, 1))

    assert transport.index_definition == {
        "settings": {
            "index": {
                "similarity": {"default": {"type": "BM25", "k1": 1.2, "b": 0.75}}
            },
            "analysis": {
                "filter": {
                    "visible_english_stop": {
                        "type": "stop",
                        "stopwords": list(scientific.context_stopwords),
                    }
                },
                "analyzer": {
                    "visible_english": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "visible_english_stop"],
                    }
                },
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "ordinal": {"type": "integer"},
                "text": {"type": "text", "analyzer": "visible_english"},
                "title": {"type": "keyword", "index": False},
                "corpus": {"type": "keyword"},
                "source_url": {"type": "keyword", "index": False},
                "classification": {"type": "keyword", "index": False},
                "authored": {"type": "keyword", "index": False},
                "doc_lang": {"type": "keyword", "index": False},
            },
        },
    }
    assert transport.indexed_documents[0]["chunk_id"] == item.id
    assert transport.refreshed_index_name == scientific.bm25_index_name
    assert manifest.chunk_count == 1
    assert manifest.scientific_hash == scientific.manifest_hash()


def test_search_never_returns_more_than_the_exact_300_result_limit() -> None:
    scientific = ScientificConfig()
    items = [chunk(index) for index in range(305)]
    transport = RecordingTransport([search_hit(item) for item in items])
    index = Bm25Index(transport, scientific=scientific)
    index.build(items, corpus_manifest(scientific, len(items)))

    results = index.search("visible context", limit=300)

    assert len(results) == 300
    assert transport.search_request is not None
    assert transport.search_request["size"] == 300
    with pytest.raises(ValueError, match="cannot exceed 300"):
        index.search("visible context", limit=301)


def test_equal_scores_use_stable_chunk_id_order() -> None:
    scientific = ScientificConfig()
    items = [chunk(index) for index in range(3)]
    transport = RecordingTransport(
        [search_hit(item, score=5.0) for item in reversed(items)]
    )
    index = Bm25Index(transport, scientific=scientific)
    index.build(items, corpus_manifest(scientific, len(items)))

    results = index.search("visible context")

    assert [result.chunk_id for result in results] == sorted(item.id for item in items)
    assert [result.rank for result in results] == [1, 2, 3]


def test_search_result_records_context_query_and_index_provenance() -> None:
    scientific = ScientificConfig()
    item = chunk()
    transport = RecordingTransport([search_hit(item, score=7.5)])
    corpus = corpus_manifest(scientific, 1)
    index = Bm25Index(transport, scientific=scientific)
    index_manifest = index.build([item], corpus)

    result = index.search("visible context")[0]

    assert result.retriever == "bm25"
    assert result.query_kind == "context"
    assert result.query_sha256 == hashlib.sha256(b"visible context").hexdigest()
    assert result.index_manifest_hash == index_manifest.manifest_hash()
    assert result.corpus_manifest_hash == corpus.manifest_hash()
    assert result.text == item.text


def test_build_rejects_a_corpus_from_another_scientific_manifest() -> None:
    scientific = ScientificConfig()
    incompatible = corpus_manifest(scientific, 1).model_copy(
        update={"scientific_hash": "0" * 64}
    )
    index = Bm25Index(RecordingTransport(), scientific=scientific)

    with pytest.raises(IncompatibleIndexManifestError, match="scientific manifest"):
        index.build([chunk()], incompatible)


def test_search_rejects_a_loaded_index_with_an_incompatible_definition() -> None:
    scientific = ScientificConfig()
    item = chunk()
    corpus = corpus_manifest(scientific, 1)
    transport = RecordingTransport([search_hit(item)])
    index = Bm25Index(transport, scientific=scientific)
    manifest = index.build([item], corpus)
    index.index_manifest = manifest.model_copy(
        update={"index_definition_hash": "0" * 64}
    )

    with pytest.raises(IncompatibleIndexManifestError, match="index definition"):
        index.search("visible context")
