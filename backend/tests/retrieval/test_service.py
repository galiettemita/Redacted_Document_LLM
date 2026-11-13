from __future__ import annotations

import hashlib
from datetime import date
from uuid import UUID

import pytest

from redacted_backend.domain.visible_examples import LengthEstimate, VisibleExample
from redacted_backend.retrieval.service import RealisticRetrievalService, RetrievalError
from redacted_backend.retrieval.types import RankedChunk


class RecordingRetriever:
    def __init__(self, retriever: str, results: list[RankedChunk]) -> None:
        self.retriever = retriever
        self.results = results
        self.calls: list[tuple[str, int, str]] = []

    def search(self, query: str, limit: int, *, query_kind: str) -> list[RankedChunk]:
        self.calls.append((query, limit, query_kind))
        return [
            result.model_copy(
                update={
                    "retriever": self.retriever,
                    "query_kind": query_kind,
                    "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                }
            )
            for result in self.results[:limit]
        ]


def test_realistic_service_makes_exactly_four_top_300_calls() -> None:
    results = [_ranked(index) for index in range(300)]
    bm25 = RecordingRetriever("bm25", results)
    dense = RecordingRetriever("dense", results)
    service = RealisticRetrievalService(bm25=bm25, dense=dense)

    trace = service.retrieve_realistic(_example(), "specific own query")

    assert [(limit, kind) for _, limit, kind in bm25.calls] == [
        (300, "context"),
        (300, "own"),
    ]
    assert [(limit, kind) for _, limit, kind in dense.calls] == [
        (300, "context"),
        (300, "own"),
    ]
    assert len(trace.lists.context_bm25) == 300
    assert len(trace.lists.context_dense) == 300
    assert len(trace.lists.own_bm25) == 300
    assert len(trace.lists.own_dense) == 300
    assert len(trace.documents) == 100
    assert trace.serialized_documents.startswith("[DOCUMENT 1]\n[EXCERPT 1]")
    assert "[[TARGET]]" not in bm25.calls[0][0]


def test_realistic_service_fails_instead_of_claiming_empty_evidence() -> None:
    service = RealisticRetrievalService(
        bm25=RecordingRetriever("bm25", []),
        dense=RecordingRetriever("dense", []),
    )

    with pytest.raises(RetrievalError, match="no_evidence_retrieved") as error:
        service.retrieve_realistic(_example(), "specific own query")

    assert error.value.code == "no_evidence_retrieved"


def _example() -> VisibleExample:
    return VisibleExample(
        passage_id=UUID("00000000-0000-0000-0000-000000000001"),
        document_id=UUID("00000000-0000-0000-0000-000000000002"),
        document_date=date(2019, 3, 24),
        before_target="Visible bulletin text before the hidden passage.",
        after_target="Visible bulletin text after the hidden passage.",
        context_window=(
            "Visible bulletin text before [[TARGET]] visible bulletin text after."
        ),
        length_cue=LengthEstimate(
            estimated_words=7,
            bucket="part of a sentence",
            median_character_width=0.01,
            median_characters_per_word=5.0,
            line_capacities=[35],
        ),
        visible_words=[],
        serialization_version="visible-transcript-v1",
    )


def _ranked(index: int) -> RankedChunk:
    value = f"chunk-{index}"
    return RankedChunk(
        chunk_id=hashlib.sha256(value.encode()).hexdigest(),
        document_id=f"document-{index}",
        ordinal=0,
        text=f"Unrelated public archive evidence {index}",
        title=f"archive-{index}.txt",
        corpus="cfpf",
        source_url=None,
        classification=None,
        authored=None,
        doc_lang="en",
        rank=index + 1,
        score=float(300 - index),
        retriever="bm25",
        query_kind="context",
        query_sha256="1" * 64,
        index_manifest_hash="2" * 64,
        corpus_manifest_hash="3" * 64,
    )
