from __future__ import annotations

import hashlib

import pytest

from redacted_backend.domain.retrieval import FourRankedLists
from redacted_backend.retrieval.fusion import fuse_documents
from redacted_backend.retrieval.types import RankedChunk


def test_document_rrf_uses_each_lists_best_chunk_and_chunk_rrf_selects_two() -> None:
    lists = FourRankedLists(
        context_bm25=[_ranked("doc-a", "a1", 1, "bm25", "context")],
        context_dense=[
            _ranked("doc-a", "a2", 2, "dense", "context"),
            _ranked("doc-a", "a1", 3, "dense", "context"),
        ],
        own_bm25=[_ranked("doc-a", "a1", 4, "bm25", "own")],
        own_dense=[_ranked("doc-a", "a3", 5, "dense", "own")],
    )

    fused = fuse_documents(lists, rrf_k=60, limit=100, chunks_per_document=2)

    assert len(fused) == 1
    assert fused[0].rrf_score == pytest.approx(
        1 / 61 + 1 / 62 + 1 / 64 + 1 / 65
    )
    assert [chunk.chunk_id for chunk in fused[0].chunks] == [
        _chunk_id("a1"),
        _chunk_id("a2"),
    ]


def test_fusion_merges_archive_copies_by_filename_or_opening_and_uses_stable_id() -> None:
    by_filename = _ranked(
        "release-z",
        "one",
        1,
        "bm25",
        "context",
        title="Cable Copy FINAL.pdf",
        text="An opening that differs",
    )
    same_filename = _ranked(
        "release-a",
        "two",
        1,
        "dense",
        "context",
        title="cable copy.pdf",
        text="Another opening entirely",
    )
    opening = " ".join(f"visible{i}" for i in range(130))
    by_opening = _ranked(
        "release-b",
        "three",
        1,
        "bm25",
        "own",
        title="unrelated-name.txt",
        text=opening,
    )
    same_opening = _ranked(
        "release-c",
        "four",
        1,
        "dense",
        "own",
        title="different-name.txt",
        text=f"{opening} ignored tail",
    )

    fused = fuse_documents(
        FourRankedLists(
            context_bm25=[by_filename],
            context_dense=[same_filename],
            own_bm25=[by_opening],
            own_dense=[same_opening],
        )
    )

    assert [(item.document_id, item.source_document_ids) for item in fused] == [
        ("release-a", ["release-a", "release-z"]),
        ("release-b", ["release-b", "release-c"]),
    ]


def test_fusion_keeps_top_100_and_breaks_equal_scores_by_document_id() -> None:
    context = [
        _ranked(f"doc-{index:03d}", f"chunk-{index}", index + 1, "bm25", "context")
        for index in range(101)
    ]

    fused = fuse_documents(
        FourRankedLists(
            context_bm25=context,
            context_dense=[],
            own_bm25=[],
            own_dense=[],
        ),
        limit=100,
    )

    assert len(fused) == 100
    assert fused[0].document_id == "doc-000"
    assert fused[-1].document_id == "doc-099"


def _ranked(
    document_id: str,
    chunk_name: str,
    rank: int,
    retriever: str,
    query_kind: str,
    *,
    title: str | None = None,
    text: str | None = None,
) -> RankedChunk:
    return RankedChunk(
        chunk_id=_chunk_id(chunk_name),
        document_id=document_id,
        ordinal=rank,
        text=text or f"Evidence for {chunk_name}",
        title=title or f"{document_id}.txt",
        corpus="cfpf",
        source_url=None,
        classification=None,
        authored=None,
        doc_lang="en",
        rank=rank,
        score=100.0 - rank,
        retriever=retriever,
        query_kind=query_kind,
        query_sha256="1" * 64,
        index_manifest_hash="2" * 64,
        corpus_manifest_hash="3" * 64,
    )


def _chunk_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
