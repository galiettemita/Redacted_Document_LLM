from __future__ import annotations

from datetime import date
from uuid import UUID

import pytest

from redacted_backend.domain.visible_examples import LengthEstimate, VisibleExample
from redacted_backend.retrieval.context_query import (
    ContextQueryError,
    build_context_query,
    prepare_context_query,
)


def test_context_query_removes_frozen_stopwords_without_stemming_names() -> None:
    query = build_context_query(
        "The Kennedy administration was discussing treaties in the cities."
    )

    assert query == "kennedy administration discussing treaties cities"


def test_context_query_normalizes_visible_unicode_and_punctuation() -> None:
    query = build_context_query("Café talks—between NATO and Türkiye—continued.")

    assert query == "café talks nato türkiye continued"


def test_context_query_removes_non_visible_structural_markers() -> None:
    query = build_context_query("Visible before [[TARGET]] and [REDACTED] after")

    assert query == "visible before after"


def test_prepared_context_query_has_stable_text_hash_count_and_versions() -> None:
    prepared = prepare_context_query(_visible_example())

    assert prepared.model_dump() == {
        "text": "visible before after",
        "sha256": "a98b530a96702a01a5fbdb3d154f4588f6cacd21da68dd20efaf298bc073a549",
        "term_count": 3,
        "query_version": "context-query-v1",
        "stopwords_version": "visible-english-v1",
    }


@pytest.mark.parametrize(
    "context_window",
    (
        "visible before visible after",
        "visible [[TARGET]] middle [[TARGET]] after",
    ),
)
def test_prepared_context_query_requires_exactly_one_target_marker(
    context_window: str,
) -> None:
    with pytest.raises(ContextQueryError, match="exactly one target marker"):
        prepare_context_query(_visible_example(context_window=context_window))


def test_context_query_rejects_a_window_containing_only_stopwords() -> None:
    with pytest.raises(ContextQueryError, match="no searchable visible terms"):
        build_context_query("The and of in to.")


def test_context_query_rejects_structural_markers_without_visible_terms() -> None:
    with pytest.raises(ContextQueryError, match="no searchable visible terms"):
        build_context_query("The [[TARGET]] and [REDACTED]")


def _visible_example(
    *,
    context_window: str = "visible before [[TARGET]] [REDACTED] after",
) -> VisibleExample:
    return VisibleExample(
        passage_id=UUID("11111111-1111-4111-8111-111111111111"),
        document_id=UUID("22222222-2222-4222-8222-222222222222"),
        document_date=date(1945, 5, 8),
        before_target="visible before",
        after_target="after",
        context_window=context_window,
        length_cue=LengthEstimate(
            estimated_words=8,
            bucket="part of a sentence",
            median_character_width=0.01,
            median_characters_per_word=5,
            line_capacities=[48],
        ),
        visible_words=[],
        serialization_version="visible-transcript-v1",
    )
