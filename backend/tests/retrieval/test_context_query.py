from __future__ import annotations

import pytest

from redacted_backend.retrieval.context_query import (
    ContextQueryError,
    build_context_query,
)


def test_context_query_removes_frozen_stopwords_without_stemming_names() -> None:
    query = build_context_query(
        "The Kennedy administration was discussing treaties in the cities."
    )

    assert query == "kennedy administration discussing treaties cities"


def test_context_query_normalizes_visible_unicode_and_punctuation() -> None:
    query = build_context_query("Café talks—between NATO and Türkiye—continued.")

    assert query == "café talks nato türkiye continued"


def test_context_query_rejects_a_window_containing_only_stopwords() -> None:
    with pytest.raises(ContextQueryError, match="no searchable visible terms"):
        build_context_query("The and of in to.")
