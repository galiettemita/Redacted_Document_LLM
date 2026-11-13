from __future__ import annotations

import pytest

from redacted_backend.retrieval.dense import (
    DenseRecallError,
    measure_recall_at_300,
    require_dense_recall,
)


def test_recall_at_300_uses_exact_result_membership_for_each_query() -> None:
    approximate = {
        "first": ["a", "b"],
        "second": ["d", "e"],
    }
    exact = {
        "first": ["a", "c"],
        "second": ["d", "e"],
    }

    recall = measure_recall_at_300(
        ["first", "second"],
        lambda query: approximate[query],
        lambda query: exact[query],
        limit=2,
    )

    assert recall == 0.75


def test_dense_readiness_fails_below_point_98_recall() -> None:
    with pytest.raises(DenseRecallError, match="0.9800"):
        require_dense_recall(0.9799, minimum=0.98)

    require_dense_recall(0.98, minimum=0.98)
