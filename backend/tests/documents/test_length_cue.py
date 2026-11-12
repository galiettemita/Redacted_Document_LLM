from __future__ import annotations

import pytest

from redacted_backend.documents.length_cue import (
    LengthCueError,
    estimate_length_cue,
)
from redacted_backend.domain.passages import PassageRect
from tests.documents.factories import word

NEARBY_WORDS = [
    word("alpha", x=0.05, y=0.1, width=0.05),
    word("bravo", x=0.12, y=0.1, width=0.05),
    word("delta", x=0.19, y=0.1, width=0.05),
]


@pytest.mark.parametrize(
    ("rects", "expected_words", "expected_bucket"),
    [
        ([PassageRect(x=0.1, y=0.2, width=0.18, height=0.025)], 3, "a few words"),
        (
            [PassageRect(x=0.1, y=0.2, width=0.6, height=0.025)],
            10,
            "part of a sentence",
        ),
        (
            [
                PassageRect(x=0.1, y=0.2, width=0.6, height=0.025),
                PassageRect(x=0.1, y=0.24, width=0.6, height=0.025),
            ],
            20,
            "about a sentence",
        ),
        (
            [
                PassageRect(x=0.1, y=0.2, width=0.6, height=0.025),
                PassageRect(x=0.1, y=0.24, width=0.6, height=0.025),
                PassageRect(x=0.1, y=0.28, width=0.6, height=0.025),
                PassageRect(x=0.1, y=0.32, width=0.6, height=0.025),
            ],
            40,
            "a paragraph",
        ),
    ],
)
def test_uses_visible_geometry_for_the_papers_four_length_buckets(
    rects: list[PassageRect],
    expected_words: int,
    expected_bucket: str,
) -> None:
    estimate = estimate_length_cue(rects, NEARBY_WORDS)

    assert estimate.estimated_words == expected_words
    assert estimate.bucket == expected_bucket


def test_refuses_to_guess_without_enough_nearby_visible_text() -> None:
    with pytest.raises(LengthCueError) as error:
        estimate_length_cue(
            [PassageRect(x=0.1, y=0.2, width=0.3, height=0.025)],
            [word("one", x=0.05, y=0.1, width=0.03)],
        )

    assert error.value.code == "insufficient_visible_text_for_length"
