from __future__ import annotations

import numpy as np

from redacted_backend.documents.layout import serialize_visible_document
from redacted_backend.documents.masks import discover_candidate_masks
from tests.documents.factories import discovery, mask, page, word


def test_covered_embedded_text_never_enters_transcript_or_provenance() -> None:
    words = [
        word("before", x=0.08, y=0.2),
        word("SECRET", x=0.35, y=0.2, width=0.12),
        word("after", x=0.7, y=0.2),
    ]
    target = mask(
        "target",
        x=0.3,
        y=0.18,
        width=0.25,
        height=0.08,
    )

    transcript = serialize_visible_document(
        [page(*words)],
        [discovery(1, target)],
        "target",
    )

    assert "SECRET" not in transcript.text
    assert "SECRET" not in [visible.text for visible in transcript.visible_words]
    assert transcript.text == "before [[TARGET]] after"


def test_discovers_a_high_contrast_black_rectangle() -> None:
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    image[180:260, 300:600] = 0

    result = discover_candidate_masks(
        image,
        [
            word("before", x=0.1, y=0.2),
            word("after", x=0.7, y=0.2),
        ],
        page_number=1,
    )

    black_masks = [candidate for candidate in result.masks if candidate.kind == "black"]
    assert len(black_masks) == 1
    assert black_masks[0].rect.x == 0.3
    assert black_masks[0].rect.y == 0.18
    assert black_masks[0].rect.width == 0.3
    assert black_masks[0].rect.height == 0.08


def test_discovers_a_white_covered_span_between_words_on_one_line() -> None:
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    words = [
        word("before", x=0.1, y=0.2, width=0.08, line_id=1),
        word("after", x=0.65, y=0.2, width=0.06, line_id=1),
        word("nearby", x=0.1, y=0.3, width=0.07, line_id=2),
        word("words", x=0.2, y=0.3, width=0.05, line_id=2),
    ]

    result = discover_candidate_masks(image, words, page_number=1)

    white_masks = [candidate for candidate in result.masks if candidate.kind == "white"]
    assert len(white_masks) == 1
    assert white_masks[0].rect.x >= 0.18
    assert white_masks[0].rect.x + white_masks[0].rect.width <= 0.65


def test_white_pixels_over_an_embedded_word_are_treated_as_a_mask() -> None:
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    image[205:215, 105:175] = 0
    image[205:215, 655:705] = 0
    words = [
        word("before", x=0.1, y=0.2, width=0.08, line_id=1),
        word("SECRET", x=0.3, y=0.2, width=0.1, line_id=1),
        word("after", x=0.65, y=0.2, width=0.06, line_id=1),
    ]

    result = discover_candidate_masks(image, words, page_number=1)

    assert any(
        candidate.kind == "white"
        and candidate.rect.x <= 0.3
        and candidate.rect.x + candidate.rect.width >= 0.4
        for candidate in result.masks
    )


def test_sparse_blank_page_reports_uncertain_mask_coverage() -> None:
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)

    result = discover_candidate_masks(image, [], page_number=1)

    assert result.coverage_confident is False
    assert result.warnings == ["insufficient_visible_content"]
