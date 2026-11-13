from __future__ import annotations

from redacted_backend.documents.layout import serialize_visible_document
from tests.documents.factories import discovery, mask, page, word


def test_serializes_columns_left_to_right_and_removes_repeated_headers() -> None:
    target = mask("target", x=0.1, y=0.4, width=0.2, height=0.04)
    first_page = page(
        word("REPEATED", x=0.3, y=0.03, block_id=0, line_id=0),
        word("HEADER", x=0.42, y=0.03, block_id=0, line_id=0),
        word("left", x=0.1, y=0.2, block_id=1, line_id=1),
        word("column", x=0.17, y=0.2, block_id=1, line_id=1),
        word("right", x=0.62, y=0.15, block_id=2, line_id=2),
        word("column", x=0.69, y=0.15, block_id=2, line_id=2),
        page_number=1,
    )
    second_page = page(
        word("REPEATED", x=0.3, y=0.03, block_id=0, line_id=0, page_number=2),
        word("HEADER", x=0.42, y=0.03, block_id=0, line_id=0, page_number=2),
        word("second", x=0.1, y=0.2, block_id=1, line_id=1, page_number=2),
        word("page", x=0.18, y=0.2, block_id=1, line_id=1, page_number=2),
        page_number=2,
    )

    transcript = serialize_visible_document(
        [first_page, second_page],
        [discovery(1, target), discovery(2)],
        "target",
    )

    assert "REPEATED" not in transcript.text
    assert transcript.text.index("left column") < transcript.text.index("right column")
    assert transcript.text.index("right column") < transcript.text.index("second page")


def test_rejoins_a_geometry_supported_line_break_hyphenation() -> None:
    target = mask("target", x=0.1, y=0.5, width=0.2, height=0.04)
    words = [
        word("inter-", x=0.1, y=0.2, block_id=1, line_id=1),
        word("national", x=0.1, y=0.235, block_id=1, line_id=2),
        word("relations", x=0.2, y=0.235, block_id=1, line_id=2),
    ]

    transcript = serialize_visible_document(
        [page(*words)],
        [discovery(1, target)],
        "target",
    )

    assert "international relations" in transcript.text
    assert "inter- national" not in transcript.text


def test_serializes_one_marker_for_a_passage_with_multiple_mask_rectangles() -> None:
    words = [
        word("before", x=0.1, y=0.2, block_id=1, line_id=1),
        word("COVERED", x=0.4, y=0.2, block_id=1, line_id=1),
        word("MORE", x=0.1, y=0.24, block_id=1, line_id=2),
        word("after", x=0.7, y=0.24, block_id=1, line_id=2),
    ]
    target_id = "passage-1"

    transcript = serialize_visible_document(
        [page(*words)],
        [
            discovery(
                1,
                mask(target_id, x=0.35, y=0.19, width=0.2, height=0.03),
                mask(target_id, x=0.08, y=0.23, width=0.2, height=0.03),
            )
        ],
        target_id,
    )

    assert transcript.text.count("[[TARGET]]") == 1
    assert "COVERED" not in transcript.text
    assert "MORE" not in transcript.text
