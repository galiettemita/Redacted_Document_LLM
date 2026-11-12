from __future__ import annotations

import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from redacted_backend.documents.visible_example import (
    VisibleExampleError,
    build_visible_example,
)
from redacted_backend.domain.passages import PassageRect
from redacted_backend.domain.visible_examples import VisibleExample
from tests.documents.factories import (
    discovery,
    mask,
    page,
    passage,
    visible_document,
    word,
)

TARGET_ID = "00000000-0000-0000-0000-000000000010"
OTHER_ID = "00000000-0000-0000-0000-000000000020"


def test_visible_example_schema_contains_no_hidden_answer_names() -> None:
    schema = json.dumps(VisibleExample.model_json_schema()).lower()

    for forbidden in ("answer", "reveal", "ground_truth", "later_release"):
        assert forbidden not in schema


def test_visible_example_forbids_unknown_hidden_fields() -> None:
    with pytest.raises(ValidationError):
        VisibleExample.model_validate(
            {
                "passage_id": TARGET_ID,
                "document_id": "00000000-0000-0000-0000-000000000001",
                "document_date": "1945-05-08",
                "before_target": "before",
                "after_target": "after",
                "context_window": "before [[TARGET]] after",
                "length_cue": {
                    "estimated_words": 3,
                    "bucket": "a few words",
                    "median_character_width": 0.01,
                    "median_characters_per_word": 5,
                    "line_capacities": [18],
                },
                "visible_words": [],
                "serialization_version": "visible-transcript-v1",
                "answer": "must never be accepted",
            }
        )


def test_refuses_to_build_without_a_visible_document_date() -> None:
    target_rect = PassageRect(x=0.3, y=0.2, width=0.18, height=0.025)
    target = passage(TARGET_ID, rects=[target_rect])
    document = visible_document(
        pages=[page(word("visible", x=0.1, y=0.2))],
        discoveries=[
            discovery(1, mask(TARGET_ID, x=0.3, y=0.2, width=0.18, height=0.025))
        ],
        visible_date=None,
    )

    with pytest.raises(VisibleExampleError) as error:
        build_visible_example(document, [target], UUID(TARGET_ID))

    assert error.value.code == "missing_visible_date"


def test_refuses_to_build_when_any_page_has_uncertain_mask_coverage() -> None:
    target_rect = PassageRect(x=0.3, y=0.2, width=0.18, height=0.025)
    target = passage(TARGET_ID, rects=[target_rect])
    document = visible_document(
        pages=[page(word("visible", x=0.1, y=0.2))],
        discoveries=[discovery(1, coverage_confident=False)],
    )

    with pytest.raises(VisibleExampleError) as error:
        build_visible_example(document, [target], UUID(TARGET_ID))

    assert error.value.code == "mask_coverage_uncertain"


def test_builds_target_and_other_markers_from_visible_inputs_only() -> None:
    target_rect = PassageRect(x=0.35, y=0.2, width=0.18, height=0.025)
    other_rect = PassageRect(x=0.3, y=0.4, width=0.2, height=0.025)
    words = [
        word("alpha", x=0.05, y=0.1, width=0.05, block_id=1, line_id=1),
        word("bravo", x=0.12, y=0.1, width=0.05, block_id=1, line_id=1),
        word("delta", x=0.19, y=0.1, width=0.05, block_id=1, line_id=1),
        word("before", x=0.1, y=0.2, block_id=1, line_id=2),
        word("HIDDEN", x=0.38, y=0.2, width=0.08, block_id=1, line_id=2),
        word("after", x=0.65, y=0.2, block_id=1, line_id=2),
        word("later", x=0.1, y=0.4, block_id=2, line_id=3),
    ]
    document = visible_document(
        pages=[page(*words)],
        discoveries=[
            discovery(
                1,
                mask(TARGET_ID, x=0.35, y=0.19, width=0.18, height=0.05),
                mask(OTHER_ID, x=0.3, y=0.39, width=0.2, height=0.05),
            )
        ],
    )

    example = build_visible_example(
        document,
        [
            passage(TARGET_ID, rects=[target_rect]),
            passage(OTHER_ID, rects=[other_rect]),
        ],
        UUID(TARGET_ID),
    )

    assert example.passage_id == UUID(TARGET_ID)
    assert "HIDDEN" not in example.before_target
    assert "HIDDEN" not in example.after_target
    assert "[[TARGET]]" not in example.before_target
    assert "[[TARGET]]" not in example.after_target
    assert "[REDACTED]" in example.after_target
    assert "[[TARGET]]" in example.context_window
    assert example.length_cue.bucket == "a few words"


def test_deduplicates_an_automatic_mask_overlapping_the_manual_target() -> None:
    target_rect = PassageRect(x=0.35, y=0.2, width=0.18, height=0.025)
    target = passage(TARGET_ID, rects=[target_rect])
    document = visible_document(
        pages=[
            page(
                word("alpha", x=0.05, y=0.2, block_id=1, line_id=1),
                word("bravo", x=0.12, y=0.2, block_id=1, line_id=1),
                word("delta", x=0.19, y=0.2, block_id=1, line_id=1),
            )
        ],
        discoveries=[
            discovery(
                1,
                mask(
                    "auto-black-1-1",
                    x=0.345,
                    y=0.195,
                    width=0.19,
                    height=0.035,
                ),
            )
        ],
    )

    example = build_visible_example(document, [target], UUID(TARGET_ID))

    assert example.context_window.count("[[TARGET]]") == 1
    assert "[REDACTED]" not in example.context_window
