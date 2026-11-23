from __future__ import annotations

from datetime import date

import pytest

from redacted_backend.documents.dates import (
    VisibleDateDetectionError,
    detect_visible_document_date,
)
from redacted_backend.domain.visible_examples import PageWord
from tests.documents.factories import discovery, mask, page, word


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["DATE:", "1945-05-08"], date(1945, 5, 8)),
        (["FEBRUARY", "19,", "1953"], date(1953, 2, 19)),
        (["19", "February", "1953"], date(1953, 2, 19)),
        (["19", "Feb.", "1953"], date(1953, 2, 19)),
    ],
)
def test_detects_supported_visible_header_dates(
    tokens: list[str],
    expected: date,
) -> None:
    words = [
        word(token, x=0.08 + index * 0.12, y=0.08, block_id=1, line_id=1)
        for index, token in enumerate(tokens)
    ]

    evidence = detect_visible_document_date(
        [page(*words)],
        [discovery(1)],
    )

    assert evidence.value == expected
    assert evidence.page_number == 1
    assert evidence.line_text == " ".join(tokens)
    assert [item.text for item in evidence.words] == tokens
    assert evidence.parser_version == "visible-date-v1"


def test_accepts_date_words_extracted_by_ocr() -> None:
    words = [
        PageWord.model_validate(
            word("March", x=0.08, y=0.09, block_id=1, line_id=1).model_dump()
            | {"source": "ocr", "confidence": 0.91}
        ),
        PageWord.model_validate(
            word("4,", x=0.18, y=0.09, block_id=1, line_id=1).model_dump()
            | {"source": "ocr", "confidence": 0.88}
        ),
        PageWord.model_validate(
            word("1961", x=0.24, y=0.09, block_id=1, line_id=1).model_dump()
            | {"source": "ocr", "confidence": 0.93}
        ),
    ]

    evidence = detect_visible_document_date([page(*words)], [discovery(1)])

    assert evidence.value == date(1961, 3, 4)
    assert {item.source.value for item in evidence.words} == {"ocr"}


def test_accepts_a_slightly_lower_confidence_numeric_day_in_a_clear_ocr_date() -> None:
    tokens = [
        ("FEB", 0.96),
        ("07", 0.73),
        ("2020", 0.96),
    ]
    words = [
        PageWord.model_validate(
            word(
                token,
                x=0.08 + index * 0.12,
                y=0.09,
                block_id=1,
                line_id=1,
            ).model_dump()
            | {"source": "ocr", "confidence": confidence}
        )
        for index, (token, confidence) in enumerate(tokens)
    ]

    evidence = detect_visible_document_date([page(*words)], [discovery(1)])

    assert evidence.value == date(2020, 2, 7)
    assert evidence.precision == "day"


@pytest.mark.parametrize(
    ("tokens", "confidences"),
    [
        (["FEB", "07", "2020"], [0.73, 0.96, 0.96]),
        (["FEB", "07", "2020"], [0.96, 0.69, 0.96]),
    ],
)
def test_keeps_low_confidence_months_and_unreliable_days_fail_closed(
    tokens: list[str],
    confidences: list[float],
) -> None:
    words = [
        PageWord.model_validate(
            word(
                token,
                x=0.08 + index * 0.12,
                y=0.09,
                block_id=1,
                line_id=1,
            ).model_dump()
            | {"source": "ocr", "confidence": confidences[index]}
        )
        for index, token in enumerate(tokens)
    ]

    with pytest.raises(VisibleDateDetectionError) as error:
        detect_visible_document_date([page(*words)], [discovery(1)])

    assert error.value.code == "visible_date_not_found"


@pytest.mark.parametrize(
    "tokens",
    [
        ["October", "2016"],
        ["2016", "OCT"],
    ],
)
def test_uses_visible_month_as_a_conservative_date_boundary(
    tokens: list[str],
) -> None:
    words = [
        word(token, x=0.08 + index * 0.12, y=0.08, block_id=1, line_id=1)
        for index, token in enumerate(tokens)
    ]

    evidence = detect_visible_document_date([page(*words)], [discovery(1)])

    assert evidence.value == date(2016, 10, 1)
    assert evidence.precision == "month"
    assert evidence.line_text == " ".join(tokens)
    assert evidence.parser_version == "visible-date-v2"


def test_prefers_a_complete_visible_date_over_an_agreeing_visible_month() -> None:
    words = [
        word("October", x=0.08, y=0.08, block_id=1, line_id=1),
        word("2016", x=0.20, y=0.08, block_id=1, line_id=1),
        word("October", x=0.08, y=0.14, block_id=2, line_id=1),
        word("21,", x=0.20, y=0.14, block_id=2, line_id=1),
        word("2016", x=0.28, y=0.14, block_id=2, line_id=1),
    ]

    evidence = detect_visible_document_date([page(*words)], [discovery(1)])

    assert evidence.value == date(2016, 10, 21)
    assert evidence.precision == "day"
    assert evidence.line_text == "October 21, 2016"


def test_rejects_conflicting_visible_months() -> None:
    words = [
        word("October", x=0.08, y=0.08, block_id=1, line_id=1),
        word("2016", x=0.20, y=0.08, block_id=1, line_id=1),
        word("November", x=0.08, y=0.14, block_id=2, line_id=1),
        word("2016", x=0.20, y=0.14, block_id=2, line_id=1),
    ]

    with pytest.raises(VisibleDateDetectionError) as error:
        detect_visible_document_date([page(*words)], [discovery(1)])

    assert error.value.code == "visible_date_ambiguous"


def test_rejects_low_confidence_ocr_date_words() -> None:
    words = [
        PageWord.model_validate(
            word("1945-05-08", x=0.08, y=0.08, block_id=1, line_id=1).model_dump()
            | {"source": "ocr", "confidence": 0.49}
        )
    ]

    with pytest.raises(VisibleDateDetectionError) as error:
        detect_visible_document_date([page(*words)], [discovery(1)])

    assert error.value.code == "visible_date_not_found"


def test_accepts_repeated_occurrences_of_the_same_visible_date() -> None:
    words = [
        word("DATE:", x=0.08, y=0.08, block_id=1, line_id=1),
        word("1945-05-08", x=0.17, y=0.08, block_id=1, line_id=1),
        word("1945-05-08", x=0.08, y=0.16, block_id=2, line_id=1),
    ]

    evidence = detect_visible_document_date([page(*words)], [discovery(1)])

    assert evidence.value == date(1945, 5, 8)
    assert evidence.line_text == "DATE: 1945-05-08"


def test_rejects_multiple_distinct_visible_header_dates() -> None:
    words = [
        word("1945-05-08", x=0.08, y=0.08, block_id=1, line_id=1),
        word("1945-05-09", x=0.08, y=0.16, block_id=2, line_id=1),
    ]

    with pytest.raises(VisibleDateDetectionError) as error:
        detect_visible_document_date([page(*words)], [discovery(1)])

    assert error.value.code == "visible_date_ambiguous"


@pytest.mark.parametrize(
    "tokens",
    [
        [],
        ["DATE:", "05/08/1945"],
    ],
)
def test_rejects_missing_or_locale_ambiguous_dates(tokens: list[str]) -> None:
    words = [
        word(token, x=0.08 + index * 0.12, y=0.08, block_id=1, line_id=1)
        for index, token in enumerate(tokens)
    ]

    with pytest.raises(VisibleDateDetectionError) as error:
        detect_visible_document_date([page(*words)], [discovery(1)])

    assert error.value.code == "visible_date_not_found"


def test_rejects_a_date_that_appears_only_in_document_body_text() -> None:
    words = [word("1945-05-08", x=0.08, y=0.24, block_id=3, line_id=1)]

    with pytest.raises(VisibleDateDetectionError) as error:
        detect_visible_document_date([page(*words)], [discovery(1)])

    assert error.value.code == "visible_date_not_found"


def test_rejects_a_date_embedded_in_narrative_header_band_text() -> None:
    tokens = ["The", "meeting", "was", "February", "19,", "1953"]
    words = [
        word(token, x=0.04 + index * 0.11, y=0.08, block_id=1, line_id=1)
        for index, token in enumerate(tokens)
    ]

    with pytest.raises(VisibleDateDetectionError) as error:
        detect_visible_document_date([page(*words)], [discovery(1)])

    assert error.value.code == "visible_date_not_found"


def test_rejects_a_date_covered_by_a_confirmed_mask() -> None:
    words = [
        word(
            "1945-05-08",
            x=0.1,
            y=0.08,
            width=0.14,
            height=0.03,
            block_id=1,
            line_id=1,
        )
    ]
    covered = mask(
        "auto-black-1-1",
        x=0.09,
        y=0.07,
        width=0.16,
        height=0.05,
    )

    with pytest.raises(VisibleDateDetectionError) as error:
        detect_visible_document_date(
            [page(*words)],
            [discovery(1, covered)],
        )

    assert error.value.code == "visible_date_not_found"
