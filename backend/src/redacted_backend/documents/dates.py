from __future__ import annotations

import re
from datetime import date
from typing import Literal

from redacted_backend.domain.documents import VisibleDateEvidence
from redacted_backend.domain.passages import PassageRect
from redacted_backend.domain.visible_examples import (
    MaskDiscovery,
    PageWord,
    PageWords,
    WordSource,
)

_HEADER_BOTTOM = 0.20
_MIN_OCR_CONFIDENCE = 0.75
_MIN_NUMERIC_DAY_OCR_CONFIDENCE = 0.70
_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_PREFIX = r"(?:date\s*:\s*)?"
_ISO_DATE = re.compile(
    rf"{_DATE_PREFIX}(?P<year>\d{{4}})-(?P<month>\d{{2}})-(?P<day>\d{{2}})\.?",
    re.IGNORECASE,
)
_MONTH_FIRST = re.compile(
    rf"{_DATE_PREFIX}(?P<month>{_MONTH_PATTERN})\.?\s+"
    rf"(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+"
    rf"(?P<year>\d{{4}})\.?",
    re.IGNORECASE,
)
_DAY_FIRST = re.compile(
    rf"{_DATE_PREFIX}(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
    rf"(?P<month>{_MONTH_PATTERN})\.?\s*,?\s+"
    rf"(?P<year>\d{{4}})\.?",
    re.IGNORECASE,
)
_MONTH_ONLY = re.compile(
    rf"{_DATE_PREFIX}(?P<month>{_MONTH_PATTERN})\.?\s+(?P<year>\d{{4}})\.?",
    re.IGNORECASE,
)
_YEAR_MONTH_ONLY = re.compile(
    rf"(?P<year>\d{{4}})\s+(?P<month>{_MONTH_PATTERN})\.?",
    re.IGNORECASE,
)


class VisibleDateDetectionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def detect_visible_document_date(
    pages: list[PageWords],
    discoveries: list[MaskDiscovery],
) -> VisibleDateEvidence:
    """Find one unambiguous date in the visible header of the first PDF page."""

    first_page = next((page for page in pages if page.page_number == 1), None)
    if first_page is None:
        raise VisibleDateDetectionError(
            "visible_date_not_found",
            "No first page was available for visible date detection",
        )
    discovery = next(
        (item for item in discoveries if item.page_number == first_page.page_number),
        None,
    )
    masks = (
        []
        if discovery is None
        else [mask for mask in discovery.masks if mask.confirmed]
    )
    header_words = [
        word
        for word in first_page.words
        if word.y < _HEADER_BOTTOM
        and not any(_intersects(word.rect, mask.rect) for mask in masks)
    ]
    lines: dict[tuple[int, int], list[PageWord]] = {}
    for word in header_words:
        lines.setdefault((word.block_id, word.line_id), []).append(word)

    matches: list[VisibleDateEvidence] = []
    for words in lines.values():
        ordered = sorted(words, key=lambda item: (item.x, item.y))
        line_text = " ".join(word.text for word in ordered)
        parsed_dates = _parse_dates(line_text)
        if not _ocr_line_is_reliable(ordered, parsed_dates):
            continue
        for parsed_date, precision in parsed_dates:
            matches.append(
                VisibleDateEvidence(
                    value=parsed_date,
                    page_number=first_page.page_number,
                    line_text=line_text,
                    words=ordered,
                    precision=precision,
                    parser_version=(
                        "visible-date-v2"
                        if precision == "month"
                        else "visible-date-v1"
                    ),
                )
            )

    if not matches:
        raise VisibleDateDetectionError(
            "visible_date_not_found",
            "No supported date was found in visible first-page header text",
        )
    distinct_months = {(match.value.year, match.value.month) for match in matches}
    distinct_days = {
        match.value for match in matches if match.precision == "day"
    }
    if len(distinct_months) > 1 or len(distinct_days) > 1:
        raise VisibleDateDetectionError(
            "visible_date_ambiguous",
            "Multiple distinct dates were found in visible first-page header text",
        )
    return next(
        (match for match in matches if match.precision == "day"),
        matches[0],
    )


def _parse_dates(text: str) -> list[tuple[date, Literal["day", "month"]]]:
    parsed: list[tuple[date, Literal["day", "month"]]] = []
    iso_match = _ISO_DATE.fullmatch(text.strip())
    if iso_match is not None:
        _append_valid_day(
            parsed,
            int(iso_match.group("year")),
            int(iso_match.group("month")),
            int(iso_match.group("day")),
        )
    for pattern in (_MONTH_FIRST, _DAY_FIRST):
        match = pattern.fullmatch(text.strip())
        if match is not None:
            month_name = match.group("month").lower()
            _append_valid_day(
                parsed,
                int(match.group("year")),
                _MONTHS[month_name],
                int(match.group("day")),
            )
    for pattern in (_MONTH_ONLY, _YEAR_MONTH_ONLY):
        match = pattern.fullmatch(text.strip())
        if match is not None:
            candidate = date(
                int(match.group("year")),
                _MONTHS[match.group("month").lower()],
                1,
            )
            item: tuple[date, Literal["day", "month"]] = (candidate, "month")
            if item not in parsed:
                parsed.append(item)
    return parsed


def _ocr_line_is_reliable(
    words: list[PageWord],
    parsed_dates: list[tuple[date, Literal["day", "month"]]],
) -> bool:
    low_confidence = [
        word
        for word in words
        if word.source == WordSource.OCR and word.confidence < _MIN_OCR_CONFIDENCE
    ]
    if not low_confidence:
        return True
    if len(low_confidence) != 1 or len(parsed_dates) != 1:
        return False
    word = low_confidence[0]
    parsed_date, precision = parsed_dates[0]
    if precision != "day" or word.confidence < _MIN_NUMERIC_DAY_OCR_CONFIDENCE:
        return False
    numeric_day = re.fullmatch(
        r"(?P<day>\d{1,2})(?:st|nd|rd|th)?[,.]?",
        word.text.strip(),
        re.IGNORECASE,
    )
    return numeric_day is not None and int(numeric_day.group("day")) == parsed_date.day


def _append_valid_day(
    parsed: list[tuple[date, Literal["day", "month"]]],
    year: int,
    month: int,
    day: int,
) -> None:
    try:
        candidate = date(year, month, day)
    except ValueError:
        return
    item: tuple[date, Literal["day", "month"]] = (candidate, "day")
    if item not in parsed:
        parsed.append(item)


def _intersects(first: PassageRect, second: PassageRect) -> bool:
    return not (
        first.x + first.width <= second.x
        or second.x + second.width <= first.x
        or first.y + first.height <= second.y
        or second.y + second.height <= first.y
    )
