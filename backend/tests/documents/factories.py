from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import UUID

from redacted_backend.domain.passages import PassageRecord, PassageRect
from redacted_backend.domain.visible_examples import (
    MaskCandidate,
    MaskDiscovery,
    PageWord,
    PageWords,
    VisibleDocument,
)


def word(
    text: str,
    *,
    x: float,
    y: float,
    width: float | None = None,
    height: float = 0.025,
    page_number: int = 1,
    block_id: int = 0,
    line_id: int = 0,
) -> PageWord:
    return PageWord(
        text=text,
        page_number=page_number,
        x=x,
        y=y,
        width=width if width is not None else max(0.01, len(text) * 0.01),
        height=height,
        block_id=block_id,
        line_id=line_id,
        source="embedded",
        confidence=1.0,
    )


def page(*words: PageWord, page_number: int = 1) -> PageWords:
    return PageWords(
        page_number=page_number,
        width_points=612,
        height_points=792,
        extraction_source="embedded",
        words=list(words),
    )


def mask(
    mask_id: str,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    page_number: int = 1,
    kind: str = "black",
) -> MaskCandidate:
    return MaskCandidate(
        id=mask_id,
        page_number=page_number,
        rect=PassageRect(x=x, y=y, width=width, height=height),
        kind=kind,
        confidence=1.0,
        confirmed=True,
    )


def discovery(
    page_number: int,
    *masks: MaskCandidate,
    coverage_confident: bool = True,
) -> MaskDiscovery:
    return MaskDiscovery(
        page_number=page_number,
        masks=list(masks),
        coverage_confident=coverage_confident,
        warnings=[],
    )


def passage(
    passage_id: str,
    *,
    rects: list[PassageRect],
    page_number: int = 1,
) -> PassageRecord:
    now = datetime.now(UTC)
    return PassageRecord(
        id=UUID(passage_id),
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        label="Passage 1",
        page_number=page_number,
        rects=rects,
        source="manual",
        status="marked",
        revision=1,
        created_at=now,
        updated_at=now,
    )


def visible_document(
    *,
    pages: list[PageWords],
    discoveries: list[MaskDiscovery],
    visible_date: date | None = date(1945, 5, 8),
) -> VisibleDocument:
    return VisibleDocument(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        visible_date=visible_date,
        pages=pages,
        mask_discoveries=discoveries,
    )
