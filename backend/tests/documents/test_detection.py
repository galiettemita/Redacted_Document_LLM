from __future__ import annotations

from pathlib import Path

import numpy as np

from redacted_backend.documents.detection import (
    detect_passages,
    group_detected_passages,
)
from redacted_backend.domain.passages import PassageRect
from redacted_backend.domain.visible_examples import (
    MaskCandidate,
    MaskDiscovery,
    PageWord,
    PageWords,
)


def test_detection_returns_black_and_white_masks_in_page_reading_order() -> None:
    pages = [_page(1)]
    discoveries = [
        _discovery(
            1,
            _mask("later", 1, x=0.1, y=0.5, kind="white"),
            _mask("earlier", 1, x=0.1, y=0.2, kind="black"),
        )
    ]

    result = group_detected_passages(pages, discoveries)

    assert result.coverage_confident is True
    assert [passage.rects[0].y for passage in result.passages] == [0.2, 0.5]
    assert [passage.kinds for passage in result.passages] == [
        ["black"],
        ["white"],
    ]


def test_touching_and_line_wrapped_masks_group_without_readable_words_between() -> None:
    pages = [_page(1)]
    discoveries = [
        _discovery(
            1,
            _mask("touch-a", 1, x=0.1, y=0.2, width=0.18),
            _mask("touch-b", 1, x=0.28, y=0.2, width=0.16),
            _mask("wrapped", 1, x=0.1, y=0.235, width=0.30),
        )
    ]

    result = group_detected_passages(pages, discoveries)

    assert len(result.passages) == 1
    assert [rect.y for rect in result.passages[0].rects] == [0.2, 0.2, 0.235]


def test_readable_word_between_masks_keeps_passages_separate() -> None:
    pages = [_page(1, _word("visible", 1, x=0.29, y=0.2, width=0.08))]
    discoveries = [
        _discovery(
            1,
            _mask("left", 1, x=0.1, y=0.2, width=0.15),
            _mask("right", 1, x=0.42, y=0.2, width=0.16),
        )
    ]

    result = group_detected_passages(pages, discoveries)

    assert len(result.passages) == 2


def test_line_wrapped_masks_in_different_columns_never_group() -> None:
    pages = [_page(1)]
    discoveries = [
        _discovery(
            1,
            _mask("left", 1, x=0.08, y=0.2, width=0.25),
            _mask("right", 1, x=0.62, y=0.235, width=0.25),
        )
    ]

    result = group_detected_passages(pages, discoveries)

    assert len(result.passages) == 2


def test_detection_orders_passages_by_page_then_column_and_position() -> None:
    pages = [_page(1), _page(2)]
    discoveries = [
        _discovery(2, _mask("page-two", 2, x=0.1, y=0.1)),
        _discovery(
            1,
            _mask("right-column", 1, x=0.62, y=0.1),
            _mask("left-later", 1, x=0.1, y=0.4),
            _mask("left-earlier", 1, x=0.1, y=0.1),
        ),
    ]

    result = group_detected_passages(pages, discoveries)

    assert [passage.source_mask_ids for passage in result.passages] == [
        ["left-earlier"],
        ["left-later"],
        ["right-column"],
        ["page-two"],
    ]


def test_detection_reports_uncertain_page_coverage_without_hiding_candidates() -> None:
    pages = [_page(1)]
    discoveries = [
        MaskDiscovery(
            page_number=1,
            masks=[_mask("candidate", 1, x=0.1, y=0.2)],
            coverage_confident=False,
            warnings=["insufficient_visible_content"],
        )
    ]

    result = group_detected_passages(pages, discoveries)

    assert result.coverage_confident is False
    assert result.uncertain_pages == [1]
    assert result.warnings == ["page 1: insufficient_visible_content"]
    assert len(result.passages) == 1


def test_detect_passages_extracts_renders_and_discovers_every_page_once() -> None:
    pages = [_page(1), _page(2)]
    images = [np.full((20, 20), 255, dtype=np.uint8) for _ in pages]
    discovery_calls: list[int] = []

    def discover(_image, _words, *, page_number: int) -> MaskDiscovery:
        discovery_calls.append(page_number)
        return _discovery(
            page_number,
            _mask(f"page-{page_number}", page_number, x=0.1, y=0.2),
        )

    result = detect_passages(
        Path("unused.pdf"),
        _extractor=lambda _path: pages,
        _renderer=lambda _path: images,
        _discoverer=discover,
    )

    assert result.pages_scanned == 2
    assert [item.page_number for item in result.passages] == [1, 2]
    assert discovery_calls == [1, 2]


def test_detect_passages_discovers_each_page_before_rendering_the_next() -> None:
    pages = [_page(1), _page(2)]
    events: list[str] = []

    def render(_path: Path):
        for page in pages:
            events.append(f"render:{page.page_number}")
            yield np.full((20, 20), 255, dtype=np.uint8)

    def discover(_image, _words, *, page_number: int) -> MaskDiscovery:
        events.append(f"discover:{page_number}")
        return _discovery(page_number)

    detect_passages(
        Path("unused.pdf"),
        _extractor=lambda _path: pages,
        _renderer=render,
        _discoverer=discover,
    )

    assert events == ["render:1", "discover:1", "render:2", "discover:2"]


def test_real_embedded_fixture_does_not_promote_plain_whitespace_to_a_passage() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "pdfs" / "visible-layout.pdf"

    result = detect_passages(fixture)

    assert result.pages_scanned == 2
    assert result.coverage_confident is True
    assert [
        (passage.page_number, [kind.value for kind in passage.kinds])
        for passage in result.passages
    ] == [
        (1, ["black"]),
        (2, ["black"]),
    ]


def _page(page_number: int, *words: PageWord) -> PageWords:
    return PageWords(
        page_number=page_number,
        width_points=612,
        height_points=792,
        extraction_source="embedded",
        words=list(words),
    )


def _word(
    text: str,
    page_number: int,
    *,
    x: float,
    y: float,
    width: float,
) -> PageWord:
    return PageWord(
        text=text,
        page_number=page_number,
        x=x,
        y=y,
        width=width,
        height=0.025,
        block_id=1,
        line_id=1,
        source="embedded",
        confidence=1.0,
    )


def _mask(
    mask_id: str,
    page_number: int,
    *,
    x: float,
    y: float,
    width: float = 0.2,
    height: float = 0.025,
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


def _discovery(
    page_number: int,
    *masks: MaskCandidate,
) -> MaskDiscovery:
    return MaskDiscovery(
        page_number=page_number,
        masks=list(masks),
        coverage_confident=True,
        warnings=[],
    )
