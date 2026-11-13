from __future__ import annotations

from dataclasses import dataclass

from redacted_backend.domain.passages import PassageRect
from redacted_backend.domain.visible_examples import (
    MaskCandidate,
    MaskDiscovery,
    PageWord,
    PageWords,
    SerializedTranscript,
)


class VisibleLayoutError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class _Token:
    text: str
    x: float
    y: float
    width: float
    height: float
    block_id: int
    line_id: int
    word: PageWord | None = None

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass
class _Block:
    block_id: int
    lines: list[list[_Token]]

    @property
    def x(self) -> float:
        return min(token.x for line in self.lines for token in line)

    @property
    def y(self) -> float:
        return min(token.y for line in self.lines for token in line)

    @property
    def right(self) -> float:
        return max(token.right for line in self.lines for token in line)

    @property
    def bottom(self) -> float:
        return max(token.bottom for line in self.lines for token in line)


def serialize_visible_document(
    pages: list[PageWords],
    masks: list[MaskDiscovery],
    target_id: str,
) -> SerializedTranscript:
    discoveries = {discovery.page_number: discovery for discovery in masks}
    if any(not discovery.coverage_confident for discovery in discoveries.values()):
        raise VisibleLayoutError(
            "mask_coverage_uncertain",
            "Mask coverage must be confirmed before serialization",
        )
    repeated_lines = _repeated_running_lines(pages)
    page_texts: list[str] = []
    visible_words: list[PageWord] = []
    target_count = 0
    emitted_marker_ids: set[str] = set()
    for page in sorted(pages, key=lambda item: item.page_number):
        discovery = discoveries.get(page.page_number)
        if discovery is None:
            raise VisibleLayoutError(
                "missing_mask_discovery", "Every page needs mask review"
            )
        confirmed_masks = [
            candidate for candidate in discovery.masks if candidate.confirmed
        ]
        retained = [
            word
            for word in page.words
            if not _word_is_masked(word, confirmed_masks)
            and _line_key(page.page_number, word) not in repeated_lines
        ]
        visible_words.extend(retained)
        tokens = [
            _Token(
                text=word.text,
                x=word.x,
                y=word.y,
                width=word.width,
                height=word.height,
                block_id=word.block_id,
                line_id=word.line_id,
                word=word,
            )
            for word in retained
        ]
        next_marker_block = -1
        for candidate in confirmed_masks:
            if candidate.id in emitted_marker_ids:
                continue
            emitted_marker_ids.add(candidate.id)
            marker = "[[TARGET]]" if candidate.id == target_id else "[REDACTED]"
            if candidate.id == target_id:
                target_count += 1
            block_id, line_id = _nearest_line(candidate, retained)
            if block_id is None:
                block_id = next_marker_block
                line_id = 0
                next_marker_block -= 1
            tokens.append(
                _Token(
                    text=marker,
                    x=candidate.rect.x,
                    y=candidate.rect.y,
                    width=candidate.rect.width,
                    height=candidate.rect.height,
                    block_id=block_id,
                    line_id=line_id,
                )
            )
        blocks = _build_blocks(tokens)
        ordered_blocks = _xy_cut(blocks)
        page_texts.append("\n\n".join(_render_block(block) for block in ordered_blocks))

    if target_count == 0:
        raise VisibleLayoutError("target_not_found", "Target marker is missing")
    text = "\n\n".join(part for part in page_texts if part).strip()
    before, marker, after = text.partition("[[TARGET]]")
    if not marker:
        raise VisibleLayoutError("target_not_found", "Target marker is missing")
    return SerializedTranscript(
        text=text,
        before_target=before.strip(),
        after_target=after.strip(),
        visible_words=visible_words,
    )


def _word_is_masked(word: PageWord, masks: list[MaskCandidate]) -> bool:
    return any(_intersects(word.rect, candidate.rect) for candidate in masks)


def _intersects(left: PassageRect, right: PassageRect) -> bool:
    return (
        left.x < right.x + right.width
        and left.x + left.width > right.x
        and left.y < right.y + right.height
        and left.y + left.height > right.y
    )


def _nearest_line(
    candidate: MaskCandidate,
    words: list[PageWord],
) -> tuple[int | None, int | None]:
    overlapping = [
        word
        for word in words
        if word.y < candidate.rect.y + candidate.rect.height
        and word.y + word.height > candidate.rect.y
    ]
    if not overlapping:
        return None, None
    nearest = min(
        overlapping,
        key=lambda word: abs(
            (word.y + word.height / 2) - (candidate.rect.y + candidate.rect.height / 2)
        ),
    )
    return nearest.block_id, nearest.line_id


def _build_blocks(tokens: list[_Token]) -> list[_Block]:
    by_block_line: dict[tuple[int, int], list[_Token]] = {}
    for token in tokens:
        by_block_line.setdefault((token.block_id, token.line_id), []).append(token)
    blocks: dict[int, list[list[_Token]]] = {}
    for (block_id, _line_id), line_tokens in by_block_line.items():
        blocks.setdefault(block_id, []).append(
            sorted(line_tokens, key=lambda token: token.x)
        )
    return [
        _Block(
            block_id=block_id,
            lines=sorted(
                lines,
                key=lambda line: (
                    min(token.y for token in line),
                    min(token.x for token in line),
                ),
            ),
        )
        for block_id, lines in blocks.items()
    ]


def _xy_cut(blocks: list[_Block]) -> list[_Block]:
    if len(blocks) <= 1:
        return blocks
    vertical = _largest_gap(blocks, axis="x")
    if vertical is not None and vertical[0] >= 0.08:
        _, split = vertical
        left = [block for block in blocks if block.right <= split]
        right = [block for block in blocks if block.x >= split]
        if left and right and len(left) + len(right) == len(blocks):
            return _xy_cut(left) + _xy_cut(right)
    horizontal = _largest_gap(blocks, axis="y")
    if horizontal is not None and horizontal[0] >= 0.025:
        _, split = horizontal
        top = [block for block in blocks if block.bottom <= split]
        bottom = [block for block in blocks if block.y >= split]
        if top and bottom and len(top) + len(bottom) == len(blocks):
            return _xy_cut(top) + _xy_cut(bottom)
    return sorted(blocks, key=lambda block: (block.y, block.x, block.block_id))


def _largest_gap(
    blocks: list[_Block],
    *,
    axis: str,
) -> tuple[float, float] | None:
    intervals = sorted(
        ((block.x, block.right) if axis == "x" else (block.y, block.bottom))
        for block in blocks
    )
    best: tuple[float, float] | None = None
    running_end = intervals[0][1]
    for start, end in intervals[1:]:
        gap = start - running_end
        if gap > 0 and (best is None or gap > best[0]):
            best = (gap, running_end + gap / 2)
        running_end = max(running_end, end)
    return best


def _render_block(block: _Block) -> str:
    rendered_lines = [" ".join(token.text for token in line) for line in block.lines]
    output = rendered_lines[0]
    for current in rendered_lines[1:]:
        if output.endswith("-") and current and current[0].islower():
            output = output[:-1] + current
        else:
            output += " " + current
    return output


def _repeated_running_lines(pages: list[PageWords]) -> set[tuple[int, int, int]]:
    occurrences: dict[str, set[int]] = {}
    keys_by_text: dict[str, set[tuple[int, int, int]]] = {}
    for page in pages:
        lines: dict[tuple[int, int], list[PageWord]] = {}
        for word in page.words:
            if word.y <= 0.1 or word.y + word.height >= 0.9:
                lines.setdefault((word.block_id, word.line_id), []).append(word)
        for (block_id, line_id), words in lines.items():
            text = " ".join(
                word.text for word in sorted(words, key=lambda item: item.x)
            )
            normalized = " ".join(text.casefold().split())
            if not normalized or normalized.isdigit():
                continue
            occurrences.setdefault(normalized, set()).add(page.page_number)
            keys_by_text.setdefault(normalized, set()).add(
                (page.page_number, block_id, line_id)
            )
    repeated = {
        text for text, page_numbers in occurrences.items() if len(page_numbers) >= 2
    }
    return {key for text in repeated for key in keys_by_text[text]}


def _line_key(page_number: int, word: PageWord) -> tuple[int, int, int]:
    return page_number, word.block_id, word.line_id
