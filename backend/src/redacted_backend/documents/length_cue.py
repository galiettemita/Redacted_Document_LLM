from __future__ import annotations

import statistics
from itertools import pairwise

from redacted_backend.domain.passages import PassageRect
from redacted_backend.domain.visible_examples import LengthEstimate, PageWord


class LengthCueError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def estimate_length_cue(
    target_rects: list[PassageRect],
    nearby_words: list[PageWord],
) -> LengthEstimate:
    candidates = [
        word
        for word in nearby_words
        if word.text.strip() and word.width > 0 and word.height > 0
    ]
    if len(candidates) < 3:
        raise LengthCueError(
            "insufficient_visible_text_for_length",
            "Not enough nearby visible text to estimate passage length",
        )
    typical_word_height = statistics.median(word.height for word in candidates)
    usable = [
        word
        for word in candidates
        if typical_word_height * 0.5 <= word.height <= typical_word_height * 2
    ]
    if len(usable) < 3:
        raise LengthCueError(
            "insufficient_visible_text_for_length",
            "Not enough nearby visible text to estimate passage length",
        )
    character_width = statistics.median(
        word.width / max(1, len(word.text)) for word in usable
    )
    characters_per_word = statistics.median(len(word.text) for word in usable)
    line_pitch = _typical_line_pitch(usable, typical_word_height)
    line_capacities = [
        max(1, round(rect.width / character_width))
        for rect in target_rects
        for _ in range(max(1, round(rect.height / line_pitch)))
    ]
    estimated_words = max(
        1,
        round(sum(line_capacities) / (characters_per_word + 1)),
    )
    if estimated_words <= 4:
        bucket = "a few words"
    elif estimated_words <= 15:
        bucket = "part of a sentence"
    elif estimated_words <= 30:
        bucket = "about a sentence"
    else:
        bucket = "a paragraph"
    return LengthEstimate(
        estimated_words=estimated_words,
        bucket=bucket,
        median_character_width=character_width,
        median_characters_per_word=characters_per_word,
        line_capacities=line_capacities,
    )


def _typical_line_pitch(words: list[PageWord], word_height: float) -> float:
    line_centers: list[list[float]] = []
    for center in sorted(word.y + word.height / 2 for word in words):
        if (
            line_centers
            and center - statistics.median(line_centers[-1]) <= word_height * 0.6
        ):
            line_centers[-1].append(center)
        else:
            line_centers.append([center])
    centers = [statistics.median(group) for group in line_centers]
    gaps = [
        second - first
        for first, second in pairwise(centers)
        if word_height * 0.8 <= second - first <= word_height * 3
    ]
    return statistics.median(gaps) if gaps else word_height * 1.5
