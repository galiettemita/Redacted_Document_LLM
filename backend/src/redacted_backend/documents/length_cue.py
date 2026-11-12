from __future__ import annotations

import statistics

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
    usable = [
        word
        for word in nearby_words
        if word.text.strip()
        and word.width > 0
        and min(rect.height for rect in target_rects) * 0.5
        <= word.height
        <= max(rect.height for rect in target_rects) * 2
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
    line_capacities = [
        max(1, round(rect.width / character_width)) for rect in target_rects
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
