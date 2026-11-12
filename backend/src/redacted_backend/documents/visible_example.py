from __future__ import annotations

from uuid import UUID

from redacted_backend.documents.layout import (
    VisibleLayoutError,
    serialize_visible_document,
)
from redacted_backend.documents.length_cue import LengthCueError, estimate_length_cue
from redacted_backend.domain.passages import PassageRecord
from redacted_backend.domain.visible_examples import (
    MaskCandidate,
    MaskDiscovery,
    VisibleDocument,
    VisibleExample,
)


class VisibleExampleError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def build_visible_example(
    document: VisibleDocument,
    passages: list[PassageRecord],
    target_id: UUID,
) -> VisibleExample:
    if document.visible_date is None:
        raise VisibleExampleError(
            "missing_visible_date",
            "A date copied from or corroborated by visible text is required",
        )
    if any(not discovery.coverage_confident for discovery in document.mask_discoveries):
        raise VisibleExampleError(
            "mask_coverage_uncertain",
            "Every page must have confirmed mask coverage",
        )
    target = next((passage for passage in passages if passage.id == target_id), None)
    if target is None:
        raise VisibleExampleError("target_not_found", "Target passage does not exist")

    passage_ids = {str(passage.id) for passage in passages}
    by_page = {
        discovery.page_number: [
            candidate
            for candidate in discovery.masks
            if candidate.id not in passage_ids
        ]
        for discovery in document.mask_discoveries
    }
    for passage in passages:
        for rect in passage.rects:
            by_page.setdefault(passage.page_number, []).append(
                MaskCandidate(
                    id=str(passage.id),
                    page_number=passage.page_number,
                    rect=rect,
                    kind="manual",
                    confidence=1.0,
                    confirmed=True,
                )
            )
    discoveries = [
        MaskDiscovery(
            page_number=discovery.page_number,
            masks=by_page.get(discovery.page_number, []),
            coverage_confident=discovery.coverage_confident,
            warnings=discovery.warnings,
        )
        for discovery in document.mask_discoveries
    ]
    try:
        transcript = serialize_visible_document(
            document.pages,
            discoveries,
            str(target_id),
        )
        nearby_words = [
            word
            for word in transcript.visible_words
            if word.page_number == target.page_number
        ]
        length_cue = estimate_length_cue(target.rects, nearby_words)
    except (VisibleLayoutError, LengthCueError) as error:
        raise VisibleExampleError(error.code, str(error)) from error

    context_window = _target_paragraph(transcript.text)
    return VisibleExample(
        passage_id=target.id,
        document_id=document.id,
        document_date=document.visible_date,
        before_target=transcript.before_target,
        after_target=transcript.after_target,
        context_window=context_window,
        length_cue=length_cue,
        visible_words=transcript.visible_words,
        serialization_version=transcript.serialization_version,
    )


def _target_paragraph(text: str) -> str:
    paragraphs = text.split("\n\n")
    target_index = next(
        (
            index
            for index, paragraph in enumerate(paragraphs)
            if "[[TARGET]]" in paragraph
        ),
        None,
    )
    if target_index is None:
        raise VisibleExampleError("target_not_found", "Target paragraph is missing")
    start = max(0, target_index - 1)
    end = min(len(paragraphs), target_index + 2)
    return "\n\n".join(paragraphs[start:end])
