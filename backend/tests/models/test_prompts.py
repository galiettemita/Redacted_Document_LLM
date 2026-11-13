from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from uuid import UUID

from redacted_backend.domain.retrieval import (
    FourRankedLists,
    FusedChunk,
    FusedDocument,
    RetrievalTrace,
)
from redacted_backend.domain.visible_examples import LengthEstimate, VisibleExample
from redacted_backend.models.prompts import (
    render_prediction_prompt,
    render_query_prompt,
)

PROMPT_ROOT = Path(__file__).parents[2] / "prompts"


def test_appendix_f_prompt_files_have_frozen_hashes() -> None:
    query = (PROMPT_ROOT / "realistic_query_v1.txt").read_bytes()
    prediction = (PROMPT_ROOT / "realistic_prediction_v1.txt").read_bytes()

    assert hashlib.sha256(query).hexdigest() == (
        "1e60c0d9d7918f32c08c0af6856393bb117242cd06091e35075cb24fcb02e8f4"
    )
    assert hashlib.sha256(prediction).hexdigest() == (
        "6e9895ad58f87529f2af31f4f0895f5a2a47339c6c7294a4b857c9efbc20d6a3"
    )


def test_query_and_prediction_render_only_visible_example_and_retrieved_evidence() -> None:
    example = _example()

    query = render_query_prompt(example)
    prediction = render_prediction_prompt(example, _retrieval())

    assert "March 24, 2019" in query.text
    assert "It is part of a sentence long." in query.text
    assert query.text.count("[[TARGET]]") >= 1
    assert "Hidden answer" not in query.text
    assert "[DOCUMENT 1]\n[EXCERPT 1]\nPublic evidence." in prediction.text
    assert "It is part of a sentence long. Answer with the" in prediction.text
    assert "Hidden answer" not in prediction.text
    assert query.sha256 == hashlib.sha256(query.text.encode()).hexdigest()
    assert prediction.sha256 == hashlib.sha256(prediction.text.encode()).hexdigest()


def _example() -> VisibleExample:
    return VisibleExample(
        passage_id=UUID("00000000-0000-0000-0000-000000000001"),
        document_id=UUID("00000000-0000-0000-0000-000000000002"),
        document_date=date(2019, 3, 24),
        before_target="Visible before. ",
        after_target=" Visible after.",
        context_window="Visible before [[TARGET]] visible after.",
        length_cue=LengthEstimate(
            estimated_words=7,
            bucket="part of a sentence",
            median_character_width=0.01,
            median_characters_per_word=5.0,
            line_capacities=[35],
        ),
        visible_words=[],
        serialization_version="visible-transcript-v1",
    )


def _retrieval() -> RetrievalTrace:
    chunk = FusedChunk(
        chunk_id="1" * 64,
        text="Public evidence.",
        rrf_score=1.0,
    )
    documents = [
        FusedDocument(
            document_id="public-document",
            source_document_ids=["public-document"],
            rrf_score=1.0,
            chunks=[chunk],
        )
    ]
    return RetrievalTrace(
        context_query_sha256="2" * 64,
        own_query_sha256="3" * 64,
        lists=FourRankedLists(
            context_bm25=[],
            context_dense=[],
            own_bm25=[],
            own_dense=[],
        ),
        excluded_document_ids=[],
        documents=documents,
        serialized_documents="[DOCUMENT 1]\n[EXCERPT 1]\nPublic evidence.",
    )
