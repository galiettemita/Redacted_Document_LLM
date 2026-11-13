from __future__ import annotations

import hashlib

from redacted_backend.domain.retrieval import FusedChunk, FusedDocument
from redacted_backend.retrieval.serialization import serialize_prompt_documents


def test_prompt_documents_use_only_exact_document_and_excerpt_markers() -> None:
    documents = [
        FusedDocument(
            document_id="doc-a",
            source_document_ids=["doc-a"],
            rrf_score=1.0,
            chunks=[
                FusedChunk(
                    chunk_id=hashlib.sha256(b"a").hexdigest(),
                    text="First retrieved passage.",
                    rrf_score=0.5,
                ),
                FusedChunk(
                    chunk_id=hashlib.sha256(b"b").hexdigest(),
                    text="Second retrieved passage.",
                    rrf_score=0.25,
                ),
            ],
        ),
        FusedDocument(
            document_id="doc-b",
            source_document_ids=["doc-b"],
            rrf_score=0.5,
            chunks=[
                FusedChunk(
                    chunk_id=hashlib.sha256(b"c").hexdigest(),
                    text="Only retrieved passage.",
                    rrf_score=0.2,
                )
            ],
        ),
    ]

    assert serialize_prompt_documents(documents) == (
        "[DOCUMENT 1]\n"
        "[EXCERPT 1]\n"
        "First retrieved passage.\n"
        "[EXCERPT 2]\n"
        "Second retrieved passage.\n\n"
        "[DOCUMENT 2]\n"
        "[EXCERPT 1]\n"
        "Only retrieved passage."
    )
