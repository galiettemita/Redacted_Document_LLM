from __future__ import annotations

import pytest

from redacted_backend.retrieval.self_exclusion import (
    CorpusDocumentCandidate,
    SelfDocumentIdentityError,
    SelfDocumentMatcher,
    UploadIdentity,
)


VISIBLE_TEXT = " ".join(
    "the embassy reported a ministerial meeting in paris on monday concerning trade".split()
    * 5
)


def test_self_matcher_excludes_known_aliases_before_text_matching() -> None:
    decision = SelfDocumentMatcher().match(
        UploadIdentity(
            filename="upload.pdf",
            archive_identifiers=("cia-123",),
            visible_text="short visible text",
        ),
        [
            CorpusDocumentCandidate(
                document_id="corpus-a",
                filename="different.pdf",
                archive_identifiers=("CIA-123", "release-9"),
                opening_text="unrelated",
            )
        ],
    )

    assert decision.excluded_document_ids == ("corpus-a",)
    assert decision.reasons == {"corpus-a": "archive_identifier"}


def test_self_matcher_excludes_a_clear_same_release_near_duplicate() -> None:
    decision = SelfDocumentMatcher().match(
        UploadIdentity(filename="bulletin.pdf", visible_text=VISIBLE_TEXT),
        [
            CorpusDocumentCandidate(
                document_id="same-release",
                filename="bulletin archive copy.pdf",
                opening_text=f"{VISIBLE_TEXT} additional visible footer",
            ),
            CorpusDocumentCandidate(
                document_id="unrelated",
                filename="another.pdf",
                opening_text="an unrelated diplomatic record about another continent",
            ),
        ],
    )

    assert decision.excluded_document_ids == ("same-release",)
    assert decision.reasons["same-release"] == "visible_shingle_match"


def test_self_matcher_fails_closed_for_multiple_possible_releases() -> None:
    candidates = [
        CorpusDocumentCandidate(
            document_id=f"release-{index}",
            filename=f"release-{index}.pdf",
            opening_text=f"{VISIBLE_TEXT} release marker {index}",
        )
        for index in range(2)
    ]

    with pytest.raises(
        SelfDocumentIdentityError,
        match="self_document_identity_unresolved",
    ):
        SelfDocumentMatcher().match(
            UploadIdentity(filename="unknown.pdf", visible_text=VISIBLE_TEXT),
            candidates,
        )


def test_self_matcher_leaves_a_clearly_unrelated_document_available() -> None:
    decision = SelfDocumentMatcher().match(
        UploadIdentity(filename="bulletin.pdf", visible_text=VISIBLE_TEXT),
        [
            CorpusDocumentCandidate(
                document_id="unrelated",
                filename="another.pdf",
                opening_text="weather observations and agricultural production statistics",
            )
        ],
    )

    assert decision.excluded_document_ids == ()
    assert decision.reasons == {}
