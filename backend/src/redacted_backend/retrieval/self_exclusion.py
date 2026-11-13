from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.scientific_config import ScientificConfig

_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


class SelfDocumentIdentityError(ValueError):
    code = "self_document_identity_unresolved"


class UploadIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    filename: str
    archive_identifiers: tuple[str, ...] = ()
    visible_text: str


class CorpusDocumentCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    filename: str | None = None
    archive_identifiers: tuple[str, ...] = ()
    opening_text: str


class ExclusionDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    excluded_document_ids: tuple[str, ...]
    reasons: dict[str, str]
    matcher_version: str = "visible-shingle-self-match-v1"


class SelfDocumentMatcher:
    def __init__(self, scientific: ScientificConfig | None = None) -> None:
        self.scientific = scientific or ScientificConfig()

    def match(
        self,
        upload: UploadIdentity,
        corpus_candidates: list[CorpusDocumentCandidate],
    ) -> ExclusionDecision:
        upload_ids = {_normalize_identifier(value) for value in upload.archive_identifiers}
        reasons: dict[str, str] = {}
        unmatched: list[CorpusDocumentCandidate] = []
        for candidate in corpus_candidates:
            candidate_ids = {
                _normalize_identifier(value) for value in candidate.archive_identifiers
            }
            if upload_ids & candidate_ids:
                reasons[candidate.document_id] = "archive_identifier"
            else:
                unmatched.append(candidate)

        upload_shingles = _shingles(
            upload.visible_text,
            self.scientific.self_match_shingle_words,
        )
        scored = [
            (
                _jaccard(
                    upload_shingles,
                    _shingles(
                        candidate.opening_text,
                        self.scientific.self_match_shingle_words,
                    ),
                ),
                candidate,
            )
            for candidate in unmatched
        ]
        clear = [
            (score, candidate)
            for score, candidate in scored
            if score >= self.scientific.self_match_threshold
        ]
        borderline = [
            (score, candidate)
            for score, candidate in scored
            if self.scientific.self_match_ambiguity_floor
            <= score
            < self.scientific.self_match_threshold
        ]
        clear_filenames = {
            _normalize_filename(candidate.filename)
            for _, candidate in clear
            if candidate.filename
        }
        if (
            len(borderline) >= 2
            or (clear and borderline)
            or (len(clear) >= 2 and len(clear_filenames) >= 2)
        ):
            raise SelfDocumentIdentityError(
                "self_document_identity_unresolved: multiple corpus releases match "
                "the visible upload"
            )
        for _score, candidate in clear:
            reasons[candidate.document_id] = "visible_shingle_match"
        return ExclusionDecision(
            excluded_document_ids=tuple(sorted(reasons)),
            reasons=dict(sorted(reasons.items())),
        )


def _normalize_identifier(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalize_filename(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"\.[a-z0-9]{1,8}$", "", normalized)
    return " ".join(normalized.replace("_", " ").split())


def _shingles(text: str, width: int) -> set[tuple[str, ...]]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = _TOKEN.findall(normalized)
    if len(tokens) < width:
        return set()
    return {tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
