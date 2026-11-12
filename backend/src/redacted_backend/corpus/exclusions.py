from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from redacted_backend.corpus.frus_metadata import FrusMetadataError, match_frus_volume


class ExclusionManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    known_corpora: set[str]
    required_corpora: set[str]
    cia_reading_room_corpus: str | None
    cia_mapping_evidence: list[str]
    cia_coverage_complete: bool
    frus_publication_years: dict[str, int]
    frus_metadata_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    frus_coverage_complete: bool
    expected_post_cutoff_frus_volumes: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def validate_policy(self) -> ExclusionManifest:
        if not self.required_corpora.issubset(self.known_corpora):
            raise ValueError("required corpora must be a subset of known corpora")
        if (
            self.cia_reading_room_corpus is not None
            and self.cia_reading_room_corpus not in self.known_corpora
        ):
            raise ValueError("CIA reading-room corpus must be a known corpus")
        if self.cia_coverage_complete and self.cia_reading_room_corpus is None:
            raise ValueError("complete CIA coverage requires a verified corpus mapping")
        if self.cia_coverage_complete and not self.cia_mapping_evidence:
            raise ValueError("complete CIA coverage requires source evidence")
        if self.frus_coverage_complete and (
            not self.frus_publication_years or self.frus_metadata_sha256 is None
        ):
            raise ValueError("complete FRUS coverage requires official frozen metadata")
        return self

    @property
    def coverage_ready(self) -> bool:
        return (
            self.cia_coverage_complete
            and self.cia_reading_room_corpus is not None
            and self.frus_coverage_complete
            and bool(self.frus_publication_years)
            and self.frus_metadata_sha256 is not None
        )

    def manifest_hash(self) -> str:
        canonical = json.dumps(
            {
                "version": self.version,
                "known_corpora": sorted(self.known_corpora),
                "required_corpora": sorted(self.required_corpora),
                "cia_reading_room_corpus": self.cia_reading_room_corpus,
                "cia_mapping_evidence": self.cia_mapping_evidence,
                "cia_coverage_complete": self.cia_coverage_complete,
                "frus_publication_years": dict(
                    sorted(self.frus_publication_years.items())
                ),
                "frus_metadata_sha256": self.frus_metadata_sha256,
                "frus_coverage_complete": self.frus_coverage_complete,
                "expected_post_cutoff_frus_volumes": (
                    self.expected_post_cutoff_frus_volumes
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class InclusionDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    included: bool
    ready: bool
    reason: str


def load_exclusion_manifest(path: Path) -> ExclusionManifest:
    with path.open("r", encoding="utf-8") as source:
        payload = yaml.safe_load(source)
    return ExclusionManifest.model_validate(payload)


def should_include(
    document: Mapping[str, Any],
    exclusion_manifest: ExclusionManifest,
) -> InclusionDecision:
    doc_id = str(document.get("doc_id", "")).strip()
    corpus = str(document.get("corpus", "")).strip().lower()
    if not doc_id:
        return InclusionDecision(
            included=False,
            ready=False,
            reason="unknown_document_identifier",
        )
    if corpus not in exclusion_manifest.known_corpora:
        return InclusionDecision(
            included=False,
            ready=False,
            reason="unknown_corpus",
        )

    if corpus == exclusion_manifest.cia_reading_room_corpus:
        return InclusionDecision(
            included=False,
            ready=exclusion_manifest.cia_coverage_complete,
            reason=(
                "cia_reading_room_2019_plus"
                if exclusion_manifest.cia_coverage_complete
                else "cia_collection_mapping_unverified"
            ),
        )

    if corpus == "frus":
        try:
            volume_id = match_frus_volume(
                doc_id,
                exclusion_manifest.frus_publication_years,
            )
        except FrusMetadataError:
            return InclusionDecision(
                included=False,
                ready=False,
                reason="unknown_release_provenance",
            )
        publication_year = exclusion_manifest.frus_publication_years[volume_id]
        if 2019 <= publication_year <= 2024:
            return InclusionDecision(
                included=False,
                ready=exclusion_manifest.frus_coverage_complete,
                reason="frus_published_2019_2024",
            )
        if publication_year > 2024:
            return InclusionDecision(
                included=False,
                ready=False,
                reason="post_cutoff_release",
            )

    return InclusionDecision(included=True, ready=True, reason="included")
