from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class ExclusionManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    included_sources: set[str]
    cia_reading_room_excluded_doc_ids: set[str]
    cia_confirmed_pre_cutoff_doc_ids: set[str]
    frus_publication_years: dict[str, int]
    cia_coverage_complete: bool
    frus_coverage_complete: bool

    @property
    def coverage_ready(self) -> bool:
        return self.cia_coverage_complete and self.frus_coverage_complete

    def manifest_hash(self) -> str:
        canonical = json.dumps(
            {
                "version": self.version,
                "included_sources": sorted(self.included_sources),
                "cia_reading_room_excluded_doc_ids": sorted(
                    self.cia_reading_room_excluded_doc_ids
                ),
                "cia_confirmed_pre_cutoff_doc_ids": sorted(
                    self.cia_confirmed_pre_cutoff_doc_ids
                ),
                "frus_publication_years": dict(
                    sorted(self.frus_publication_years.items())
                ),
                "cia_coverage_complete": self.cia_coverage_complete,
                "frus_coverage_complete": self.frus_coverage_complete,
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
    source = str(document.get("source", "")).strip().lower()
    if not doc_id:
        return InclusionDecision(
            included=False,
            ready=False,
            reason="unknown_document_identifier",
        )
    if source not in exclusion_manifest.included_sources:
        return InclusionDecision(
            included=False,
            ready=False,
            reason="unknown_source",
        )

    if source == "cia":
        if doc_id in exclusion_manifest.cia_reading_room_excluded_doc_ids:
            return InclusionDecision(
                included=False,
                ready=True,
                reason="cia_reading_room_2019_plus",
            )
        if doc_id in exclusion_manifest.cia_confirmed_pre_cutoff_doc_ids:
            return InclusionDecision(included=True, ready=True, reason="included")
        return InclusionDecision(
            included=False,
            ready=False,
            reason="unknown_release_provenance",
        )

    if source == "frus":
        publication_year = exclusion_manifest.frus_publication_years.get(doc_id)
        if publication_year is None:
            return InclusionDecision(
                included=False,
                ready=False,
                reason="unknown_release_provenance",
            )
        if 2019 <= publication_year <= 2024:
            return InclusionDecision(
                included=False,
                ready=True,
                reason="frus_published_2019_2024",
            )
        if publication_year > 2024:
            return InclusionDecision(
                included=False,
                ready=False,
                reason="post_cutoff_release",
            )

    return InclusionDecision(included=True, ready=True, reason="included")
