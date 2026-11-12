from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_FRUS_METADATA_URL = "https://history.state.gov/open/frus-metadata.xml"
_YEAR_PATTERN = re.compile(r"^\d{4}$")


class FrusMetadataError(ValueError):
    pass


class FrusPublicationManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: str
    retrieved_at: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication_years: dict[str, int | None]

    @property
    def known_publication_years(self) -> dict[str, int]:
        return {
            volume_id: year
            for volume_id, year in self.publication_years.items()
            if year is not None
        }

    def manifest_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def load_frus_metadata(
    path: Path,
    *,
    source_url: str = DEFAULT_FRUS_METADATA_URL,
    retrieved_at: str = "unknown",
) -> FrusPublicationManifest:
    try:
        source_bytes = path.read_bytes()
    except OSError as error:
        raise FrusMetadataError(f"Unable to read FRUS metadata: {path}") from error
    try:
        root = ET.fromstring(source_bytes)
    except ET.ParseError as error:
        raise FrusMetadataError("Official FRUS metadata is not valid XML") from error

    publication_years: dict[str, int | None] = {}
    for volume in root.findall("{*}volume"):
        volume_id = (volume.get("id") or "").strip()
        if not volume_id:
            raise FrusMetadataError("FRUS metadata contains a volume without an ID")
        if volume_id in publication_years:
            raise FrusMetadataError(
                f"FRUS metadata contains duplicate volume {volume_id}"
            )
        published = (volume.findtext("{*}published") or "").strip()
        if not published:
            publication_years[volume_id] = None
            continue
        if not _YEAR_PATTERN.fullmatch(published):
            raise FrusMetadataError(
                f"FRUS volume {volume_id} has an invalid publication year"
            )
        publication_years[volume_id] = int(published)
    if not publication_years:
        raise FrusMetadataError("Official FRUS metadata contains no volumes")

    return FrusPublicationManifest(
        source_url=source_url,
        retrieved_at=retrieved_at,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        publication_years=dict(sorted(publication_years.items())),
    )


def match_frus_volume(
    document_id: str,
    publication_years: Mapping[str, object],
) -> str:
    normalized_id = document_id.strip()
    for volume_id in sorted(publication_years, key=lambda item: (-len(item), item)):
        if normalized_id.startswith(volume_id):
            return volume_id
    raise FrusMetadataError(
        f"FRUS document ID {document_id!r} does not match an official volume"
    )
