from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from redacted_backend.corpus.frus_metadata import (
    FrusMetadataError,
    load_frus_metadata,
    match_frus_volume,
)


def write_feed(path: Path, volumes: list[tuple[str, str]]) -> None:
    body = "".join(
        f'<volume id="{volume_id}"><published>{year}</published></volume>'
        for volume_id, year in volumes
    )
    path.write_text(
        f'<volumes xmlns="http://history.state.gov/ns/1.0">{body}</volumes>',
        encoding="utf-8",
    )


def test_official_frus_feed_is_parsed_and_hashed_without_losing_volume_ids(
    tmp_path: Path,
) -> None:
    feed = tmp_path / "frus-metadata.xml"
    write_feed(feed, [("frus1969-76v01", "2003"), ("frus1981-88v04", "2021")])

    manifest = load_frus_metadata(
        feed,
        source_url="https://history.state.gov/open/frus-metadata.xml",
        retrieved_at="2026-11-12T12:00:00Z",
    )

    assert manifest.publication_years == {
        "frus1969-76v01": 2003,
        "frus1981-88v04": 2021,
    }
    assert manifest.source_sha256 == hashlib.sha256(feed.read_bytes()).hexdigest()
    assert manifest.source_url == "https://history.state.gov/open/frus-metadata.xml"


def test_frus_document_matching_uses_the_longest_canonical_volume_prefix() -> None:
    years = {"frus1981-88v04": 2021, "frus1981-88v04p1": 2024}

    assert match_frus_volume("frus1981-88v04p1d123", years) == "frus1981-88v04p1"
    assert match_frus_volume("frus1981-88v04d9", years) == "frus1981-88v04"


def test_unmatched_frus_document_id_fails_closed() -> None:
    with pytest.raises(FrusMetadataError, match="does not match"):
        match_frus_volume("frus-unknown-document", {"frus1981-88v04": 2021})


def test_official_volume_with_missing_publication_year_is_retained_as_unknown(
    tmp_path: Path,
) -> None:
    feed = tmp_path / "frus-metadata.xml"
    write_feed(feed, [("frus1958-60v05mSupp", "")])

    manifest = load_frus_metadata(feed)

    assert manifest.publication_years == {"frus1958-60v05mSupp": None}
    assert manifest.known_publication_years == {}


@pytest.mark.parametrize(
    ("volumes", "message"),
    [
        ([("frus1", "2000"), ("frus1", "2001")], "duplicate"),
        ([("frus1", "not-a-year")], "publication year"),
    ],
)
def test_frus_feed_rejects_ambiguous_or_invalid_metadata(
    tmp_path: Path,
    volumes: list[tuple[str, str]],
    message: str,
) -> None:
    feed = tmp_path / "frus-metadata.xml"
    write_feed(feed, volumes)

    with pytest.raises(FrusMetadataError, match=message):
        load_frus_metadata(feed)
