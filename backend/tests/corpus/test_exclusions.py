from __future__ import annotations

from pathlib import Path

from redacted_backend.corpus.exclusions import (
    ExclusionManifest,
    load_exclusion_manifest,
    should_include,
)


def document(doc_id: str, source: str) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "classification": "CONFIDENTIAL",
        "authored": "Office",
        "title": "Visible title",
        "body": "Visible body",
        "source": source,
        "pg_cnt": 1,
        "char_cnt": 12,
        "word_cnt": 2,
    }


def complete_manifest() -> ExclusionManifest:
    return ExclusionManifest(
        version="v1-test",
        included_sources={"cia", "frus", "cfpf"},
        cia_reading_room_excluded_doc_ids={"cia-2019-release"},
        cia_confirmed_pre_cutoff_doc_ids={"cia-old"},
        frus_publication_years={
            "frus-old": 2018,
            "frus-cutoff": 2019,
            "frus-2024": 2024,
        },
        cia_coverage_complete=True,
        frus_coverage_complete=True,
    )


def test_excludes_the_cia_reading_room_post_cutoff_bucket() -> None:
    decision = should_include(document("cia-2019-release", "cia"), complete_manifest())

    assert decision.included is False
    assert decision.ready is True
    assert decision.reason == "cia_reading_room_2019_plus"


def test_excludes_frus_volumes_published_from_2019_through_2024() -> None:
    manifest = complete_manifest()

    assert should_include(document("frus-old", "frus"), manifest).included is True
    assert (
        should_include(document("frus-cutoff", "frus"), manifest).reason
        == "frus_published_2019_2024"
    )
    assert (
        should_include(document("frus-2024", "frus"), manifest).reason
        == "frus_published_2019_2024"
    )


def test_unknown_release_provenance_fails_closed() -> None:
    decision = should_include(document("cia-unmapped", "cia"), complete_manifest())

    assert decision.included is False
    assert decision.ready is False
    assert decision.reason == "unknown_release_provenance"


def test_unknown_collection_fails_closed_instead_of_broadening_the_corpus() -> None:
    decision = should_include(document("mystery-1", "unlisted"), complete_manifest())

    assert decision.included is False
    assert decision.ready is False
    assert decision.reason == "unknown_source"


def test_repository_manifest_stays_unready_until_identifier_maps_are_generated() -> (
    None
):
    path = Path(__file__).parents[2] / "corpus" / "exclusions" / "v1.yaml"

    manifest = load_exclusion_manifest(path)

    assert manifest.cia_coverage_complete is False
    assert manifest.frus_coverage_complete is False
    assert manifest.coverage_ready is False
