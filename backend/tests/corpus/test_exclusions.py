from __future__ import annotations

from pathlib import Path

from redacted_backend.corpus.exclusions import (
    ExclusionManifest,
    load_exclusion_manifest,
    should_include,
)


def document(doc_id: str, corpus: str) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "classification": "CONFIDENTIAL",
        "authored": "1975-01-01",
        "title": "Visible title",
        "body": "Visible body",
        "corpus": corpus,
        "source_url": "https://example.test/source",
        "pg_cnt": 1,
        "char_cnt": 12,
        "word_cnt": 2,
        "doc_lang": "en",
    }


def complete_manifest() -> ExclusionManifest:
    return ExclusionManifest(
        version="v2-test",
        known_corpora={"briefing", "frus", "cfpf", "cables"},
        required_corpora={"cables"},
        cia_reading_room_corpus="briefing",
        cia_mapping_evidence=["paper Appendix C", "History Lab collections"],
        cia_coverage_complete=True,
        frus_publication_years={
            "frus-old": 2018,
            "frus-cutoff": 2019,
            "frus-2024": 2024,
        },
        frus_metadata_sha256="a" * 64,
        frus_coverage_complete=True,
        expected_post_cutoff_frus_volumes=2,
    )


def test_excludes_the_entire_verified_cia_reading_room_collection() -> None:
    decision = should_include(
        document("briefing-any-id", "briefing"), complete_manifest()
    )

    assert decision.included is False
    assert decision.ready is True
    assert decision.reason == "cia_reading_room_2019_plus"


def test_excludes_frus_volumes_published_from_2019_through_2024() -> None:
    manifest = complete_manifest()

    assert should_include(document("frus-oldd1", "frus"), manifest).included is True
    assert (
        should_include(document("frus-cutoffd1", "frus"), manifest).reason
        == "frus_published_2019_2024"
    )
    assert (
        should_include(document("frus-2024d1", "frus"), manifest).reason
        == "frus_published_2019_2024"
    )


def test_unknown_release_provenance_fails_closed() -> None:
    decision = should_include(document("frus-unmapped", "frus"), complete_manifest())

    assert decision.included is False
    assert decision.ready is False
    assert decision.reason == "unknown_release_provenance"


def test_unknown_collection_fails_closed_instead_of_broadening_the_corpus() -> None:
    decision = should_include(document("mystery-1", "unlisted"), complete_manifest())

    assert decision.included is False
    assert decision.ready is False
    assert decision.reason == "unknown_corpus"


def test_repository_manifest_stays_unready_until_public_mappings_are_verified() -> None:
    path = Path(__file__).parents[2] / "corpus" / "exclusions" / "v1.yaml"

    manifest = load_exclusion_manifest(path)

    assert manifest.cia_reading_room_corpus == "briefing"
    assert manifest.cia_coverage_complete is True
    assert manifest.cia_mapping_evidence
    assert manifest.frus_coverage_complete is False
    assert manifest.coverage_ready is False
