from __future__ import annotations

import json
from pathlib import Path

import pytest

from redacted_backend.corpus.build_bm25 import build_bm25_from_files
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.scientific_config import ScientificConfig


def test_build_rejects_a_chunk_file_that_does_not_match_its_manifest(
    tmp_path: Path,
) -> None:
    scientific = ScientificConfig()
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text('{"tampered":true}\n', encoding="utf-8")
    corpus = CorpusManifest(
        schema_version="historylab-normalized-v1",
        source_manifest_hash="1" * 64,
        coverage_report_hash="4" * 64,
        exclusion_hash="2" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=1,
        chunk_count=1,
        corpus_counts={"cfpf": 1},
        exclusion_counts={},
        coverage_ready=True,
        chunks_sha256="3" * 64,
    )
    corpus_path = tmp_path / "corpus-manifest.json"
    corpus_path.write_text(
        json.dumps(corpus.model_dump(mode="json")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum does not match"):
        build_bm25_from_files(
            chunks_path,
            corpus_path,
            tmp_path / "bm25-manifest.json",
            opensearch_url="http://127.0.0.1:1",
        )

    assert not (tmp_path / "bm25-manifest.json").exists()
