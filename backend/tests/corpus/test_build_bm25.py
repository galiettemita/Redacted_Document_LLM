from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import redacted_backend.corpus.build_bm25 as build_bm25_module
from redacted_backend.corpus.best_available import (
    BEST_AVAILABLE_BM25_INDEX_NAME,
    BestAvailableCorpusManifest,
)
from redacted_backend.corpus.build_bm25 import (
    build_best_available_bm25,
    build_bm25_from_files,
)
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.scientific_config import ScientificConfig


class DurableSearchTransport:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.replace_calls: list[str] = []
        self.bulk_calls: list[list[str]] = []
        self.refreshed: list[str] = []

    def replace_index(self, index_name: str, definition: dict[str, Any]) -> None:
        self.replace_calls.append(index_name)
        self.documents.clear()

    def bulk_index(
        self,
        index_name: str,
        documents: Sequence[dict[str, Any]],
    ) -> None:
        self.bulk_calls.append([str(item["chunk_id"]) for item in documents])
        for document in documents:
            self.documents[str(document["chunk_id"])] = dict(document)

    def refresh(self, index_name: str) -> None:
        self.refreshed.append(index_name)

    def search(self, index_name: str, request: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    def count(self, index_name: str) -> int:
        return len(self.documents)


def _chunk(index: int) -> CorpusChunk:
    text = f"Visible archive context {index}"
    return CorpusChunk(
        id=hashlib.sha256(f"chunk-{index}".encode()).hexdigest(),
        document_id=f"document-{index}",
        ordinal=0,
        text=text,
        word_count=len(text.split()),
        title=f"Title {index}",
        corpus="historylab-unlabeled",
        source_url="https://example.test/source",
        classification="UNCLASSIFIED",
        authored="1975-01-01",
        doc_lang=None,
    )


def _best_available_fixture(
    root: Path,
    count: int,
) -> tuple[Path, Path, list[CorpusChunk]]:
    scientific = ScientificConfig()
    chunks = [_chunk(index) for index in range(count)]
    chunks_path = root / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        "".join(
            json.dumps(
                chunk.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
    )
    manifest = BestAvailableCorpusManifest(
        source_profile_hash="1" * 64,
        source_sha256="2" * 64,
        source_size_bytes=100,
        audit_manifest_hash="3" * 64,
        limitations_sha256="4" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        parsed_row_count=count,
        included_row_count=count,
        empty_body_count=0,
        malformed_row_count=0,
        document_count=count,
        chunk_count=count,
        corpus_counts={"historylab-unlabeled": count},
        chunks_sha256=hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
    )
    manifest_path = root / "corpus-manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    return chunks_path, manifest_path, chunks


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


def test_best_available_bm25_resumes_idempotently_after_uncheckpointed_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks_path, corpus_path, chunks = _best_available_fixture(tmp_path, 5)
    output = tmp_path / "bm25-manifest.json"
    checkpoint = tmp_path / "bm25-checkpoint.json"
    transport = DurableSearchTransport()
    original_write = build_bm25_module._write_checkpoint_atomic
    writes = 0

    def fail_before_second_checkpoint(path: Path, value: object) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated interruption")
        original_write(path, value)

    monkeypatch.setattr(
        build_bm25_module,
        "_write_checkpoint_atomic",
        fail_before_second_checkpoint,
    )
    with pytest.raises(OSError, match="simulated interruption"):
        build_best_available_bm25(
            chunks_path,
            corpus_path,
            output,
            checkpoint,
            opensearch_url="http://search.test:9200",
            transport=transport,
            batch_size=2,
        )
    assert not output.exists()
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["indexed_chunk_count"] == 2
    assert len(transport.documents) == 4

    monkeypatch.setattr(
        build_bm25_module,
        "_write_checkpoint_atomic",
        original_write,
    )
    report = build_best_available_bm25(
        chunks_path,
        corpus_path,
        output,
        checkpoint,
        opensearch_url="http://search.test:9200",
        transport=transport,
        batch_size=2,
    )

    assert transport.replace_calls == [BEST_AVAILABLE_BM25_INDEX_NAME]
    assert set(transport.documents) == {chunk.id for chunk in chunks}
    assert transport.bulk_calls == [
        [chunks[0].id, chunks[1].id],
        [chunks[2].id, chunks[3].id],
        [chunks[2].id, chunks[3].id],
        [chunks[4].id],
    ]
    assert report.index_manifest.index_name == BEST_AVAILABLE_BM25_INDEX_NAME
    assert report.index_manifest.retrieval_profile == (
        "best_available_foiarchive_v1"
    )
    assert report.index_manifest.chunk_count == 5
    assert not checkpoint.exists()


def test_best_available_bm25_rejects_an_incompatible_checkpoint(
    tmp_path: Path,
) -> None:
    chunks_path, corpus_path, _ = _best_available_fixture(tmp_path, 1)
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": "best-available-bm25-checkpoint-v1",
                "corpus_manifest_hash": "0" * 64,
                "scientific_hash": ScientificConfig().manifest_hash(),
                "index_definition_hash": "1" * 64,
                "index_name": BEST_AVAILABLE_BM25_INDEX_NAME,
                "indexed_chunk_count": 0,
                "last_chunk_id": None,
            }
        ),
        encoding="utf-8",
    )
    transport = DurableSearchTransport()

    with pytest.raises(ValueError, match="checkpoint is incompatible"):
        build_best_available_bm25(
            chunks_path,
            corpus_path,
            tmp_path / "output.json",
            checkpoint,
            opensearch_url="http://search.test:9200",
            transport=transport,
        )

    assert transport.replace_calls == []
    assert not (tmp_path / "output.json").exists()


def test_best_available_bm25_rejects_live_index_behind_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks_path, corpus_path, chunks = _best_available_fixture(tmp_path, 3)
    output = tmp_path / "output.json"
    checkpoint = tmp_path / "checkpoint.json"
    transport = DurableSearchTransport()
    original_write = build_bm25_module._write_checkpoint_atomic

    def stop_after_checkpoint(path: Path, value: object) -> None:
        original_write(path, value)
        raise OSError("stop")

    monkeypatch.setattr(
        build_bm25_module,
        "_write_checkpoint_atomic",
        stop_after_checkpoint,
    )
    with pytest.raises(OSError, match="stop"):
        build_best_available_bm25(
            chunks_path,
            corpus_path,
            output,
            checkpoint,
            opensearch_url="http://search.test:9200",
            transport=transport,
            batch_size=2,
        )
    del transport.documents[chunks[0].id]
    monkeypatch.setattr(
        build_bm25_module,
        "_write_checkpoint_atomic",
        original_write,
    )

    with pytest.raises(ValueError, match="behind checkpoint"):
        build_best_available_bm25(
            chunks_path,
            corpus_path,
            output,
            checkpoint,
            opensearch_url="http://search.test:9200",
            transport=transport,
            batch_size=2,
        )

    assert not output.exists()
