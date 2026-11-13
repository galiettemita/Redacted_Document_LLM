from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from redacted_backend.corpus.build_dense import FaissHnswIndex, build_dense_from_files
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.retrieval.types import DenseIndexManifest
from redacted_backend.scientific_config import ScientificConfig


class FixedEncoder:
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([_vector(text) for text in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(_vector(text), dtype=np.float32)


class ExactCandidateBuilder:
    def __init__(self) -> None:
        self.vectors = np.empty((0, 2), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.concatenate((self.vectors, vectors), axis=0)

    def search(self, query: np.ndarray, candidates: int) -> np.ndarray:
        scores = self.vectors @ query
        rows = sorted(range(len(scores)), key=lambda row: (-float(scores[row]), row))
        return np.asarray(rows[:candidates], dtype=np.int64)

    def save(self, path: Path) -> None:
        path.write_bytes(b"real-index-fixture")


def test_build_dense_writes_checksummed_float16_artifacts_and_recall_gate(
    tmp_path: Path,
) -> None:
    scientific = ScientificConfig(
        embedding_dimension=2,
        dense_candidate_count=300,
    )
    chunks = [_chunk(0, "first evidence"), _chunk(1, "second evidence")]
    chunks_path, corpus_path = _write_corpus(tmp_path, chunks, scientific)
    output_dir = tmp_path / "dense"
    manifest_path = tmp_path / "dense-manifest.json"
    builder = ExactCandidateBuilder()

    report = build_dense_from_files(
        chunks_path,
        corpus_path,
        output_dir,
        manifest_path,
        recall_queries=["first evidence", "second evidence"],
        encoder=FixedEncoder(),
        candidate_index=builder,
        scientific=scientific,
        batch_size=1,
    )

    manifest = DenseIndexManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    assert report.index_manifest == manifest
    assert manifest.chunk_count == 2
    assert manifest.recall_at_300 == 1.0
    assert manifest.corpus_manifest_hash == report.corpus_manifest_hash
    assert manifest.embeddings_sha256 == _sha256(output_dir / "embeddings.f16")
    assert manifest.metadata_sha256 == _sha256(output_dir / "metadata.jsonl")
    assert manifest.candidate_index_sha256 == _sha256(output_dir / "candidate.faiss")
    embeddings = np.memmap(
        output_dir / "embeddings.f16",
        dtype=np.float16,
        mode="r",
        shape=(2, 2),
    )
    np.testing.assert_allclose(embeddings, [[1.0, 0.0], [0.0, 1.0]])
    assert builder.vectors.shape == (2, 2)


def test_build_dense_rejects_a_tampered_chunk_file_before_encoding(
    tmp_path: Path,
) -> None:
    scientific = ScientificConfig(embedding_dimension=2, dense_candidate_count=300)
    chunks_path, corpus_path = _write_corpus(
        tmp_path,
        [_chunk(0, "first evidence")],
        scientific,
    )
    chunks_path.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="checksum does not match"):
        build_dense_from_files(
            chunks_path,
            corpus_path,
            tmp_path / "dense",
            tmp_path / "dense-manifest.json",
            recall_queries=["first evidence"],
            encoder=FixedEncoder(),
            candidate_index=ExactCandidateBuilder(),
            scientific=scientific,
        )

    assert not (tmp_path / "dense-manifest.json").exists()


def test_real_faiss_hnsw_index_round_trips_candidates(tmp_path: Path) -> None:
    scientific = ScientificConfig(
        embedding_dimension=2,
        dense_hnsw_m=2,
        dense_ef_construction=16,
        dense_ef_search=16,
    )
    index = FaissHnswIndex.create(scientific)
    index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    path = tmp_path / "candidate.faiss"
    index.save(path)

    loaded = FaissHnswIndex.load(path, ef_search=scientific.dense_ef_search)
    rows = loaded.search(np.asarray([1.0, 0.0], dtype=np.float32), candidates=2)

    assert rows.tolist() == [0, 1]


def _vector(text: str) -> list[float]:
    return [1.0, 0.0] if "first" in text else [0.0, 1.0]


def _chunk(index: int, text: str) -> CorpusChunk:
    return CorpusChunk(
        id=hashlib.sha256(f"chunk-{index}".encode()).hexdigest(),
        document_id=f"document-{index}",
        ordinal=0,
        text=text,
        word_count=2,
        title=f"Title {index}",
        corpus="cfpf",
        source_url=None,
        classification=None,
        authored=None,
        doc_lang="en",
    )


def _write_corpus(
    tmp_path: Path,
    chunks: Sequence[CorpusChunk],
    scientific: ScientificConfig,
) -> tuple[Path, Path]:
    chunks_path = tmp_path / "chunks.jsonl"
    serialized = "".join(
        json.dumps(chunk.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n"
        for chunk in chunks
    )
    chunks_path.write_text(serialized, encoding="utf-8")
    corpus = CorpusManifest(
        schema_version="historylab-normalized-v1",
        source_manifest_hash="1" * 64,
        coverage_report_hash="4" * 64,
        exclusion_hash="2" * 64,
        scientific_hash=scientific.manifest_hash(),
        chunker_version=scientific.corpus_chunker_version,
        chunk_words=scientific.chunk_words,
        chunk_overlap=scientific.chunk_overlap,
        document_count=len(chunks),
        chunk_count=len(chunks),
        corpus_counts={"cfpf": len(chunks)},
        exclusion_counts={},
        coverage_ready=True,
        chunks_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
    )
    corpus_path = tmp_path / "corpus-manifest.json"
    corpus_path.write_text(
        json.dumps(corpus.model_dump(mode="json")),
        encoding="utf-8",
    )
    return chunks_path, corpus_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
