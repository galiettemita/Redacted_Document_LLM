from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from redacted_backend.corpus.best_available import BestAvailableCorpusManifest
from redacted_backend.corpus.build_dense_sharded import benchmark_sharded_dense_build
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.dense_profile import DenseArtifactProfile
from redacted_backend.scientific_config import ScientificConfig


class _Encoder:
    def __init__(self) -> None:
        self.encoded_rows = 0

    def encode_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, int(text.rsplit(" ", 1)[-1]) % 4] = 1.0
        self.encoded_rows += len(texts)
        return vectors

    def encode_query(self, text: str) -> NDArray[np.float32]:
        del text
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _benchmark_fixture(root: Path) -> tuple[Path, Path, ScientificConfig]:
    scientific = ScientificConfig(embedding_dimension=4)
    chunks = [
        CorpusChunk(
            id=hashlib.sha256(f"chunk-{index}".encode()).hexdigest(),
            document_id=f"document-{index}",
            ordinal=0,
            text=f"benchmark passage {index}",
            word_count=3,
            title=None,
            corpus="historylab-unlabeled",
            source_url=None,
            classification=None,
            authored=None,
            doc_lang=None,
        )
        for index in range(4)
    ]
    chunks_path = root / "chunks.jsonl"
    chunks_path.write_text(
        "".join(
            json.dumps(chunk.model_dump(mode="json"), sort_keys=True) + "\n"
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
        parsed_row_count=4,
        included_row_count=4,
        empty_body_count=0,
        malformed_row_count=0,
        document_count=4,
        chunk_count=4,
        corpus_counts={"historylab-unlabeled": 4},
        chunks_sha256=hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
    )
    manifest_path = root / "corpus.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    return chunks_path, manifest_path, scientific


def test_benchmark_defaults_to_one_hundred_thousand_real_rows() -> None:
    parameter = inspect.signature(benchmark_sharded_dense_build).parameters[
        "sample_rows"
    ]

    assert parameter.default == 100_000


def test_benchmark_writes_nonpublishable_artifacts_and_projects_without_cost(
    tmp_path: Path,
) -> None:
    chunks, corpus, scientific = _benchmark_fixture(tmp_path)
    encoder = _Encoder()
    profile = DenseArtifactProfile(
        embedding_dimension=4,
        ivf_nlist=1,
        pq_subvectors=1,
        pq_bits=1,
        nprobe=1,
        candidate_count_per_shard=4,
        training_sample_count=4,
        shard_size_limit=4,
        max_artifact_bytes=1024 * 1024,
    )

    report = benchmark_sharded_dense_build(
        chunks,
        corpus,
        tmp_path / "benchmark",
        encoder=encoder,
        scientific=scientific,
        profile=profile,
        sample_rows=4,
        batch_size=2,
    )

    assert encoder.encoded_rows == 4
    assert report.sample_rows == 4
    assert report.production_chunk_count == 4
    assert report.publishable is False
    assert report.training_mode == "benchmark-only-all-sample-rows"
    assert report.rows_per_second > 0
    assert report.embedding_bytes_per_row == 4
    assert report.candidate_bytes_per_row > 0
    assert report.offset_bytes_per_row == 8
    assert report.projected_full_artifact_bytes > 0
    assert report.projected_gpu_hours >= 0
    assert "cost" not in report.model_dump(mode="json")
    assert (tmp_path / "benchmark/sample-embeddings.i8").is_file()
    assert not (tmp_path / "benchmark/sample-embeddings.f16").exists()
    assert (tmp_path / "benchmark/candidate.faiss").is_file()
    assert not (tmp_path / "benchmark/manifest.json").exists()
