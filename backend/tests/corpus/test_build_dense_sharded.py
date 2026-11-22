from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import redacted_backend.corpus.build_dense_sharded as dense_builder
from redacted_backend.corpus.best_available import BestAvailableCorpusManifest
from redacted_backend.corpus.build_dense_sharded import (
    StorageBudgetError,
    build_sharded_dense_from_files,
)
from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.dense_profile import (
    DenseArtifactProfile,
    dense_semantics_hash,
)
from redacted_backend.retrieval.faiss_ivfpq import FaissIvfPqIndex
from redacted_backend.retrieval.types import (
    CompressedDenseShardManifest,
    CompressedShardedDenseIndexManifest,
)
from redacted_backend.scientific_config import ScientificConfig


class DeterministicEncoder:
    def __init__(self, dimension: int, *, fail_after_rows: int | None = None) -> None:
        self.dimension = dimension
        self.fail_after_rows = fail_after_rows
        self.encoded_rows = 0

    def encode_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if (
            self.fail_after_rows is not None
            and self.encoded_rows + len(texts) > self.fail_after_rows
        ):
            raise OSError("simulated encoding interruption")
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            index = int(text.rsplit(" ", 1)[-1]) % self.dimension
            vectors[row, index] = 1.0
        self.encoded_rows += len(texts)
        return vectors

    def encode_query(self, text: str) -> NDArray[np.float32]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        suffix = text.rsplit(" ", 1)[-1]
        index = int(suffix) % self.dimension if suffix.isdigit() else 0
        vector[index] = 1.0
        return vector


def _scientific() -> ScientificConfig:
    return ScientificConfig(embedding_dimension=8)


def _profile(**updates: object) -> DenseArtifactProfile:
    values = {
        "embedding_dimension": 8,
        "ivf_nlist": 1,
        "pq_subvectors": 1,
        "pq_bits": 1,
        "nprobe": 1,
        "candidate_count_per_shard": 5,
        "training_sample_count": 5,
        "shard_size_limit": 2,
        "max_artifact_bytes": 1024 * 1024,
    }
    values.update(updates)
    return DenseArtifactProfile(**values)


def _chunk(index: int) -> CorpusChunk:
    text = f"archive context {index}"
    return CorpusChunk(
        id=hashlib.sha256(f"chunk-{index}".encode()).hexdigest(),
        document_id=f"document-{index}",
        ordinal=0,
        text=text,
        word_count=3,
        title=f"Public title {index}",
        corpus="historylab-unlabeled",
        source_url="https://example.test/document.pdf",
        classification="UNCLASSIFIED",
        authored="1975-01-01",
        doc_lang=None,
    )


def _fixture(
    root: Path,
    count: int = 5,
) -> tuple[Path, Path, Path, list[CorpusChunk], ScientificConfig]:
    scientific = _scientific()
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
    manifest_path = root / "corpus.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    application_queries = root / "application-queries.txt"
    application_queries.write_text(
        "public query 0\npublic query 1\n",
        encoding="utf-8",
    )
    return chunks_path, manifest_path, application_queries, chunks, scientific


def _build(
    root: Path,
    *,
    encoder: DeterministicEncoder | None = None,
    profile: DenseArtifactProfile | None = None,
) -> tuple[
    CompressedShardedDenseIndexManifest,
    Path,
    DeterministicEncoder,
]:
    chunks, corpus, queries, _, scientific = _fixture(root)
    output = root / "dense"
    effective_encoder = encoder or DeterministicEncoder(8)
    report = build_sharded_dense_from_files(
        chunks,
        corpus,
        output,
        output / "manifest.json",
        application_queries_path=queries,
        certification_queries_path=output / "recall-queries.txt",
        encoder=effective_encoder,
        scientific=scientific,
        profile=profile or _profile(),
        batch_size=2,
        shard_size=2,
        application_query_count=2,
        corpus_query_count=2,
        recall_block_rows=2,
    )
    return report.index_manifest, output, effective_encoder


def test_build_publishes_compressed_shards_without_duplicate_metadata(
    tmp_path: Path,
) -> None:
    manifest, output, _ = _build(tmp_path)

    assert manifest.schema_version == "sharded-dense-index-manifest-v2"
    assert manifest.chunk_count == 5
    assert manifest.artifact_profile_hash == _profile().manifest_hash()
    assert manifest.dense_semantics_hash == dense_semantics_hash(_scientific())
    assert manifest.total_artifact_bytes <= manifest.artifact_profile.max_artifact_bytes
    assert (output / manifest.chunk_offsets_path).stat().st_size == 5 * 8
    assert (output / manifest.candidate_template_path).is_file()
    assert [(item.start_row, item.end_row) for item in manifest.shards] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    for shard in manifest.shards:
        directory = output / shard.path
        assert (directory / "embeddings.f16").stat().st_size == (
            shard.chunk_count * 8 * 2
        )
        assert (directory / "candidate.faiss").is_file()
        assert not (directory / "metadata.jsonl").exists()
        assert not (directory / "metadata.offsets.u64").exists()
        assert hashlib.sha256(
            (directory / "embeddings.f16").read_bytes()
        ).hexdigest() == shard.embeddings_sha256
        assert hashlib.sha256(
            (directory / "candidate.faiss").read_bytes()
        ).hexdigest() == shard.candidate_index_sha256
    assert manifest.recall_at_300 == 1.0
    assert (output / manifest.recall_report_path).is_file()


def test_embedding_phase_resumes_without_reencoding_verified_checkpoints(
    tmp_path: Path,
) -> None:
    chunks, corpus, queries, _, scientific = _fixture(tmp_path)
    output = tmp_path / "dense"
    first_encoder = DeterministicEncoder(8, fail_after_rows=2)
    kwargs = {
        "application_queries_path": queries,
        "certification_queries_path": output / "recall-queries.txt",
        "scientific": scientific,
        "profile": _profile(),
        "batch_size": 2,
        "shard_size": 2,
        "application_query_count": 2,
        "corpus_query_count": 2,
        "recall_block_rows": 2,
    }

    with pytest.raises(OSError, match="encoding interruption"):
        build_sharded_dense_from_files(
            chunks,
            corpus,
            output,
            output / "manifest.json",
            encoder=first_encoder,
            **kwargs,
        )
    assert first_encoder.encoded_rows == 2
    assert (output / "embedding-checkpoints/shard-00000/manifest.json").is_file()

    second_encoder = DeterministicEncoder(8)
    report = build_sharded_dense_from_files(
        chunks,
        corpus,
        output,
        output / "manifest.json",
        encoder=second_encoder,
        **kwargs,
    )

    assert report.index_manifest.chunk_count == 5
    assert second_encoder.encoded_rows == 3


def test_template_training_maps_each_embedding_shard_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_memmap = np.memmap
    checkpoint_reads = 0

    def tracking_memmap(filename, *args, **kwargs):
        nonlocal checkpoint_reads
        path = Path(filename)
        if _CHECKPOINT_PART in path.parts and kwargs.get("mode") == "r":
            checkpoint_reads += 1
        return original_memmap(filename, *args, **kwargs)

    _CHECKPOINT_PART = "embedding-checkpoints"
    monkeypatch.setattr(dense_builder.np, "memmap", tracking_memmap)

    _build(tmp_path)

    assert checkpoint_reads == 3


def test_corrupt_candidate_rebuilds_from_embeddings_without_encoding(
    tmp_path: Path,
) -> None:
    manifest, output, _ = _build(tmp_path)
    (output / "manifest.json").unlink()
    candidate = output / manifest.shards[0].path / "candidate.faiss"
    candidate.write_bytes(candidate.read_bytes() + b"tamper")
    rebuilt_candidate_encoder = DeterministicEncoder(8, fail_after_rows=0)

    repaired, _, _ = _build(tmp_path, encoder=rebuilt_candidate_encoder)

    assert repaired.chunk_count == 5
    assert rebuilt_candidate_encoder.encoded_rows == 0


def test_underfilled_candidate_rebuilds_from_embeddings_without_encoding(
    tmp_path: Path,
) -> None:
    manifest, output, _ = _build(tmp_path)
    (output / "manifest.json").unlink()
    first = manifest.shards[0]
    directory = output / first.path
    candidate_path = directory / "candidate.faiss"
    candidate = FaissIvfPqIndex.create(
        output / manifest.candidate_template_path,
        profile=manifest.artifact_profile,
    )
    embeddings = np.memmap(
        directory / "embeddings.f16",
        dtype=np.float16,
        mode="r",
        shape=(first.chunk_count, manifest.embedding_dimension),
    )
    candidate.add(np.asarray(embeddings[:1], dtype=np.float32))
    del embeddings
    candidate.save(candidate_path)
    underfilled = first.model_copy(
        update={
            "candidate_index_size_bytes": candidate_path.stat().st_size,
            "candidate_index_sha256": hashlib.sha256(
                candidate_path.read_bytes()
            ).hexdigest(),
        }
    )
    (directory / "manifest.json").write_text(
        CompressedDenseShardManifest.model_validate(underfilled).model_dump_json(),
        encoding="utf-8",
    )
    encoder = DeterministicEncoder(8, fail_after_rows=0)

    repaired, _, _ = _build(tmp_path, encoder=encoder)

    loaded = FaissIvfPqIndex.load(
        output / repaired.shards[0].path / "candidate.faiss",
        profile=repaired.artifact_profile,
    )
    assert loaded.ntotal == repaired.shards[0].chunk_count
    assert encoder.encoded_rows == 0


def test_corrupt_embedding_checkpoint_fails_closed(tmp_path: Path) -> None:
    chunks, corpus, queries, _, scientific = _fixture(tmp_path)
    output = tmp_path / "dense"
    first_encoder = DeterministicEncoder(8, fail_after_rows=2)
    with pytest.raises(OSError, match="encoding interruption"):
        build_sharded_dense_from_files(
            chunks,
            corpus,
            output,
            output / "manifest.json",
            application_queries_path=queries,
            certification_queries_path=output / "recall-queries.txt",
            encoder=first_encoder,
            scientific=scientific,
            profile=_profile(),
            batch_size=2,
            shard_size=2,
            application_query_count=2,
            corpus_query_count=2,
        )
    embedding = output / "embedding-checkpoints/shard-00000/embeddings.f16"
    embedding.write_bytes(embedding.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="embedding.*checksum"):
        _build(tmp_path, encoder=DeterministicEncoder(8, fail_after_rows=0))


def test_wrong_training_row_file_fails_closed(tmp_path: Path) -> None:
    manifest, output, _ = _build(tmp_path)
    (output / "manifest.json").unlink()
    training_rows = output / "training-rows.u64"
    training_rows.write_bytes(training_rows.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="training row.*checksum"):
        _build(tmp_path, encoder=DeterministicEncoder(8, fail_after_rows=0))

    assert manifest.chunk_count == 5
    assert not (output / "manifest.json").exists()


def test_candidate_write_failure_keeps_embeddings_but_does_not_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_save = FaissIvfPqIndex.save
    calls = 0

    def fail_second_save(self: FaissIvfPqIndex, path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated candidate write failure")
        original_save(self, path)

    monkeypatch.setattr(FaissIvfPqIndex, "save", fail_second_save)

    with pytest.raises(OSError, match="candidate write failure"):
        _build(tmp_path)

    assert (tmp_path / "dense/embedding-checkpoints/shard-00000/manifest.json").is_file()
    assert not (tmp_path / "dense/manifest.json").exists()


def test_stale_chunk_checksum_fails_before_encoding(tmp_path: Path) -> None:
    chunks, corpus, queries, _, scientific = _fixture(tmp_path)
    chunks.write_bytes(chunks.read_bytes() + b"tamper")
    encoder = DeterministicEncoder(8)

    with pytest.raises(ValueError, match="chunk checksum"):
        build_sharded_dense_from_files(
            chunks,
            corpus,
            tmp_path / "dense",
            tmp_path / "dense/manifest.json",
            application_queries_path=queries,
            certification_queries_path=tmp_path / "dense/recall-queries.txt",
            encoder=encoder,
            scientific=scientific,
            profile=_profile(),
            shard_size=2,
            application_query_count=2,
            corpus_query_count=2,
        )

    assert encoder.encoded_rows == 0


def test_incomplete_temporary_directory_is_never_reused(tmp_path: Path) -> None:
    output = tmp_path / "dense"
    stale = output / ".shard-00000.interrupted"
    stale.mkdir(parents=True)
    (stale / "embeddings.f16").write_bytes(b"invented")

    manifest, _, _ = _build(tmp_path)

    assert manifest.chunk_count == 5
    assert not stale.exists()


def test_insufficient_space_stops_before_the_first_embedding_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "redacted_backend.corpus.build_dense_sharded._available_bytes",
        lambda _path: 1,
    )
    encoder = DeterministicEncoder(8)

    with pytest.raises(
        StorageBudgetError,
        match=r"required=.*available=.*dense",
    ):
        _build(tmp_path, encoder=encoder)

    assert encoder.encoded_rows == 0


def test_final_artifact_size_over_profile_limit_refuses_publication(
    tmp_path: Path,
) -> None:
    with pytest.raises(StorageBudgetError, match="storage ceiling"):
        _build(tmp_path, profile=_profile(max_artifact_bytes=1))

    assert not (tmp_path / "dense/manifest.json").exists()


def test_rejects_more_than_one_million_chunks_per_shard(tmp_path: Path) -> None:
    chunks, corpus, queries, _, scientific = _fixture(tmp_path)

    with pytest.raises(ValueError, match="1000000"):
        build_sharded_dense_from_files(
            chunks,
            corpus,
            tmp_path / "dense",
            tmp_path / "dense/manifest.json",
            application_queries_path=queries,
            certification_queries_path=tmp_path / "dense/recall-queries.txt",
            encoder=DeterministicEncoder(8),
            scientific=scientific,
            profile=_profile(),
            shard_size=1_000_001,
            application_query_count=2,
            corpus_query_count=2,
        )
