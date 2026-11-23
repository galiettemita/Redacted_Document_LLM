from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import numpy as np

from redacted_backend.corpus.faiss_build_worker import (
    FaissBenchmarkJob,
    FaissProductionJob,
    run_faiss_benchmark_job,
    run_faiss_production_job,
)
from redacted_backend.corpus.build_dense_sharded import (
    EmbeddingShardCheckpoint,
    TrainingRowsManifest,
)
from redacted_backend.retrieval.dense_profile import dense_training_selection_hash
from redacted_backend.retrieval.dense_profile import DenseArtifactProfile
from redacted_backend.retrieval.int8_embeddings import quantize_directions


def _profile() -> DenseArtifactProfile:
    return DenseArtifactProfile(
        embedding_dimension=8,
        ivf_nlist=1,
        pq_subvectors=1,
        pq_bits=1,
        nprobe=1,
        candidate_count_per_shard=5,
        training_sample_count=5,
        shard_size_limit=5,
        max_artifact_bytes=1024 * 1024,
    )


def test_faiss_benchmark_runs_in_a_fresh_torch_free_process(
    tmp_path: Path,
) -> None:
    vectors = np.eye(8, dtype=np.float32)[:5]
    embeddings_path = tmp_path / "sample-embeddings.i8"
    quantize_directions(vectors).tofile(embeddings_path)
    output = tmp_path / "faiss"
    output.mkdir()

    result = run_faiss_benchmark_job(
        FaissBenchmarkJob(
            embeddings_path=embeddings_path,
            output_directory=output,
            sample_rows=5,
            batch_size=2,
            top_chunks_per_list=5,
            profile=_profile(),
        )
    )

    assert result.torch_imported is False
    assert result.certification_probe_queries == 5
    assert result.training_seconds >= 0
    assert result.candidate_add_seconds >= 0
    assert result.certification_probe_seconds >= 0
    assert (output / "candidate-template.faiss").is_file()
    assert (output / "empty-candidate.faiss").is_file()
    assert (output / "candidate.faiss").is_file()
    assert "torch" not in sys.modules


def test_production_faiss_artifacts_and_recall_run_in_torch_free_process(
    tmp_path: Path,
) -> None:
    profile = _profile().model_copy(update={"shard_size_limit": 2})
    output = tmp_path / "dense"
    checkpoint_root = output / "embedding-checkpoints"
    checkpoint_root.mkdir(parents=True)
    vectors = np.eye(8, dtype=np.float32)[:5]
    quantized = quantize_directions(vectors)
    checkpoints: list[EmbeddingShardCheckpoint] = []
    for shard_id, start in enumerate(range(0, 5, 2)):
        end = min(start + 2, 5)
        name = f"shard-{shard_id:05d}"
        directory = checkpoint_root / name
        directory.mkdir()
        payload = quantized[start:end].tobytes()
        (directory / profile.embedding_filename).write_bytes(payload)
        checkpoints.append(
            EmbeddingShardCheckpoint(
                dense_semantics_hash="a" * 64,
                chunks_sha256="b" * 64,
                shard_id=shard_id,
                path=name,
                start_row=start,
                end_row=end,
                chunk_count=end - start,
                embeddings_size_bytes=len(payload),
                embeddings_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    rows = np.arange(5, dtype="<u8")
    rows_payload = rows.tobytes()
    (output / "training-rows.u64").write_bytes(rows_payload)
    training = TrainingRowsManifest(
        training_selection_hash=dense_training_selection_hash(profile),
        chunks_sha256="b" * 64,
        chunk_count=5,
        sample_count=5,
        selection_version=profile.training_selection_version,
        rows_size_bytes=len(rows_payload),
        rows_sha256=hashlib.sha256(rows_payload).hexdigest(),
    )
    query_vectors = vectors[:2]
    query_path = output / ".recall-query-vectors.f32"
    query_vectors.tofile(query_path)

    result = run_faiss_production_job(
        FaissProductionJob(
            output_directory=output,
            checkpoints=[item.model_dump(mode="json") for item in checkpoints],
            training_rows=training.model_dump(mode="json"),
            profile=profile,
            query_vectors_path=query_path,
            query_count=2,
            batch_size=2,
            recall_block_rows=2,
            top_chunks_per_list=5,
        )
    )

    assert result.torch_imported is False
    assert len(result.shards) == 3
    assert len(result.per_query_recall) == 2
    assert all(value == 1.0 for value in result.per_query_recall)
    assert (output / "candidate-template.faiss").is_file()
    assert all((output / item["path"] / "candidate.faiss").is_file() for item in result.shards)
    assert "torch" not in sys.modules
