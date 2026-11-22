from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from redacted_backend.retrieval.dense_profile import (
    DenseArtifactProfile,
    dense_semantics_hash,
    dense_training_selection_hash,
    faiss_layout_hash,
)
from redacted_backend.retrieval.types import (
    CompressedDenseShardManifest,
    CompressedShardedDenseIndexManifest,
    DenseFaissVerification,
    DenseLatencyReport,
)
from redacted_backend.scientific_config import ScientificConfig


def test_default_dense_artifact_profile_freezes_the_approved_build_contract() -> None:
    profile = DenseArtifactProfile()

    assert profile.candidate_index == "faiss-ivfpq-exact-rescore-v1"
    assert profile.ivf_nlist == 4_096
    assert profile.pq_subvectors == 96
    assert profile.pq_bits == 8
    assert profile.nprobe == 256
    assert profile.candidate_count_per_shard == 4_000
    assert profile.training_sample_count == 200_000
    assert profile.max_parallel_shards == 4
    assert profile.max_artifact_bytes == 28 * 1024**3
    assert profile.max_warm_search_seconds == 5.0
    assert profile.manifest_hash() == DenseArtifactProfile().manifest_hash()


def test_dense_semantics_hash_ignores_unrelated_scientific_settings() -> None:
    scientific = ScientificConfig()
    unrelated_change = scientific.model_copy(update={"rrf_k": scientific.rrf_k + 1})
    semantic_change = scientific.model_copy(
        update={"embedding_query_instruction": "Different query instruction: "}
    )

    assert dense_semantics_hash(scientific) == dense_semantics_hash(unrelated_change)
    assert dense_semantics_hash(scientific) != dense_semantics_hash(semantic_change)


@pytest.mark.parametrize("pq_subvectors", [0, 95, 769])
def test_dense_profile_rejects_pq_layouts_that_cannot_encode_vectors(
    pq_subvectors: int,
) -> None:
    with pytest.raises(ValidationError, match="PQ subvectors|greater than or equal"):
        DenseArtifactProfile(pq_subvectors=pq_subvectors)


def _profile() -> DenseArtifactProfile:
    return DenseArtifactProfile(
        embedding_dimension=8,
        ivf_nlist=2,
        pq_subvectors=2,
        nprobe=2,
        candidate_count_per_shard=4,
        training_sample_count=8,
        shard_size_limit=3,
    )


def test_phase_hashes_allow_search_tuning_without_reencoding() -> None:
    profile = _profile()
    search_tuned = profile.model_copy(
        update={"nprobe": 1, "candidate_count_per_shard": 2}
    )
    layout_tuned = profile.model_copy(update={"pq_bits": 7})
    selection_tuned = profile.model_copy(update={"training_sample_count": 9})

    assert faiss_layout_hash(search_tuned) == faiss_layout_hash(profile)
    assert dense_training_selection_hash(search_tuned) == (
        dense_training_selection_hash(profile)
    )
    assert faiss_layout_hash(layout_tuned) != faiss_layout_hash(profile)
    assert dense_training_selection_hash(selection_tuned) != (
        dense_training_selection_hash(profile)
    )


def _shard_data(profile: DenseArtifactProfile) -> dict[str, object]:
    return {
        "artifact_profile_hash": profile.manifest_hash(),
        "candidate_template_sha256": "9" * 64,
        "shard_id": 0,
        "path": "shard-00000",
        "start_row": 0,
        "end_row": 2,
        "chunk_count": 2,
        "embeddings_size_bytes": 2 * profile.embedding_dimension * 2,
        "embeddings_sha256": "1" * 64,
        "candidate_index_size_bytes": 32,
        "candidate_index_sha256": "2" * 64,
    }


def _manifest_data(
    profile: DenseArtifactProfile,
    *,
    shards: list[CompressedDenseShardManifest] | None = None,
) -> dict[str, object]:
    effective_shards = shards or [CompressedDenseShardManifest(**_shard_data(profile))]
    listed_bytes = (
        2 * 8
        + 64
        + 128
        + sum(
            shard.embeddings_size_bytes + shard.candidate_index_size_bytes
            for shard in effective_shards
        )
    )
    return {
        "model_name": "BAAI/bge-base-en-v1.5",
        "model_revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "embedding_dimension": profile.embedding_dimension,
        "corpus_manifest_hash": "3" * 64,
        "dense_semantics_hash": "4" * 64,
        "chunks_sha256": "5" * 64,
        "chunk_count": 2,
        "artifact_profile": profile,
        "artifact_profile_hash": profile.manifest_hash(),
        "chunk_offsets_path": "chunks.offsets.u64",
        "chunk_offsets_format": "shared-chunk-offsets-v1",
        "chunk_offsets_size_bytes": 2 * 8,
        "chunk_offsets_sha256": "6" * 64,
        "candidate_template_path": "candidate-template.faiss",
        "candidate_template_size_bytes": 64,
        "candidate_template_sha256": "9" * 64,
        "shard_size_limit": profile.shard_size_limit,
        "shards": effective_shards,
        "dtype": "float16",
        "metric": "inner_product",
        "candidate_index": profile.candidate_index,
        "recall_report_path": "recall-report.json",
        "recall_report_size_bytes": 128,
        "recall_report_sha256": "7" * 64,
        "recall_at_300": 0.99,
        "recall_queries_sha256": "8" * 64,
        "recall_query_count": 100,
        "total_artifact_bytes": listed_bytes,
    }


@pytest.mark.parametrize("path", ["../shard", "/absolute", "nested/shard", "."])
def test_compressed_shard_rejects_unsafe_paths(path: str) -> None:
    values = _shard_data(_profile()) | {"path": path}

    with pytest.raises(ValidationError, match="safe relative directory"):
        CompressedDenseShardManifest(**values)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"artifact_profile_hash": "0" * 64}, "profile"),
        ({"candidate_template_sha256": "0" * 64}, "template"),
        ({"embeddings_size_bytes": 1}, "embedding"),
    ],
)
def test_compressed_manifest_rejects_incompatible_shards(
    updates: Mapping[str, object],
    message: str,
) -> None:
    profile = _profile()
    shard = CompressedDenseShardManifest(**(_shard_data(profile) | dict(updates)))
    values = _manifest_data(profile, shards=[shard])

    with pytest.raises(ValidationError, match=message):
        CompressedShardedDenseIndexManifest(**values)


def test_compressed_shard_rejects_a_range_that_differs_from_its_count() -> None:
    profile = _profile()

    with pytest.raises(ValidationError, match="range"):
        CompressedDenseShardManifest(
            **(_shard_data(profile) | {"end_row": 3})
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"artifact_profile_hash": "0" * 64}, "profile"),
        ({"candidate_index": "faiss-hnsw-flat-v1"}, "candidate"),
        ({"chunk_offsets_size_bytes": 8}, "offset"),
        ({"chunk_offsets_path": "../offsets"}, "safe relative file"),
    ],
)
def test_compressed_manifest_rejects_incompatible_top_level_values(
    updates: Mapping[str, object],
    message: str,
) -> None:
    profile = _profile()

    with pytest.raises(ValidationError, match=message):
        CompressedShardedDenseIndexManifest(
            **(_manifest_data(profile) | dict(updates))
        )


def test_compressed_manifest_requires_consecutive_complete_shard_ranges() -> None:
    profile = _profile()
    first = CompressedDenseShardManifest(**(_shard_data(profile) | {"end_row": 1, "chunk_count": 1, "embeddings_size_bytes": 16}))
    second = CompressedDenseShardManifest(
        **(
            _shard_data(profile)
            | {
                "shard_id": 1,
                "path": "shard-00001",
                "start_row": 2,
                "end_row": 3,
                "chunk_count": 1,
                "embeddings_size_bytes": 16,
            }
        )
    )

    with pytest.raises(ValidationError, match="consecutive"):
        CompressedShardedDenseIndexManifest(
            **(_manifest_data(profile, shards=[first, second]) | {"chunk_count": 3, "chunk_offsets_size_bytes": 24})
        )


def test_latency_report_rejects_manifest_or_query_identity_mismatches() -> None:
    manifest = CompressedShardedDenseIndexManifest(**_manifest_data(_profile()))
    verification = DenseFaissVerification(
        dense_manifest_hash=manifest.manifest_hash(),
        faiss_version=manifest.artifact_profile.faiss_version,
        template_ntotal=0,
        candidate_shard_count=len(manifest.shards),
        candidate_row_count=manifest.chunk_count,
    )
    report = DenseLatencyReport(
        dense_manifest_hash=manifest.manifest_hash(),
        faiss_verification=verification,
        recall_queries_sha256=manifest.recall_queries_sha256,
        query_count=100,
        warm_search_seconds=[0.5] * 100,
        max_observed_seconds=0.5,
        maximum_allowed_seconds=5.0,
        platform="macOS",
        machine="arm64",
        runtime_environment="test-worker",
    )

    assert report.require_compatible(manifest) is report
    assert report.require_runtime_compatible(
        manifest,
        runtime_environment="test-worker",
        platform="macOS",
        machine="arm64",
    ) is report
    with pytest.raises(ValueError, match="manifest"):
        report.model_copy(update={"dense_manifest_hash": "0" * 64}).require_compatible(
            manifest
        )
    with pytest.raises(ValueError, match="quer"):
        report.model_copy(update={"recall_queries_sha256": "0" * 64}).require_compatible(
            manifest
        )
    with pytest.raises(ValueError, match="platform"):
        report.require_runtime_compatible(
            manifest,
            runtime_environment="test-worker",
            platform="Linux-cloud",
            machine="arm64",
        )
    with pytest.raises(ValueError, match="machine"):
        report.require_runtime_compatible(
            manifest,
            runtime_environment="test-worker",
            platform="macOS",
            machine="x86_64",
        )


def test_latency_report_rejects_slow_or_incomplete_certification() -> None:
    with pytest.raises(ValidationError, match="query count"):
        DenseLatencyReport(
            dense_manifest_hash="1" * 64,
            faiss_verification=DenseFaissVerification(
                dense_manifest_hash="1" * 64,
                faiss_version=_profile().faiss_version,
                template_ntotal=0,
                candidate_shard_count=1,
                candidate_row_count=2,
            ),
            recall_queries_sha256="2" * 64,
            query_count=2,
            warm_search_seconds=[0.5],
            max_observed_seconds=0.5,
            maximum_allowed_seconds=5.0,
            platform="macOS",
            machine="arm64",
            runtime_environment="test-worker",
        )

    with pytest.raises(ValidationError, match="maximum"):
        DenseLatencyReport(
            dense_manifest_hash="1" * 64,
            faiss_verification=DenseFaissVerification(
                dense_manifest_hash="1" * 64,
                faiss_version=_profile().faiss_version,
                template_ntotal=0,
                candidate_shard_count=1,
                candidate_row_count=1,
            ),
            recall_queries_sha256="2" * 64,
            query_count=1,
            warm_search_seconds=[5.01],
            max_observed_seconds=5.01,
            maximum_allowed_seconds=5.0,
            platform="macOS",
            machine="arm64",
            runtime_environment="test-worker",
        )
