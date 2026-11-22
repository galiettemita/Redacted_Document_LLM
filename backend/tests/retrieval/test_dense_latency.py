from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from redacted_backend.retrieval.dense_profile import DenseArtifactProfile
from redacted_backend.retrieval.dense_sharded import (
    DenseLatencyError,
    certify_dense_latency,
)
from redacted_backend.retrieval.types import (
    CompressedDenseShardManifest,
    CompressedShardedDenseIndexManifest,
    DenseFaissVerification,
    DenseLatencyReport,
)


def _manifest(queries_sha256: str) -> CompressedShardedDenseIndexManifest:
    profile = DenseArtifactProfile(
        embedding_dimension=2,
        ivf_nlist=1,
        pq_subvectors=1,
        pq_bits=1,
        nprobe=1,
        candidate_count_per_shard=1,
        training_sample_count=1,
        shard_size_limit=1,
        max_artifact_bytes=1024 * 1024,
    )
    shard = CompressedDenseShardManifest(
        artifact_profile_hash=profile.manifest_hash(),
        candidate_template_sha256="9" * 64,
        shard_id=0,
        path="shard-00000",
        start_row=0,
        end_row=1,
        chunk_count=1,
        embeddings_size_bytes=4,
        embeddings_sha256="1" * 64,
        candidate_index_size_bytes=8,
        candidate_index_sha256="2" * 64,
    )
    return CompressedShardedDenseIndexManifest(
        model_name="BAAI/bge-base-en-v1.5",
        model_revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        embedding_dimension=2,
        corpus_manifest_hash="3" * 64,
        dense_semantics_hash="4" * 64,
        chunks_sha256="5" * 64,
        chunk_count=1,
        artifact_profile=profile,
        artifact_profile_hash=profile.manifest_hash(),
        chunk_offsets_path="chunks.offsets.u64",
        chunk_offsets_format=profile.metadata_lookup_version,
        chunk_offsets_size_bytes=8,
        chunk_offsets_sha256="6" * 64,
        candidate_template_path="candidate-template.faiss",
        candidate_template_size_bytes=8,
        candidate_template_sha256="9" * 64,
        shard_size_limit=1,
        shards=[shard],
        dtype="float16",
        metric="inner_product",
        candidate_index=profile.candidate_index,
        recall_report_path="recall-report.json",
        recall_report_size_bytes=8,
        recall_report_sha256="7" * 64,
        recall_at_300=1.0,
        recall_queries_sha256=queries_sha256,
        recall_query_count=100,
        total_artifact_bytes=36,
    )


class _Runtime:
    def __init__(self, manifest: CompressedShardedDenseIndexManifest) -> None:
        self.manifest = manifest
        self.calls: list[str] = []
        self.faiss_verification = DenseFaissVerification(
            dense_manifest_hash=manifest.manifest_hash(),
            faiss_version=manifest.artifact_profile.faiss_version,
            template_ntotal=0,
            candidate_shard_count=len(manifest.shards),
            candidate_row_count=manifest.chunk_count,
        )

    def search(self, query: str):
        self.calls.append(query)
        return [object()]


class _AlternatingClock:
    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.calls = 0

    def __call__(self) -> float:
        value = 0.0 if self.calls % 2 == 0 else self.duration
        self.calls += 1
        return value


def _queries() -> tuple[list[str], str]:
    queries = [f"certification query {index}" for index in range(100)]
    value = ("\n".join(queries) + "\n").encode("utf-8")
    return queries, hashlib.sha256(value).hexdigest()


def test_latency_certification_warms_once_and_publishes_all_timings(
    tmp_path: Path,
) -> None:
    queries, queries_sha256 = _queries()
    runtime = _Runtime(_manifest(queries_sha256))
    output = tmp_path / "latency.json"

    report = certify_dense_latency(
        runtime,
        queries,
        output,
        clock=_AlternatingClock(4.99),
        platform_name=lambda: "macOS-test",
        machine_name=lambda: "arm64-test",
        runtime_environment=lambda: "serving-worker-container",
    )

    assert len(runtime.calls) == 101
    assert runtime.calls[0] == queries[0]
    assert report.query_count == 100
    assert report.warm_search_seconds == [4.99] * 100
    assert report.max_observed_seconds == 4.99
    assert report.dense_manifest_hash == runtime.manifest.manifest_hash()
    assert report.recall_queries_sha256 == queries_sha256
    assert report.runtime_environment == "serving-worker-container"
    assert report.faiss_verification == runtime.faiss_verification
    assert DenseLatencyReport.model_validate_json(
        output.read_text(encoding="utf-8")
    ) == report


def test_latency_certification_refuses_a_slow_query_without_publishing(
    tmp_path: Path,
) -> None:
    queries, queries_sha256 = _queries()
    runtime = _Runtime(_manifest(queries_sha256))
    output = tmp_path / "latency.json"
    output.write_text("stale passing report", encoding="utf-8")

    with pytest.raises(DenseLatencyError, match="5.01.*5.0"):
        certify_dense_latency(
            runtime,
            queries,
            output,
            clock=_AlternatingClock(5.01),
        )

    assert not output.exists()


def test_latency_certification_refuses_incomplete_dense_results(
    tmp_path: Path,
) -> None:
    queries, queries_sha256 = _queries()
    runtime = _Runtime(_manifest(queries_sha256))
    runtime.search = lambda _query: []  # type: ignore[method-assign]
    output = tmp_path / "latency.json"

    with pytest.raises(DenseLatencyError, match="complete.*result"):
        certify_dense_latency(runtime, queries, output)

    assert not output.exists()


def test_latency_certification_requires_faiss_deserialization_evidence(
    tmp_path: Path,
) -> None:
    queries, queries_sha256 = _queries()
    runtime = _Runtime(_manifest(queries_sha256))
    del runtime.faiss_verification
    output = tmp_path / "latency.json"

    with pytest.raises(DenseLatencyError, match="FAISS.*verification"):
        certify_dense_latency(runtime, queries, output)

    assert not output.exists()


def test_latency_certification_invalidates_stale_report_before_query_checks(
    tmp_path: Path,
) -> None:
    queries, queries_sha256 = _queries()
    runtime = _Runtime(_manifest(queries_sha256))
    output = tmp_path / "latency.json"
    output.write_text("stale passing report", encoding="utf-8")

    with pytest.raises(DenseLatencyError, match="query count"):
        certify_dense_latency(runtime, queries[:-1], output)

    assert not output.exists()


def test_latency_report_rejects_nonfinite_measurements() -> None:
    queries, queries_sha256 = _queries()
    with pytest.raises(ValueError, match="finite"):
        DenseLatencyReport(
            dense_manifest_hash=_manifest(queries_sha256).manifest_hash(),
            faiss_verification=DenseFaissVerification(
                dense_manifest_hash=_manifest(queries_sha256).manifest_hash(),
                faiss_version=_manifest(queries_sha256).artifact_profile.faiss_version,
                template_ntotal=0,
                candidate_shard_count=1,
                candidate_row_count=1,
            ),
            recall_queries_sha256=queries_sha256,
            query_count=100,
            warm_search_seconds=[1.0] * 99 + [float("nan")],
            max_observed_seconds=1.0,
            maximum_allowed_seconds=5.0,
            platform="test",
            machine="test",
            runtime_environment="serving-worker-container",
        )
