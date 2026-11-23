from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from redacted_backend.corpus import cli
from redacted_backend.corpus.bulk_export import (
    BulkExportAudit,
    BulkExportAuditError,
    BulkExportCatalog,
)
from redacted_backend.corpus.best_available import (
    BestAvailableCorpusManifest,
    BestAvailableSourceProfile,
)
from redacted_backend.corpus.build_bm25 import Bm25BuildReport
from redacted_backend.corpus.build_dense_sharded import (
    BenchmarkEmbeddingCheckpoint,
    DenseBuildBenchmarkReport,
    ShardedDenseBuildReport,
)
from redacted_backend.corpus.historylab_export import CatalogEntry
from redacted_backend.corpus.preflight import (
    PreflightReport,
    PreflightStatus,
    VolumeReport,
)
from redacted_backend.corpus.snapshot import SnapshotFile, SnapshotLock
from redacted_backend.retrieval.dense_profile import DenseArtifactProfile
from redacted_backend.retrieval.types import (
    Bm25IndexManifest,
    CompressedDenseShardManifest,
    CompressedShardedDenseIndexManifest,
    DenseFaissVerification,
    DenseLatencyReport,
)

REVISION = "411cf6f86b38342683e27148be167c6094d1a10c"


def _report(status: PreflightStatus) -> PreflightReport:
    sufficient = status is not PreflightStatus.INSUFFICIENT
    return PreflightReport(
        status=status,
        sufficient=sufficient,
        recommended=status is PreflightStatus.RECOMMENDED,
        minimum_free_gib=150,
        recommended_free_gib=250,
        volumes=[
            VolumeReport(
                path="/corpus",
                total_bytes=500 * 1024**3,
                free_bytes=(100 if not sufficient else 300) * 1024**3,
            )
        ],
        ram_bytes=32 * 1024**3,
        cpu_count=12,
        accelerator="none",
    )


def test_preflight_reports_insufficient_capacity_without_blocking_by_default(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_preflight",
        lambda _paths: _report(PreflightStatus.INSUFFICIENT),
    )

    exit_code = cli.run(["preflight", "--path", "/corpus"])

    assert exit_code == 0
    assert '"status": "insufficient"' in capsys.readouterr().out


def test_preflight_can_explicitly_enforce_the_reported_minimum(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_preflight",
        lambda _paths: _report(PreflightStatus.INSUFFICIENT),
    )

    exit_code = cli.run(
        ["preflight", "--path", "/corpus", "--enforce-minimum"]
    )

    assert exit_code == 2


def test_preflight_enforcement_accepts_recommended_capacity(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_preflight",
        lambda _paths: _report(PreflightStatus.RECOMMENDED),
    )

    exit_code = cli.run(
        ["preflight", "--path", "/corpus", "--enforce-minimum"]
    )

    assert exit_code == 0


def _audit_inputs(
    tmp_path: Path,
    *,
    row: dict[str, object] | None = None,
    raw_row: bytes | None = None,
    catalog_counts: dict[str, int],
) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source.json.gz"
    with gzip.open(source, "wb") as target:
        target.write(
            raw_row
            if raw_row is not None
            else json.dumps(row, sort_keys=True).encode("utf-8") + b"\n"
        )
    item = SnapshotFile(
        path=source.name,
        size_bytes=source.stat().st_size,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    snapshot = SnapshotLock(
        repo_id="HistoryLab/foiarchive",
        revision=REVISION,
        license="cc-by-nc-4.0",
        files=[item],
    )
    catalog = BulkExportCatalog(
        schema_version="historylab-huggingface-catalog-v1",
        repo_id="HistoryLab/foiarchive",
        revision=REVISION,
        release_url="https://lab.history.columbia.edu/news/foiarchive-hugging-face",
        source=item,
        document_count=sum(catalog_counts.values()),
        collections=[
            CatalogEntry(corpus=corpus, doc_cnt=count)
            for corpus, count in sorted(catalog_counts.items())
        ],
    )
    snapshot_path = tmp_path / "snapshot.json"
    catalog_path = tmp_path / "catalog.json"
    snapshot_path.write_text(snapshot.model_dump_json(), encoding="utf-8")
    catalog_path.write_text(catalog.model_dump_json(), encoding="utf-8")
    exclusions_path = (
        Path(__file__).parents[2] / "corpus/exclusions/v1.yaml"
    )
    return source, snapshot_path, catalog_path, exclusions_path


def _document_row(source: str | None) -> dict[str, object]:
    return {
        "doc_id": "doc-1",
        "classification": "SECRET",
        "authored": "1974-01-01",
        "title": "A title",
        "body": "Document body",
        "source": source,
        "pg_cnt": 1,
        "char_cnt": 13,
        "word_cnt": 2,
    }


def _audit_command(
    *,
    source: Path,
    snapshot: Path,
    catalog: Path,
    exclusions: Path,
    output: Path,
) -> list[str]:
    return [
        "audit-bulk-export",
        "--source",
        str(source),
        "--snapshot-lock",
        str(snapshot),
        "--catalog",
        str(catalog),
        "--exclusions",
        str(exclusions),
        "--output",
        str(output),
    ]


def test_audit_bulk_export_writes_ready_report_atomically(
    tmp_path: Path,
    capsys,
) -> None:
    source, snapshot, catalog, exclusions = _audit_inputs(
        tmp_path,
        row=_document_row("cables"),
        catalog_counts={"cables": 1},
    )
    output = tmp_path / "reports/audit.json"

    exit_code = cli.run(
        _audit_command(
            source=source,
            snapshot=snapshot,
            catalog=catalog,
            exclusions=exclusions,
            output=output,
        )
    )

    report = BulkExportAudit.model_validate_json(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report.ready_for_import is True
    assert BulkExportAudit.model_validate_json(capsys.readouterr().out) == report
    assert not list(output.parent.glob(".audit.json.*"))


def test_audit_bulk_export_writes_unready_report_and_returns_two(
    tmp_path: Path,
) -> None:
    source, snapshot, catalog, exclusions = _audit_inputs(
        tmp_path,
        row=_document_row("https://history.state.gov/example"),
        catalog_counts={"cia": 1},
    )
    output = tmp_path / "audit.json"

    exit_code = cli.run(
        _audit_command(
            source=source,
            snapshot=snapshot,
            catalog=catalog,
            exclusions=exclusions,
            output=output,
        )
    )

    report = BulkExportAudit.model_validate_json(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report.ready_for_import is False
    assert "unresolved_collection_labels" in report.reasons
    assert "required_corpus_missing:cables" in report.reasons


def test_audit_bulk_export_does_not_publish_report_after_parse_failure(
    tmp_path: Path,
) -> None:
    source, snapshot, catalog, exclusions = _audit_inputs(
        tmp_path,
        raw_row=b"{not-json}\n",
        catalog_counts={"cables": 1},
    )
    output = tmp_path / "audit.json"

    with pytest.raises(BulkExportAuditError, match="invalid row"):
        cli.run(
            _audit_command(
                source=source,
                snapshot=snapshot,
                catalog=catalog,
                exclusions=exclusions,
                output=output,
            )
        )

    assert not output.exists()


def test_failed_audit_rerun_preserves_the_previous_atomic_report(
    tmp_path: Path,
) -> None:
    source, snapshot, catalog, exclusions = _audit_inputs(
        tmp_path,
        raw_row=b"{not-json}\n",
        catalog_counts={"cables": 1},
    )
    output = tmp_path / "audit.json"
    previous_report = b'{"previous_verified_report": true}\n'
    output.write_bytes(previous_report)

    with pytest.raises(BulkExportAuditError, match="invalid row"):
        cli.run(
            _audit_command(
                source=source,
                snapshot=snapshot,
                catalog=catalog,
                exclusions=exclusions,
                output=output,
            )
        )

    assert output.read_bytes() == previous_report


def test_build_best_available_chunks_cli_publishes_real_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.json.gz"
    with gzip.GzipFile(filename=str(source), mode="wb", mtime=0) as destination:
        destination.write(
            json.dumps(
                _document_row("https://example.test/document.pdf"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    profile = BestAvailableSourceProfile(
        source=SnapshotFile(
            path=source.name,
            size_bytes=source.stat().st_size,
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ),
        snapshot_manifest_hash="1" * 64,
        catalog_manifest_hash="2" * 64,
        audit_manifest_hash="3" * 64,
        limitations_sha256="4" * 64,
        catalog_document_count=1,
        parsed_row_count=1,
        unresolved_collection_count=1,
        unresolved_value_counts={
            "blank": 0,
            "http_url": 1,
            "null": 0,
            "unknown": 0,
        },
        canonical_collection_counts={},
        audit_reasons=["unresolved_collection_labels"],
    )
    monkeypatch.setattr(
        cli,
        "load_best_available_source",
        lambda **_kwargs: profile,
        raising=False,
    )
    chunks = tmp_path / "artifacts/chunks.jsonl"
    checkpoint = tmp_path / "artifacts/checkpoint.json"
    manifest = tmp_path / "artifacts/manifest.json"

    exit_code = cli.run(
        [
            "build-best-available-chunks",
            "--source",
            str(source),
            "--snapshot-lock",
            str(tmp_path / "snapshot.json"),
            "--catalog",
            str(tmp_path / "catalog.json"),
            "--audit",
            str(tmp_path / "audit.json"),
            "--limitations",
            str(tmp_path / "limitations.md"),
            "--output",
            str(chunks),
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--checkpoint-interval-rows",
            "1",
        ]
    )

    published = BestAvailableCorpusManifest.model_validate_json(
        manifest.read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert published.document_count == 1
    assert published.chunk_count == 1
    assert published.paper_coverage_ready is False
    assert BestAvailableCorpusManifest.model_validate_json(
        capsys.readouterr().out
    ) == published
    assert not checkpoint.exists()


def test_build_best_available_bm25_cli_reports_verified_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    index_manifest = Bm25IndexManifest(
        index_name="foiarchive-best-available-bm25-v1",
        retrieval_profile="best_available_foiarchive_v1",
        opensearch_version="3.9.0",
        corpus_manifest_hash="1" * 64,
        scientific_hash="2" * 64,
        index_definition_hash="3" * 64,
        stopwords_hash="4" * 64,
        document_count=10,
        chunk_count=12,
    )
    report = Bm25BuildReport(elapsed_seconds=1.5, index_manifest=index_manifest)
    monkeypatch.setattr(
        cli,
        "build_best_available_bm25",
        lambda *_args, **_kwargs: report,
        raising=False,
    )

    exit_code = cli.run(
        [
            "build-best-available-bm25",
            "--chunks",
            str(tmp_path / "chunks.jsonl"),
            "--corpus-manifest",
            str(tmp_path / "corpus.json"),
            "--output-manifest",
            str(tmp_path / "bm25.json"),
            "--checkpoint",
            str(tmp_path / "checkpoint.json"),
            "--opensearch-url",
            "http://127.0.0.1:59200",
            "--batch-size",
            "500",
        ]
    )

    assert exit_code == 0
    assert Bm25BuildReport.model_validate_json(capsys.readouterr().out) == report


def test_build_best_available_dense_cli_reports_verified_sharded_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _compressed_manifest_for_cli()
    report = ShardedDenseBuildReport(
        elapsed_seconds=2.5,
        corpus_manifest_hash=manifest.corpus_manifest_hash,
        index_manifest=manifest,
    )
    captured: dict[str, object] = {}

    def fake_builder(*args, **kwargs) -> ShardedDenseBuildReport:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return report

    monkeypatch.setattr(
        cli,
        "build_sharded_dense_from_files",
        fake_builder,
        raising=False,
    )

    exit_code = cli.run(
        [
            "build-best-available-dense",
            "--chunks",
            str(tmp_path / "chunks.jsonl"),
            "--corpus-manifest",
            str(tmp_path / "corpus.json"),
            "--application-queries",
            str(tmp_path / "application.txt"),
            "--certification-queries",
            str(tmp_path / "certification.txt"),
            "--output-directory",
            str(tmp_path / "dense"),
            "--output-manifest",
            str(tmp_path / "dense/manifest.json"),
            "--batch-size",
            "32",
            "--shard-size",
            "1000000",
            "--recall-block-rows",
            "8192",
        ]
    )

    assert exit_code == 0
    assert ShardedDenseBuildReport.model_validate_json(capsys.readouterr().out) == report
    assert captured["kwargs"] == {
        "application_queries_path": tmp_path / "application.txt",
        "certification_queries_path": tmp_path / "certification.txt",
        "batch_size": 32,
        "encoder_threads": 8,
        "shard_size": 1_000_000,
        "recall_block_rows": 8_192,
        "contract": cli.PRODUCTION_BEST_AVAILABLE_RETRIEVAL_CONTRACT,
    }


def test_benchmark_best_available_dense_cli_uses_exact_paths_and_100k_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = DenseBuildBenchmarkReport(
        sample_rows=100_000,
        production_chunk_count=14_549_269,
        elapsed_seconds=10.0,
        setup_seconds=1.0,
        encoding_seconds=5.0,
        training_seconds=1.0,
        candidate_add_seconds=1.0,
        certification_probe_seconds=2.0,
        certification_probe_queries=10,
        rows_per_second=10_000.0,
        peak_rss_bytes=123,
        peak_gpu_memory_bytes=456,
        embedding_bytes_per_row=768.0,
        candidate_bytes_per_row=100.0,
        candidate_fixed_bytes_per_shard=1024,
        projected_shard_count=15,
        offset_bytes_per_row=8.0,
        projected_full_artifact_bytes=24 * 1024**3,
        projected_full_seconds=1800.0,
        projected_gpu_hours=0.5,
    )
    captured: dict[str, object] = {}

    def fake_benchmark(*args, **kwargs) -> DenseBuildBenchmarkReport:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return report

    monkeypatch.setattr(
        cli,
        "benchmark_sharded_dense_build",
        fake_benchmark,
        raising=False,
    )
    chunks = tmp_path / "chunks.jsonl"
    corpus_manifest = tmp_path / "corpus.json"
    output = tmp_path / "benchmark"

    exit_code = cli.run(
        [
            "benchmark-best-available-dense",
            "--chunks",
            str(chunks),
            "--corpus-manifest",
            str(corpus_manifest),
            "--output-directory",
            str(output),
            "--batch-size",
            "32",
        ]
    )

    assert exit_code == 0
    assert DenseBuildBenchmarkReport.model_validate_json(
        capsys.readouterr().out
    ) == report
    assert captured == {
        "args": (chunks, corpus_manifest, output),
        "kwargs": {
            "sample_rows": 100_000,
            "batch_size": 32,
            "encoder_threads": 8,
            "contract": cli.PRODUCTION_BEST_AVAILABLE_RETRIEVAL_CONTRACT,
        },
    }


def test_recover_best_available_dense_benchmark_cli_binds_completed_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}
    checkpoint = BenchmarkEmbeddingCheckpoint(
        dense_semantics_hash="1" * 64,
        training_selection_hash="2" * 64,
        chunks_sha256="3" * 64,
        offsets_sha256="4" * 64,
        sample_rows=100_000,
        embedding_dimension=768,
        embedding_size_bytes=76_800_000,
        embedding_sha256="5" * 64,
        setup_seconds=275.0,
        encoding_seconds=10_500.0,
    )

    def fake_recover(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return checkpoint

    monkeypatch.setattr(
        cli,
        "recover_benchmark_embedding_checkpoint",
        fake_recover,
        raising=False,
    )
    chunks = tmp_path / "chunks.jsonl"
    corpus_manifest = tmp_path / "corpus.json"
    output = tmp_path / "benchmark"

    exit_code = cli.run(
        [
            "recover-best-available-dense-benchmark",
            "--chunks",
            str(chunks),
            "--corpus-manifest",
            str(corpus_manifest),
            "--output-directory",
            str(output),
            "--setup-seconds",
            "275",
            "--encoding-seconds",
            "10500",
        ]
    )

    assert exit_code == 0
    assert BenchmarkEmbeddingCheckpoint.model_validate_json(
        capsys.readouterr().out
    ) == checkpoint
    assert captured == {
        "args": (chunks, corpus_manifest, output),
        "kwargs": {
            "sample_rows": 100_000,
            "setup_seconds": 275.0,
            "encoding_seconds": 10_500.0,
            "contract": cli.PRODUCTION_BEST_AVAILABLE_RETRIEVAL_CONTRACT,
        },
    }


def test_certify_best_available_dense_latency_cli_uses_frozen_profile_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _compressed_manifest_for_cli()
    queries = [f"certification query {index}" for index in range(100)]
    queries_path = tmp_path / "recall-queries.txt"
    queries_path.write_text("\n".join(queries) + "\n", encoding="utf-8")
    output = tmp_path / "latency.json"
    report = DenseLatencyReport(
        dense_manifest_hash=manifest.manifest_hash(),
        faiss_verification=DenseFaissVerification(
            dense_manifest_hash=manifest.manifest_hash(),
            faiss_version=manifest.artifact_profile.faiss_version,
            template_ntotal=0,
            candidate_shard_count=len(manifest.shards),
            candidate_row_count=manifest.chunk_count,
        ),
        recall_queries_sha256=manifest.recall_queries_sha256,
        query_count=100,
        warm_search_seconds=[0.1] * 100,
        max_observed_seconds=0.1,
        maximum_allowed_seconds=5.0,
        platform="test",
        machine="test",
        runtime_environment="serving-worker-container",
        worker_build_id="a" * 64,
        serving_host_id="b" * 64,
    )
    captured: dict[str, object] = {}

    class Runtime:
        def close(self) -> None:
            captured["closed"] = True

    def fake_loader(*args, **kwargs):
        captured["load_args"] = args
        captured["load_kwargs"] = kwargs
        return Runtime()

    def fake_certifier(runtime, received_queries, output_path):
        captured["certify"] = (runtime, received_queries, output_path)
        return report

    monkeypatch.setattr(cli, "_load_compressed_dense_runtime", fake_loader, raising=False)
    monkeypatch.setattr(cli, "certify_dense_latency", fake_certifier, raising=False)
    chunks = tmp_path / "chunks.jsonl"
    dense_root = tmp_path / "dense"
    dense_manifest = dense_root / "manifest.json"

    exit_code = cli.run(
        [
            "certify-best-available-dense-latency",
            "--chunks",
            str(chunks),
            "--dense-directory",
            str(dense_root),
            "--dense-manifest",
            str(dense_manifest),
            "--certification-queries",
            str(queries_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert DenseLatencyReport.model_validate_json(capsys.readouterr().out) == report
    assert captured["load_args"] == (chunks, dense_root, dense_manifest)
    assert captured["load_kwargs"] == {}
    assert captured["certify"][1:] == (queries, output)
    assert captured["closed"] is True


def test_latency_cli_invalidates_stale_report_before_runtime_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "latency.json"
    output.write_text("stale passing report", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "_load_compressed_dense_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("load failed")),
    )

    with pytest.raises(RuntimeError, match="load failed"):
        cli.run(
            [
                "certify-best-available-dense-latency",
                "--chunks",
                str(tmp_path / "chunks.jsonl"),
                "--dense-directory",
                str(tmp_path / "dense"),
                "--dense-manifest",
                str(tmp_path / "dense/manifest.json"),
                "--certification-queries",
                str(tmp_path / "queries.txt"),
                "--output",
                str(output),
            ]
        )

    assert not output.exists()


def test_verify_best_available_dense_cli_is_local_and_provider_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = cli.DenseVerificationReport(
        dense_manifest_hash="1" * 64,
        corpus_manifest_hash="2" * 64,
        chunk_count=14_549_269,
        shard_count=15,
        recall_query_count=100,
        recall_at_300=0.99,
        max_warm_search_seconds=4.5,
        total_artifact_bytes=24 * 1024**3,
    )
    captured: dict[str, object] = {}

    def fake_verify(*args):
        captured["args"] = args
        return report

    monkeypatch.setattr(
        cli,
        "verify_best_available_dense",
        fake_verify,
        raising=False,
    )
    paths = [
        tmp_path / "chunks.jsonl",
        tmp_path / "corpus.json",
        tmp_path / "dense",
        tmp_path / "dense/manifest.json",
        tmp_path / "recall-queries.txt",
        tmp_path / "latency.json",
    ]

    exit_code = cli.run(
        [
            "verify-best-available-dense",
            "--chunks",
            str(paths[0]),
            "--corpus-manifest",
            str(paths[1]),
            "--dense-directory",
            str(paths[2]),
            "--dense-manifest",
            str(paths[3]),
            "--certification-queries",
            str(paths[4]),
            "--latency-report",
            str(paths[5]),
        ]
    )

    assert exit_code == 0
    assert cli.DenseVerificationReport.model_validate_json(
        capsys.readouterr().out
    ) == report
    assert captured["args"] == tuple(paths)


def test_production_dense_commands_do_not_expose_gate_or_profile_bypasses() -> None:
    parser = cli._parser()
    commands = next(
        action
        for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    ).choices
    build_options = {
        option
        for action in commands["build-best-available-dense"]._actions
        for option in action.option_strings
    }
    latency_options = {
        option
        for action in commands["certify-best-available-dense-latency"]._actions
        for option in action.option_strings
    }

    assert {"--batch-size", "--shard-size", "--recall-block-rows"} <= build_options
    assert {
        "--skip-checksums",
        "--skip-recall",
        "--skip-storage-gate",
        "--nprobe",
        "--ivf-nlist",
        "--pq-subvectors",
        "--candidate-count",
    }.isdisjoint(build_options)
    assert {"--maximum-seconds", "--latency-limit"}.isdisjoint(latency_options)


def _compressed_manifest_for_cli() -> CompressedShardedDenseIndexManifest:
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
        embeddings_size_bytes=2,
        embeddings_sha256="1" * 64,
        candidate_index_size_bytes=8,
        candidate_index_sha256="2" * 64,
    )
    return CompressedShardedDenseIndexManifest(
        model_name="BAAI/bge-base-en-v1.5",
        model_revision=REVISION,
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
        embedding_file="embeddings.i8",
        dtype="int8",
        metric="inner_product",
        candidate_index=profile.candidate_index,
        recall_report_path="recall-report.json",
        recall_report_size_bytes=8,
        recall_report_sha256="7" * 64,
        recall_at_300=1.0,
        recall_queries_sha256="8" * 64,
        recall_query_count=100,
        total_artifact_bytes=34,
    )
