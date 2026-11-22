from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.corpus.bulk_export import (
    BulkExportCatalog,
    audit_historylab_bulk_export,
)
from redacted_backend.corpus.best_available import (
    BestAvailableCorpusManifest,
    build_best_available_chunks,
    load_best_available_source,
)
from redacted_backend.corpus.build_bm25 import (
    build_best_available_bm25,
    build_bm25_from_files,
)
from redacted_backend.corpus.build_dense import build_dense_from_files
from redacted_backend.corpus.build_dense_sharded import (
    benchmark_sharded_dense_build,
    build_sharded_dense_from_files,
)
from redacted_backend.corpus.exclusions import load_exclusion_manifest
from redacted_backend.corpus.frus_metadata import (
    FrusPublicationManifest,
    load_frus_metadata,
)
from redacted_backend.corpus.historylab_export import (
    ApiCaptureManifest,
    CorpusCoverageReport,
    capture_historylab_corpus,
    validate_capture_for_paper,
)
from redacted_backend.corpus.manifest import build_chunk_file
from redacted_backend.corpus.preflight import run_preflight
from redacted_backend.corpus.snapshot import SnapshotLock, resolve_snapshot
from redacted_backend.retrieval.dense import BgeEncoder
from redacted_backend.retrieval.dense_profile import (
    DenseArtifactProfile,
    dense_semantics_hash,
)
from redacted_backend.retrieval.dense_sharded import (
    ShardedDenseIndex,
    certify_dense_latency,
    verify_compressed_dense_artifacts,
)
from redacted_backend.retrieval.faiss_ivfpq import FaissIvfPqIndex
from redacted_backend.retrieval.types import (
    CompressedShardedDenseIndexManifest,
    DenseLatencyReport,
)
from redacted_backend.scientific_config import ScientificConfig


class DenseVerificationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["compressed-dense-verification-v1"] = (
        "compressed-dense-verification-v1"
    )
    verified: Literal[True] = True
    dense_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_count: int = Field(ge=1)
    shard_count: int = Field(ge=1)
    recall_query_count: int = Field(ge=1)
    recall_at_300: float = Field(ge=0, le=1)
    max_warm_search_seconds: float = Field(gt=0)
    total_artifact_bytes: int = Field(ge=1)


def run(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "preflight":
        report = run_preflight([Path(path) for path in args.path])
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2))
        return 2 if args.enforce_minimum and not report.sufficient else 0
    if args.command == "lock-snapshot":
        lock = resolve_snapshot(revision=args.revision, repo_id=args.repo_id)
        _write_json(Path(args.output), lock.model_dump(mode="json"))
        print(json.dumps(lock.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    if args.command == "audit-bulk-export":
        snapshot = SnapshotLock.model_validate_json(
            Path(args.snapshot_lock).read_text(encoding="utf-8")
        )
        catalog = BulkExportCatalog.model_validate_json(
            Path(args.catalog).read_text(encoding="utf-8")
        )
        exclusions = load_exclusion_manifest(Path(args.exclusions))
        report = audit_historylab_bulk_export(
            Path(args.source),
            snapshot=snapshot,
            catalog=catalog,
            exclusions=exclusions,
        )
        payload = report.model_dump(mode="json")
        _write_json(Path(args.output), payload)
        print(json.dumps(payload, sort_keys=True, indent=2))
        return 0 if report.ready_for_import else 2
    if args.command == "capture-api":
        manifest = capture_historylab_corpus(
            args.base_url,
            Path(args.output_root),
            page_size=args.page_size,
            corpora=args.corpus,
            resume=not args.no_resume,
        )
        print(json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    if args.command == "freeze-frus":
        manifest = load_frus_metadata(
            Path(args.input),
            source_url=args.source_url,
            retrieved_at=args.retrieved_at,
        )
        _write_json(Path(args.output), manifest.model_dump(mode="json"))
        print(json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    if args.command == "validate-capture":
        capture = ApiCaptureManifest.model_validate_json(
            Path(args.capture_manifest).read_text(encoding="utf-8")
        )
        exclusions = load_exclusion_manifest(Path(args.exclusions))
        frus = FrusPublicationManifest.model_validate_json(
            Path(args.frus_manifest).read_text(encoding="utf-8")
        )
        report = validate_capture_for_paper(
            capture,
            Path(args.capture_root),
            exclusions,
            frus,
        )
        _write_json(Path(args.output), report.model_dump(mode="json"))
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0 if report.ready else 2
    if args.command == "build-chunks":
        capture = ApiCaptureManifest.model_validate_json(
            Path(args.capture_manifest).read_text(encoding="utf-8")
        )
        coverage = CorpusCoverageReport.model_validate_json(
            Path(args.coverage_report).read_text(encoding="utf-8")
        )
        exclusions = load_exclusion_manifest(Path(args.exclusions))
        manifest = build_chunk_file(
            Path(args.capture_root),
            Path(args.output),
            capture=capture,
            coverage=coverage,
            exclusions=exclusions,
            scientific=ScientificConfig(),
        )
        _write_json(Path(args.manifest), manifest.model_dump(mode="json"))
        print(json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    if args.command == "build-best-available-chunks":
        profile = load_best_available_source(
            snapshot_path=Path(args.snapshot_lock),
            catalog_path=Path(args.catalog),
            audit_path=Path(args.audit),
            limitations_path=Path(args.limitations),
        )
        manifest = build_best_available_chunks(
            Path(args.source),
            Path(args.output),
            Path(args.manifest),
            Path(args.checkpoint),
            profile=profile,
            scientific=ScientificConfig(),
            checkpoint_interval_rows=args.checkpoint_interval_rows,
        )
        print(json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    if args.command == "build-bm25":
        report = build_bm25_from_files(
            Path(args.chunks),
            Path(args.corpus_manifest),
            Path(args.output_manifest),
            opensearch_url=args.opensearch_url,
        )
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    if args.command == "build-best-available-bm25":
        report = build_best_available_bm25(
            Path(args.chunks),
            Path(args.corpus_manifest),
            Path(args.output_manifest),
            Path(args.checkpoint),
            opensearch_url=args.opensearch_url,
            batch_size=args.batch_size,
        )
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    if args.command == "build-dense":
        recall_queries = [
            line.strip()
            for line in Path(args.recall_queries).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report = build_dense_from_files(
            Path(args.chunks),
            Path(args.corpus_manifest),
            Path(args.output_directory),
            Path(args.output_manifest),
            recall_queries=recall_queries,
            batch_size=args.batch_size,
        )
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    if args.command == "benchmark-best-available-dense":
        report = benchmark_sharded_dense_build(
            Path(args.chunks),
            Path(args.corpus_manifest),
            Path(args.output_directory),
            sample_rows=100_000,
            batch_size=args.batch_size,
        )
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    if args.command == "build-best-available-dense":
        report = build_sharded_dense_from_files(
            Path(args.chunks),
            Path(args.corpus_manifest),
            Path(args.output_directory),
            Path(args.output_manifest),
            application_queries_path=Path(args.application_queries),
            certification_queries_path=Path(args.certification_queries),
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            recall_block_rows=args.recall_block_rows,
        )
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    if args.command == "certify-best-available-dense-latency":
        runtime = _load_compressed_dense_runtime(
            Path(args.chunks),
            Path(args.dense_directory),
            Path(args.dense_manifest),
        )
        try:
            report = certify_dense_latency(
                runtime,
                _read_nonempty_lines(Path(args.certification_queries)),
                Path(args.output),
            )
        finally:
            runtime.close()
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    if args.command == "verify-best-available-dense":
        report = verify_best_available_dense(
            Path(args.chunks),
            Path(args.corpus_manifest),
            Path(args.dense_directory),
            Path(args.dense_manifest),
            Path(args.certification_queries),
            Path(args.latency_report),
        )
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    parser.error("A corpus command is required")
    return 2


def main() -> None:
    raise SystemExit(run())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="corpus")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--path", action="append", required=True)
    preflight.add_argument("--enforce-minimum", action="store_true")

    snapshot = subparsers.add_parser("lock-snapshot")
    snapshot.add_argument("--repo-id", default="HistoryLab/foiarchive")
    snapshot.add_argument("--revision", required=True)
    snapshot.add_argument("--output", required=True)

    audit = subparsers.add_parser("audit-bulk-export")
    audit.add_argument("--source", required=True)
    audit.add_argument("--snapshot-lock", required=True)
    audit.add_argument("--catalog", required=True)
    audit.add_argument("--exclusions", required=True)
    audit.add_argument("--output", required=True)

    capture = subparsers.add_parser("capture-api")
    capture.add_argument("--base-url", default="https://api.foiarchive.org")
    capture.add_argument("--output-root", required=True)
    capture.add_argument("--page-size", type=int, default=1_000)
    capture.add_argument("--corpus", action="append")
    capture.add_argument("--no-resume", action="store_true")

    frus = subparsers.add_parser("freeze-frus")
    frus.add_argument("--input", required=True)
    frus.add_argument("--output", required=True)
    frus.add_argument(
        "--source-url",
        default="https://history.state.gov/open/frus-metadata.xml",
    )
    frus.add_argument("--retrieved-at", required=True)

    coverage = subparsers.add_parser("validate-capture")
    coverage.add_argument("--capture-root", required=True)
    coverage.add_argument("--capture-manifest", required=True)
    coverage.add_argument("--exclusions", required=True)
    coverage.add_argument("--frus-manifest", required=True)
    coverage.add_argument("--output", required=True)

    chunks = subparsers.add_parser("build-chunks")
    chunks.add_argument("--capture-root", required=True)
    chunks.add_argument("--capture-manifest", required=True)
    chunks.add_argument("--coverage-report", required=True)
    chunks.add_argument("--exclusions", required=True)
    chunks.add_argument("--output", required=True)
    chunks.add_argument("--manifest", required=True)

    best_chunks = subparsers.add_parser("build-best-available-chunks")
    best_chunks.add_argument("--source", required=True)
    best_chunks.add_argument("--snapshot-lock", required=True)
    best_chunks.add_argument("--catalog", required=True)
    best_chunks.add_argument("--audit", required=True)
    best_chunks.add_argument("--limitations", required=True)
    best_chunks.add_argument("--output", required=True)
    best_chunks.add_argument("--checkpoint", required=True)
    best_chunks.add_argument("--manifest", required=True)
    best_chunks.add_argument("--checkpoint-interval-rows", type=int, default=10_000)

    bm25 = subparsers.add_parser("build-bm25")
    bm25.add_argument("--chunks", required=True)
    bm25.add_argument("--corpus-manifest", required=True)
    bm25.add_argument("--output-manifest", required=True)
    bm25.add_argument("--opensearch-url", default="http://127.0.0.1:59200")

    best_bm25 = subparsers.add_parser("build-best-available-bm25")
    best_bm25.add_argument("--chunks", required=True)
    best_bm25.add_argument("--corpus-manifest", required=True)
    best_bm25.add_argument("--output-manifest", required=True)
    best_bm25.add_argument("--checkpoint", required=True)
    best_bm25.add_argument(
        "--opensearch-url",
        default="http://127.0.0.1:59200",
    )
    best_bm25.add_argument("--batch-size", type=int, default=None)

    dense = subparsers.add_parser("build-dense")
    dense.add_argument("--chunks", required=True)
    dense.add_argument("--corpus-manifest", required=True)
    dense.add_argument("--recall-queries", required=True)
    dense.add_argument("--output-directory", required=True)
    dense.add_argument("--output-manifest", required=True)
    dense.add_argument("--batch-size", type=int, default=64)

    benchmark_dense = subparsers.add_parser(
        "benchmark-best-available-dense",
        help="run the fixed 100,000-row compressed dense benchmark",
    )
    benchmark_dense.add_argument("--chunks", required=True)
    benchmark_dense.add_argument("--corpus-manifest", required=True)
    benchmark_dense.add_argument("--output-directory", required=True)
    benchmark_dense.add_argument("--batch-size", type=_positive_int, default=64)

    best_dense = subparsers.add_parser("build-best-available-dense")
    best_dense.add_argument("--chunks", required=True)
    best_dense.add_argument("--corpus-manifest", required=True)
    best_dense.add_argument("--application-queries", required=True)
    best_dense.add_argument("--certification-queries", required=True)
    best_dense.add_argument("--output-directory", required=True)
    best_dense.add_argument("--output-manifest", required=True)
    best_dense.add_argument("--batch-size", type=_positive_int, default=64)
    best_dense.add_argument(
        "--shard-size",
        type=lambda value: _bounded_int(value, maximum=1_000_000),
        default=1_000_000,
    )
    best_dense.add_argument(
        "--recall-block-rows",
        type=lambda value: _bounded_int(value, maximum=65_536),
        default=16_384,
    )

    latency = subparsers.add_parser(
        "certify-best-available-dense-latency",
        help="certify all frozen dense queries against the five-second profile gate",
    )
    latency.add_argument("--chunks", required=True)
    latency.add_argument("--dense-directory", required=True)
    latency.add_argument("--dense-manifest", required=True)
    latency.add_argument("--certification-queries", required=True)
    latency.add_argument("--output", required=True)

    verify = subparsers.add_parser(
        "verify-best-available-dense",
        help="verify compressed dense artifacts without provider calls",
    )
    verify.add_argument("--chunks", required=True)
    verify.add_argument("--corpus-manifest", required=True)
    verify.add_argument("--dense-directory", required=True)
    verify.add_argument("--dense-manifest", required=True)
    verify.add_argument("--certification-queries", required=True)
    verify.add_argument("--latency-report", required=True)
    return parser


def _positive_int(value: str) -> int:
    return _bounded_int(value, maximum=None)


def _bounded_int(value: str, *, maximum: int | None) -> int:
    parsed = int(value)
    if parsed < 1 or (maximum is not None and parsed > maximum):
        upper = f" and {maximum}" if maximum is not None else ""
        raise argparse.ArgumentTypeError(f"value must be between 1{upper}")
    return parsed


def _read_nonempty_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_compressed_dense_runtime(
    chunks_path: Path,
    dense_root: Path,
    dense_manifest_path: Path,
) -> ShardedDenseIndex:
    scientific = ScientificConfig()
    manifest = CompressedShardedDenseIndexManifest.model_validate_json(
        dense_manifest_path.read_text(encoding="utf-8")
    )
    return ShardedDenseIndex(
        root=dense_root,
        chunks_path=chunks_path,
        manifest=manifest,
        encoder=BgeEncoder(scientific=scientific),
        candidate_loader=FaissIvfPqIndex.load,
        scientific=scientific,
    )


def verify_best_available_dense(
    chunks_path: Path,
    corpus_manifest_path: Path,
    dense_root: Path,
    dense_manifest_path: Path,
    certification_queries_path: Path,
    latency_report_path: Path,
) -> DenseVerificationReport:
    scientific = ScientificConfig()
    corpus = BestAvailableCorpusManifest.model_validate_json(
        corpus_manifest_path.read_text(encoding="utf-8")
    )
    dense = CompressedShardedDenseIndexManifest.model_validate_json(
        dense_manifest_path.read_text(encoding="utf-8")
    )
    supported_profile = DenseArtifactProfile()
    if corpus.scientific_hash != scientific.manifest_hash():
        raise ValueError("Best-available corpus scientific manifest mismatch")
    if (
        dense.model_name != scientific.embedding_model
        or dense.model_revision != scientific.embedding_model_revision
        or dense.embedding_dimension != scientific.embedding_dimension
        or dense.dense_semantics_hash != dense_semantics_hash(scientific)
    ):
        raise ValueError("Compressed dense semantic manifest mismatch")
    if (
        dense.corpus_manifest_hash != corpus.manifest_hash()
        or dense.chunks_sha256 != corpus.chunks_sha256
        or dense.chunk_count != corpus.chunk_count
    ):
        raise ValueError("Compressed dense corpus manifest mismatch")
    if (
        dense.artifact_profile != supported_profile
        or dense.artifact_profile_hash != supported_profile.manifest_hash()
    ):
        raise ValueError("Unsupported compressed dense artifact profile")
    if dense.recall_query_count != 100:
        raise ValueError("Compressed dense recall query count mismatch")
    if dense.recall_at_300 < scientific.dense_recall_minimum:
        raise ValueError("Compressed dense recall is below the publication gate")

    query_bytes = certification_queries_path.read_bytes()
    query_count = len(query_bytes.decode("utf-8").splitlines())
    if (
        hashlib.sha256(query_bytes).hexdigest() != dense.recall_queries_sha256
        or query_count != dense.recall_query_count
    ):
        raise ValueError("Compressed dense recall query artifact mismatch")
    latency = DenseLatencyReport.model_validate_json(
        latency_report_path.read_text(encoding="utf-8")
    )
    latency.require_compatible(dense)
    verify_compressed_dense_artifacts(dense_root, chunks_path, dense)
    return DenseVerificationReport(
        dense_manifest_hash=dense.manifest_hash(),
        corpus_manifest_hash=corpus.manifest_hash(),
        chunk_count=dense.chunk_count,
        shard_count=len(dense.shards),
        recall_query_count=dense.recall_query_count,
        recall_at_300=dense.recall_at_300,
        max_warm_search_seconds=latency.max_observed_seconds,
        total_artifact_bytes=dense.total_artifact_bytes,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            json.dump(value, destination, sort_keys=True, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
