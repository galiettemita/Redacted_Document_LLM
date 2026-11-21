from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from redacted_backend.corpus.bulk_export import (
    BulkExportCatalog,
    audit_historylab_bulk_export,
)
from redacted_backend.corpus.best_available import (
    build_best_available_chunks,
    load_best_available_source,
)
from redacted_backend.corpus.build_bm25 import (
    build_best_available_bm25,
    build_bm25_from_files,
)
from redacted_backend.corpus.build_dense import build_dense_from_files
from redacted_backend.corpus.build_dense_sharded import (
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
from redacted_backend.scientific_config import ScientificConfig


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

    best_dense = subparsers.add_parser("build-best-available-dense")
    best_dense.add_argument("--chunks", required=True)
    best_dense.add_argument("--corpus-manifest", required=True)
    best_dense.add_argument("--application-queries", required=True)
    best_dense.add_argument("--certification-queries", required=True)
    best_dense.add_argument("--output-directory", required=True)
    best_dense.add_argument("--output-manifest", required=True)
    best_dense.add_argument("--batch-size", type=int, default=64)
    best_dense.add_argument("--shard-size", type=int, default=1_000_000)
    return parser


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
