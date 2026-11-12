from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from redacted_backend.corpus.exclusions import load_exclusion_manifest
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
        return 0 if report.sufficient else 2
    if args.command == "lock-snapshot":
        lock = resolve_snapshot(revision=args.revision, repo_id=args.repo_id)
        _write_json(Path(args.output), lock.model_dump(mode="json"))
        print(json.dumps(lock.model_dump(mode="json"), sort_keys=True, indent=2))
        return 0
    if args.command == "build-chunks":
        snapshot = SnapshotLock.model_validate_json(
            Path(args.snapshot).read_text(encoding="utf-8")
        )
        exclusions = load_exclusion_manifest(Path(args.exclusions))
        manifest = build_chunk_file(
            [Path(path) for path in args.input],
            Path(args.output),
            snapshot=snapshot,
            exclusions=exclusions,
            scientific=ScientificConfig(),
        )
        _write_json(Path(args.manifest), manifest.model_dump(mode="json"))
        print(json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2))
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

    snapshot = subparsers.add_parser("lock-snapshot")
    snapshot.add_argument("--repo-id", default="HistoryLab/foiarchive")
    snapshot.add_argument("--revision", required=True)
    snapshot.add_argument("--output", required=True)

    chunks = subparsers.add_parser("build-chunks")
    chunks.add_argument("--input", action="append", required=True)
    chunks.add_argument("--snapshot", required=True)
    chunks.add_argument("--exclusions", required=True)
    chunks.add_argument("--output", required=True)
    chunks.add_argument("--manifest", required=True)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
