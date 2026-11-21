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
from redacted_backend.corpus.historylab_export import CatalogEntry
from redacted_backend.corpus.preflight import (
    PreflightReport,
    PreflightStatus,
    VolumeReport,
)
from redacted_backend.corpus.snapshot import SnapshotFile, SnapshotLock
from redacted_backend.retrieval.types import Bm25IndexManifest

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
