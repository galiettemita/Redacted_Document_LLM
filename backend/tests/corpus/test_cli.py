from __future__ import annotations

from redacted_backend.corpus import cli
from redacted_backend.corpus.preflight import (
    PreflightReport,
    PreflightStatus,
    VolumeReport,
)


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
