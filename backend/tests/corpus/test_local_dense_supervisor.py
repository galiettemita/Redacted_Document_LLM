from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPERVISOR = PROJECT_ROOT / "scripts" / "supervise-local-dense-pipeline.zsh"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def supervisor_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    command_log = tmp_path / "commands.log"
    scripts: dict[str, Path] = {}
    for component in ("runner", "finalizer", "activator"):
        script = tmp_path / component
        _write_executable(
            script,
            """#!/bin/zsh
set -eu
component="${0:t}"
if [[ "$1" == "status" ]]; then
  print -r -- "$(<"$DENSE_SUPERVISOR_TEST_STATUS_DIR/$component.json")"
  exit 0
fi
if [[ "$1" == "start" ]]; then
  print -r -- "$component start" >> "$DENSE_SUPERVISOR_TEST_COMMAND_LOG"
  exit 0
fi
exit 64
""",
        )
        scripts[component] = script
    environment = os.environ.copy()
    environment.update(
        {
            "DENSE_SUPERVISOR_RUNNER": str(scripts["runner"]),
            "DENSE_SUPERVISOR_FINALIZER": str(scripts["finalizer"]),
            "DENSE_SUPERVISOR_ACTIVATOR": str(scripts["activator"]),
            "DENSE_SUPERVISOR_TEST_STATUS_DIR": str(tmp_path),
            "DENSE_SUPERVISOR_TEST_COMMAND_LOG": str(command_log),
        }
    )
    return environment, command_log


def _write_statuses(root: Path, **statuses: dict[str, object]) -> None:
    for component, status in statuses.items():
        (root / f"{component}.json").write_text(
            json.dumps(status),
            encoding="utf-8",
        )


def _run(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SUPERVISOR), "run"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )


def test_supervisor_restarts_only_resumable_stopped_components(
    supervisor_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = supervisor_environment
    root = Path(environment["DENSE_SUPERVISOR_TEST_STATUS_DIR"])
    _write_statuses(
        root,
        runner={
            "phase": "building",
            "runner_status": "stopped",
            "resumable": True,
        },
        finalizer={"phase": "waiting", "process_status": "stopped"},
        activator={"phase": "waiting", "process_status": "stopped"},
    )

    result = _run(environment)

    assert result.returncode == 0, result.stderr
    assert command_log.read_text(encoding="utf-8").splitlines() == [
        "runner start",
        "finalizer start",
        "activator start",
    ]
    report = json.loads(result.stdout)
    assert report["actions"] == [
        "runner_started",
        "finalizer_started",
        "activator_started",
    ]


def test_supervisor_noops_for_live_complete_and_terminal_components(
    supervisor_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = supervisor_environment
    root = Path(environment["DENSE_SUPERVISOR_TEST_STATUS_DIR"])
    _write_statuses(
        root,
        runner={
            "phase": "building",
            "runner_status": "running",
            "resumable": False,
        },
        finalizer={"phase": "failed", "process_status": "stopped"},
        activator={"phase": "blocked", "process_status": "stopped"},
    )

    result = _run(environment)

    assert result.returncode == 0, result.stderr
    assert not command_log.exists()
    assert json.loads(result.stdout)["actions"] == []


def test_supervisor_fails_closed_on_invalid_status_without_starting(
    supervisor_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = supervisor_environment
    root = Path(environment["DENSE_SUPERVISOR_TEST_STATUS_DIR"])
    (root / "runner.json").write_text("not json\n", encoding="utf-8")
    _write_statuses(
        root,
        finalizer={"phase": "waiting", "process_status": "stopped"},
        activator={"phase": "waiting", "process_status": "stopped"},
    )

    result = _run(environment)

    assert result.returncode != 0
    assert not command_log.exists()
    assert "runner status is invalid" in result.stderr


def test_supervisor_source_contains_no_destructive_or_forced_restart_commands() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")

    assert "kill" not in source
    assert "rm -" not in source
    assert "git clean" not in source
    assert "docker system prune" not in source
    assert "build_failed" not in source
    assert "failed" not in source
    assert "blocked" not in source
