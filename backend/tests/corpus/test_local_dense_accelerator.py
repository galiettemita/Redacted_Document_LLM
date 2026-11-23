from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACCELERATOR = PROJECT_ROOT / "scripts" / "run-local-dense-accelerator.zsh"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def accelerator_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    project = tmp_path / "project"
    corpus = project / "var" / "corpus" / "best-available"
    dense = corpus / "dense"
    state = corpus / "dense-accelerator-state"
    dense.mkdir(parents=True)
    (project / "backend").mkdir()
    (corpus / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    (corpus / "corpus-manifest.json").write_text("{}\n", encoding="utf-8")
    command_log = tmp_path / "commands.log"
    fake_uv = tmp_path / "fake-uv"
    _write_executable(
        fake_uv,
        """#!/bin/zsh
set -eu
print -r -- "$*" >> "$DENSE_ACCELERATOR_TEST_COMMAND_LOG"
print -r -- '{"schema_version":"dense-embedding-acceleration-v1","shard_count":213,"encoded_shard_ids":[212],"reused_shard_ids":[],"forward_frontier_shard":0,"stopped_for_safety_gap":false}'
""",
    )
    fake_caffeinate = tmp_path / "fake-caffeinate"
    _write_executable(
        fake_caffeinate,
        """#!/bin/zsh
set -eu
if [[ "$1" == "-i" ]]; then shift; fi
exec "$@"
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "DENSE_ACCELERATOR_PROJECT_ROOT": str(project),
            "DENSE_ACCELERATOR_CORPUS_ROOT": str(project / "var" / "corpus"),
            "DENSE_ACCELERATOR_UV": str(fake_uv),
            "DENSE_ACCELERATOR_CAFFEINATE": str(fake_caffeinate),
            "DENSE_ACCELERATOR_TEST_MODE": "1",
            "DENSE_ACCELERATOR_TEST_COMMAND_LOG": str(command_log),
            "DENSE_ACCELERATOR_SHARD_SIZE": "68572",
        }
    )
    return environment, state, command_log


def _run(environment: dict[str, str], command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/zsh", str(ACCELERATOR), command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )


def test_accelerator_starts_once_and_publishes_terminal_report(
    accelerator_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, state, command_log = accelerator_environment

    started = _run(environment, "start")

    assert started.returncode == 0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if (state / "state.json").exists():
            current = json.loads((state / "state.json").read_text(encoding="utf-8"))
            if current["phase"] == "completed":
                break
        time.sleep(0.05)
    else:
        raise AssertionError("accelerator did not complete")
    command = command_log.read_text(encoding="utf-8")
    assert command.count("accelerate-best-available-dense") == 1
    assert "--encoder-threads 4" in command
    assert "--safety-gap-shards 4" in command
    report = json.loads((state / "report.json").read_text(encoding="utf-8"))
    assert report["encoded_shard_ids"] == [212]


def test_accelerator_status_never_signals_a_foreign_pid(
    accelerator_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, state, _command_log = accelerator_environment
    state.mkdir(parents=True)
    (state / "state.json").write_text(
        json.dumps({"schema_version": "local-dense-accelerator-v1", "phase": "running"}),
        encoding="utf-8",
    )
    (state / "runner.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    status = _run(environment, "status")

    assert status.returncode != 0
    payload = json.loads(status.stdout)
    assert payload["process_status"] == "foreign_pid"
    os.kill(os.getpid(), 0)
