from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINALIZER = PROJECT_ROOT / "scripts" / "finalize-local-dense-build.zsh"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def finalizer_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    project = tmp_path / "project"
    corpus = project / "var" / "corpus" / "best-available"
    dense = corpus / "dense"
    build_state = corpus / "dense-build-state"
    finalizer_state = corpus / "dense-finalizer-state"
    dense.mkdir(parents=True)
    build_state.mkdir(parents=True)
    (project / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    (corpus / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    (corpus / "corpus-manifest.json").write_text("{}\n", encoding="utf-8")
    (dense / "manifest.json").write_text("{}\n", encoding="utf-8")
    (dense / "recall-queries.txt").write_text("query\n", encoding="utf-8")
    (build_state / "state.json").write_text(
        json.dumps({"schema_version": "local-dense-runner-v1", "phase": "completed"}),
        encoding="utf-8",
    )
    command_log = tmp_path / "docker-commands.log"
    fake_docker = tmp_path / "fake-docker"
    _write_executable(
        fake_docker,
        r"""#!/bin/zsh
set -eu
print -r -- "$*" >> "$DENSE_FINALIZER_TEST_COMMAND_LOG"
if [[ "$*" == *"certify-best-available-dense-latency"* ]]; then
  if [[ "${DENSE_FINALIZER_TEST_LATENCY_FAIL:-0}" == "1" ]]; then
    exit 17
  fi
  output=""
  previous=""
  for argument in "$@"; do
    if [[ "$previous" == "--output" ]]; then
      output="$argument"
      break
    fi
    previous="$argument"
  done
  output="${output/\/artifacts\/corpus/$DENSE_FINALIZER_TEST_CORPUS_ROOT}"
  mkdir -p "${output:h}"
  print -r -- '{"schema_version":"dense-latency-report-v2","max_observed_seconds":1.0}' > "$output"
  print -r -- '{"schema_version":"dense-latency-report-v2","max_observed_seconds":1.0}'
  exit 0
fi
if [[ "$*" == *"verify-best-available-dense"* ]]; then
  print -r -- '{"chunk_count":14549269,"shard_count":15,"recall_at_300":0.99,"max_warm_search_seconds":1.0}'
  exit 0
fi
exit 64
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "DENSE_FINALIZER_PROJECT_ROOT": str(project),
            "DENSE_FINALIZER_CORPUS_ROOT": str(project / "var" / "corpus"),
            "DENSE_FINALIZER_STATE_DIR": str(finalizer_state),
            "DENSE_FINALIZER_DOCKER": str(fake_docker),
            "DENSE_FINALIZER_TEST_MODE": "1",
            "DENSE_FINALIZER_POLL_SECONDS": "0.02",
            "DENSE_FINALIZER_TEST_COMMAND_LOG": str(command_log),
            "DENSE_FINALIZER_TEST_CORPUS_ROOT": str(project / "var" / "corpus"),
        }
    )
    return environment, finalizer_state, command_log


def _run(environment: dict[str, str], command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(FINALIZER), command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )


def test_finalizer_source_is_durable_and_has_no_destructive_commands() -> None:
    source = FINALIZER.read_text(encoding="utf-8")

    assert "nohup" in source
    assert "caffeinate" in source
    assert "launchctl" in source
    assert re.search(r"(?m)^\s*rm\s+-", source) is None
    assert "git clean" not in source
    assert "docker system prune" not in source
    assert "kill -9" not in source


def test_completed_build_is_certified_and_verified_in_worker_image(
    finalizer_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, state, command_log = finalizer_environment

    result = _run(environment, "internal-run")

    assert result.returncode == 0, result.stderr
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["phase"] == "completed"
    assert current["exit_code"] == 0
    verification = json.loads(
        (state / "verification-report.json").read_text(encoding="utf-8")
    )
    assert verification["chunk_count"] == 14_549_269
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert len(commands) == 2
    assert "worker corpus certify-best-available-dense-latency" in commands[0]
    assert "worker corpus verify-best-available-dense" in commands[1]
    assert ":/artifacts/corpus:rw" in commands[0]
    assert "--no-deps" in commands[0]


def test_finalizer_waits_for_build_completion_before_certifying(
    finalizer_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, state, command_log = finalizer_environment
    build_state = state.parent / "dense-build-state" / "state.json"
    build_state.write_text(
        json.dumps({"schema_version": "local-dense-runner-v1", "phase": "building"}),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [str(FINALIZER), "internal-run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        time.sleep(0.08)
        assert process.poll() is None
        assert not command_log.exists()
        build_state.write_text(
            json.dumps(
                {"schema_version": "local-dense-runner-v1", "phase": "completed"}
            ),
            encoding="utf-8",
        )
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)

    assert process.returncode == 0, (stdout, stderr)
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["phase"] == "completed"


@pytest.mark.parametrize(
    "build_phase",
    ["benchmark_failed", "storage_rejected", "build_failed"],
)
def test_terminal_build_failure_blocks_finalization_without_docker(
    finalizer_environment: tuple[dict[str, str], Path, Path],
    build_phase: str,
) -> None:
    environment, state, command_log = finalizer_environment
    (state.parent / "dense-build-state" / "state.json").write_text(
        json.dumps({"schema_version": "local-dense-runner-v1", "phase": build_phase}),
        encoding="utf-8",
    )

    result = _run(environment, "internal-run")

    assert result.returncode != 0
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["phase"] == "blocked"
    assert current["reason"] == f"dense_build_{build_phase}"
    assert not command_log.exists()


def test_latency_failure_is_recorded_and_verification_does_not_run(
    finalizer_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, state, command_log = finalizer_environment
    environment["DENSE_FINALIZER_TEST_LATENCY_FAIL"] = "1"

    result = _run(environment, "internal-run")

    assert result.returncode == 17
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["phase"] == "failed"
    assert current["reason"] == "latency_certification_failed"
    commands = command_log.read_text(encoding="utf-8")
    assert "certify-best-available-dense-latency" in commands
    assert "verify-best-available-dense" not in commands


def test_start_archives_stale_pid_and_launches_one_detached_finalizer(
    finalizer_environment: tuple[dict[str, str], Path, Path],
    tmp_path: Path,
) -> None:
    environment, state, _ = finalizer_environment
    state.mkdir(parents=True, exist_ok=True)
    (state / "finalizer.pid").write_text("99999999\n", encoding="utf-8")
    fake_caffeinate = tmp_path / "fake-caffeinate"
    _write_executable(
        fake_caffeinate,
        """#!/bin/zsh
set -eu
if [[ "$1" == "-i" ]]; then
  shift
fi
exec "$@"
""",
    )
    environment["DENSE_FINALIZER_CAFFEINATE"] = str(fake_caffeinate)

    result = _run(environment, "start")

    assert result.returncode == 0, result.stderr
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if (state / "state.json").exists():
            current = json.loads((state / "state.json").read_text(encoding="utf-8"))
            if current["phase"] == "completed":
                break
        time.sleep(0.02)
    else:
        raise AssertionError((state / "finalizer.log").read_text(encoding="utf-8"))
    assert len(list(state.glob("finalizer.pid.stale.*"))) == 1


def test_start_reports_existing_live_finalizer_without_starting_another(
    finalizer_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, state, command_log = finalizer_environment
    state.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "/bin/zsh",
            "-c",
            "while true; do sleep 1; done",
            str(FINALIZER),
            "internal-run",
        ]
    )
    try:
        time.sleep(0.08)
        (state / "finalizer.pid").write_text(f"{process.pid}\n", encoding="utf-8")

        result = _run(environment, "start")

        assert result.returncode == 0
        assert "already running" in result.stdout.lower()
        assert not command_log.exists()
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)
