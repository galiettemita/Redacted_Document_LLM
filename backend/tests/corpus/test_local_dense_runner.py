from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT_ROOT / "scripts" / "run-local-dense-build.zsh"
GIB = 1024**3


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _benchmark_report(**updates: object) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": "dense-build-benchmark-v2",
        "sample_rows": 100_000,
        "production_chunk_count": 14_549_269,
        "elapsed_seconds": 1.0,
        "setup_seconds": 0.1,
        "encoding_seconds": 1.0,
        "training_seconds": 0.1,
        "candidate_add_seconds": 0.1,
        "certification_probe_seconds": 0.1,
        "certification_probe_queries": 10,
        "rows_per_second": 100.0,
        "peak_rss_bytes": 1_000,
        "peak_gpu_memory_bytes": None,
        "embedding_bytes_per_row": 768.0,
        "candidate_bytes_per_row": 50.0,
        "candidate_fixed_bytes_per_shard": 1_000_000,
        "projected_shard_count": 15,
        "offset_bytes_per_row": 8.0,
        "projected_full_artifact_bytes": 12 * GIB,
        "projected_full_seconds": 100.0,
        "projected_gpu_hours": 0.0,
        "publishable": False,
        "training_mode": "benchmark-only-all-sample-rows",
    }
    report.update(updates)
    return report


@pytest.fixture
def runner_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    project = tmp_path / "project"
    corpus = project / "var" / "corpus" / "best-available"
    state = corpus / "dense-build-state"
    (project / "backend" / "corpus").mkdir(parents=True)
    corpus.mkdir(parents=True)
    (corpus / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    (corpus / "corpus-manifest.json").write_text("{}\n", encoding="utf-8")
    (project / "backend" / "corpus" / "best_available_application_queries.txt").write_text(
        "query\n",
        encoding="utf-8",
    )
    command_log = tmp_path / "commands.log"
    fake_uv = tmp_path / "fake-uv"
    _write_executable(
        fake_uv,
        """#!/bin/zsh
set -eu
print -r -- "$*" >> "$DENSE_RUNNER_TEST_COMMAND_LOG"
if [[ "$*" == *"benchmark-best-available-dense"* ]]; then
  if [[ "${DENSE_RUNNER_TEST_BENCHMARK_FAIL:-0}" == "1" ]]; then
    exit 17
  fi
  print -r -- "$DENSE_RUNNER_TEST_BENCHMARK_JSON"
  exit 0
fi
if [[ "$*" == *"build-best-available-dense"* ]]; then
  output_manifest=""
  previous=""
  for argument in "$@"; do
    if [[ "$previous" == "--output-manifest" ]]; then
      output_manifest="$argument"
      break
    fi
    previous="$argument"
  done
  mkdir -p "${output_manifest:h}"
  print -r -- '{"schema_version":"sharded-dense-index-manifest-v3"}' > "$output_manifest"
  print -r -- '{"elapsed_seconds":1,"corpus_manifest_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","index_manifest":{"schema_version":"sharded-dense-index-manifest-v3"}}'
  exit 0
fi
exit 64
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "DENSE_RUNNER_PROJECT_ROOT": str(project),
            "DENSE_RUNNER_CORPUS_ROOT": str(project / "var" / "corpus"),
            "DENSE_RUNNER_UV": str(fake_uv),
            "DENSE_RUNNER_TEST_MODE": "1",
            "DENSE_RUNNER_TEST_AVAILABLE_BYTES": str(40 * GIB),
            "DENSE_RUNNER_TEST_COMMAND_LOG": str(command_log),
            "DENSE_RUNNER_TEST_BENCHMARK_JSON": json.dumps(_benchmark_report()),
        }
    )
    return environment, state, command_log


def _run(environment: dict[str, str], command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(RUNNER), command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )


def test_runner_source_uses_durable_launch_without_destructive_commands() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "nohup" in source
    assert "caffeinate" in source
    assert "launchctl" in source
    assert "rm -" not in source
    assert "git clean" not in source
    assert "docker system prune" not in source
    assert "kill -9" not in source


def test_status_distinguishes_stale_resumable_and_live_foreign_pids(
    runner_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, state, _ = runner_environment
    state.mkdir(parents=True)
    (state / "state.json").write_text(
        json.dumps({"schema_version": "local-dense-runner-v1", "phase": "building"}),
        encoding="utf-8",
    )
    (state / "runner.pid").write_text("99999999\n", encoding="utf-8")

    stale = _run(environment, "status")

    assert stale.returncode == 0
    stale_status = json.loads(stale.stdout)
    assert stale_status["runner_status"] == "stopped"
    assert stale_status["interrupted_phase"] == "building"
    assert stale_status["resumable"] is True

    (state / "state.json").write_text(
        json.dumps(
            {"schema_version": "local-dense-runner-v1", "phase": "benchmarking"}
        ),
        encoding="utf-8",
    )
    benchmark_stale = _run(environment, "status")
    benchmark_status = json.loads(benchmark_stale.stdout)
    assert benchmark_status["restart_required"] is True
    assert benchmark_status["resumable"] is False

    (state / "runner.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    foreign = _run(environment, "status")

    assert foreign.returncode != 0
    foreign_status = json.loads(foreign.stdout)
    assert foreign_status["runner_status"] == "foreign_pid"
    os.kill(os.getpid(), 0)


def test_start_reports_an_existing_live_runner_without_starting_another(
    runner_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, state, command_log = runner_environment
    state.mkdir(parents=True)
    process = subprocess.Popen(
        [
            "/bin/zsh",
            "-c",
            "while true; do sleep 1; done",
            str(RUNNER),
            "internal-run",
        ]
    )
    try:
        time.sleep(0.1)
        (state / "runner.pid").write_text(f"{process.pid}\n", encoding="utf-8")

        result = _run(environment, "start")

        assert result.returncode == 0
        assert "already running" in result.stdout.lower()
        assert not command_log.exists()
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_start_archives_a_stale_pid_and_launches_one_detached_runner(
    runner_environment: tuple[dict[str, str], Path, Path],
    tmp_path: Path,
) -> None:
    environment, state, command_log = runner_environment
    state.mkdir(parents=True)
    (state / "runner.pid").write_text("99999999\n", encoding="utf-8")
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
    environment["DENSE_RUNNER_CAFFEINATE"] = str(fake_caffeinate)

    started = _run(environment, "start")

    assert started.returncode == 0
    assert "started with PID" in started.stdout
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if (state / "state.json").exists():
            current = json.loads((state / "state.json").read_text(encoding="utf-8"))
            if current["phase"] == "completed":
                break
        time.sleep(0.05)
    else:
        raise AssertionError((state / "runner.log").read_text(encoding="utf-8"))
    assert len(list(state.glob("runner.pid.stale.*"))) == 1
    commands = command_log.read_text(encoding="utf-8")
    assert commands.count("benchmark-best-available-dense") == 1
    assert commands.count("build-best-available-dense") == 1


def test_benchmark_failure_is_recorded_and_never_starts_production(
    runner_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, state, command_log = runner_environment
    environment["DENSE_RUNNER_TEST_BENCHMARK_FAIL"] = "1"

    result = _run(environment, "internal-run")

    assert result.returncode == 17
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["phase"] == "benchmark_failed"
    assert current["exit_code"] == 17
    commands = command_log.read_text(encoding="utf-8")
    assert "benchmark-best-available-dense" in commands
    assert "build-best-available-dense" not in commands


@pytest.mark.parametrize(
    ("report", "available", "reason"),
    [
        (_benchmark_report(embedding_bytes_per_row=1_400.0), 40 * GIB, "artifact_limit"),
        (_benchmark_report(), 13 * GIB, "free_storage"),
    ],
)
def test_storage_gate_rejects_unsafe_projection_before_production(
    runner_environment: tuple[dict[str, str], Path, Path],
    report: dict[str, object],
    available: int,
    reason: str,
) -> None:
    environment, state, command_log = runner_environment
    environment["DENSE_RUNNER_TEST_BENCHMARK_JSON"] = json.dumps(report)
    environment["DENSE_RUNNER_TEST_AVAILABLE_BYTES"] = str(available)

    result = _run(environment, "internal-run")

    assert result.returncode != 0
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["phase"] == "storage_rejected"
    assert current["reason"] == reason
    assert "build-best-available-dense" not in command_log.read_text(encoding="utf-8")


def test_build_failure_resumes_verified_report_and_preserves_checkpoint(
    runner_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, state, command_log = runner_environment
    dense = state.parent / "dense"
    checkpoint = dense / ".embedding-checkpoints" / "shard-00000" / "manifest.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint\n", encoding="utf-8")
    state.mkdir(parents=True)
    (state / "benchmark-report.json").write_text(
        json.dumps(_benchmark_report()),
        encoding="utf-8",
    )
    (state / "state.json").write_text(
        json.dumps({"schema_version": "local-dense-runner-v1", "phase": "build_failed"}),
        encoding="utf-8",
    )

    result = _run(environment, "internal-run")

    assert result.returncode == 0
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert all("benchmark-best-available-dense" not in command for command in commands)
    build = next(command for command in commands if "build-best-available-dense" in command)
    assert "--shard-size 720000" in build
    assert "--chunks " + str(state.parent / "chunks.jsonl") in build
    assert checkpoint.read_text(encoding="utf-8") == "checkpoint\n"
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["phase"] == "completed"
    assert current["selected_shard_size"] == 720_000
    assert not list(state.glob("*.tmp"))


def test_runner_increases_checkpoint_size_only_as_much_as_storage_requires(
    runner_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, state, command_log = runner_environment
    environment["DENSE_RUNNER_TEST_BENCHMARK_JSON"] = json.dumps(
        _benchmark_report(
            rows_per_second=10.0,
            candidate_fixed_bytes_per_shard=100_000_000,
            projected_shard_count=15,
            projected_full_artifact_bytes=14 * GIB,
        )
    )

    result = _run(environment, "internal-run")

    assert result.returncode == 0
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    selected = current["selected_shard_size"]
    assert 72_000 < selected <= 1_000_000
    assert current["projected_artifact_bytes"] <= 16 * GIB
    build = next(
        command
        for command in command_log.read_text(encoding="utf-8").splitlines()
        if "build-best-available-dense" in command
    )
    assert f"--shard-size {selected}" in build
