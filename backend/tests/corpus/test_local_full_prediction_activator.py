from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACTIVATOR = PROJECT_ROOT / "scripts" / "activate-local-full-predictions.zsh"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _finalizer_state(path: Path, phase: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "local-dense-finalizer-v1",
                "phase": phase,
                "exit_code": 0 if phase == "completed" else None,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def activator_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    env_file = project / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=secret-that-must-never-be-logged\n"
        "REDACTED_ALLOW_PAID_MODEL_CALLS=true\n"
        "REDACTED_ALLOW_STAGED_BM25_CALLS=true\n"
        "REDACTED_ALLOW_BEST_AVAILABLE_CALLS=false\n"
        "REDACTED_ALLOW_PAPER_EXACT_CALLS=false\n",
        encoding="utf-8",
    )
    finalizer = project / "var" / "corpus" / "best-available" / "dense-finalizer-state"
    finalizer.mkdir(parents=True)
    _finalizer_state(finalizer / "state.json", "completed")
    (finalizer / "verification-report.json").write_text(
        json.dumps(
            {
                "chunk_count": 14_549_269,
                "recall_at_300": 0.99,
                "max_warm_search_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    activation = project / "var" / "corpus" / "best-available" / "dense-activation-state"
    command_log = tmp_path / "commands.log"
    fake_docker = tmp_path / "fake-docker"
    _write_executable(
        fake_docker,
        """#!/bin/zsh
set -eu
print -r -- "docker $*" >> "$FULL_ACTIVATOR_TEST_COMMAND_LOG"
if [[ "${FULL_ACTIVATOR_TEST_DOCKER_FAIL:-0}" == "1" ]]; then
  exit 17
fi
""",
    )
    fake_curl = tmp_path / "fake-curl"
    _write_executable(
        fake_curl,
        r'''#!/bin/zsh
set -eu
print -r -- "curl $*" >> "$FULL_ACTIVATOR_TEST_COMMAND_LOG"
if [[ "${FULL_ACTIVATOR_TEST_READINESS_READY:-1}" == "1" ]]; then
  print -r -- '{"prediction_bases":{"best_available_foiarchive_v1":{"operational":true,"approved":true,"ready":true}},"strongest_ready_basis":"best_available_foiarchive_v1"}'
else
  print -r -- '{"prediction_bases":{"best_available_foiarchive_v1":{"operational":false,"approved":true,"ready":false}},"strongest_ready_basis":"best_available_foiarchive_bm25_v1"}'
fi
''',
    )
    environment = os.environ.copy()
    environment.update(
        {
            "FULL_ACTIVATOR_PROJECT_ROOT": str(project),
            "FULL_ACTIVATOR_STATE_DIR": str(activation),
            "FULL_ACTIVATOR_FINALIZER_STATE_FILE": str(finalizer / "state.json"),
            "FULL_ACTIVATOR_VERIFICATION_REPORT": str(
                finalizer / "verification-report.json"
            ),
            "FULL_ACTIVATOR_ENV_FILE": str(env_file),
            "FULL_ACTIVATOR_DOCKER": str(fake_docker),
            "FULL_ACTIVATOR_CURL": str(fake_curl),
            "FULL_ACTIVATOR_POLL_SECONDS": "0.02",
            "FULL_ACTIVATOR_READINESS_TIMEOUT_SECONDS": "0.12",
            "FULL_ACTIVATOR_READINESS_PROBE_TIMEOUT_SECONDS": "0.02",
            "FULL_ACTIVATOR_TEST_MODE": "1",
            "FULL_ACTIVATOR_TEST_COMMAND_LOG": str(command_log),
        }
    )
    return environment, activation, env_file, command_log


def _run(environment: dict[str, str], command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ACTIVATOR), command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )


def test_activator_source_is_durable_and_has_no_prediction_or_destructive_call() -> None:
    source = ACTIVATOR.read_text(encoding="utf-8")

    assert "nohup" in source
    assert "caffeinate" in source
    assert "launchctl" in source
    assert "FULL_ACTIVATOR_LAUNCH_LABEL" in source
    assert re.search(r"(?m)^\s*rm\s+-", source) is None
    assert "git clean" not in source
    assert "docker system prune" not in source
    assert "kill -9" not in source
    assert "/predictions" not in source
    assert "/analysis" not in source


def test_verified_completion_enables_only_full_basis_and_proves_live_readiness(
    activator_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, state, env_file, command_log = activator_environment
    before = env_file.read_text(encoding="utf-8")

    result = _run(environment, "internal-run")

    assert result.returncode == 0, (result.stdout, result.stderr)
    after = env_file.read_text(encoding="utf-8")
    assert after == before.replace(
        "REDACTED_ALLOW_BEST_AVAILABLE_CALLS=false",
        "REDACTED_ALLOW_BEST_AVAILABLE_CALLS=true",
    )
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert commands[0] == "docker compose up -d --force-recreate backend worker"
    assert commands[1].startswith(
        "curl -fsS --connect-timeout 0.02 --max-time 0.02 "
    )
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["phase"] == "activated"
    assert current["exit_code"] == 0
    disclosed = result.stdout + result.stderr + (state / "activator.log").read_text(
        encoding="utf-8"
    ) if (state / "activator.log").exists() else result.stdout + result.stderr
    assert "secret-that-must-never-be-logged" not in disclosed


def test_activator_waits_for_finalizer_before_mutating_or_deploying(
    activator_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, state, env_file, command_log = activator_environment
    finalizer_state = Path(environment["FULL_ACTIVATOR_FINALIZER_STATE_FILE"])
    _finalizer_state(finalizer_state, "waiting")
    before = env_file.read_text(encoding="utf-8")
    process = subprocess.Popen(
        [str(ACTIVATOR), "internal-run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        time.sleep(0.08)
        assert process.poll() is None
        assert env_file.read_text(encoding="utf-8") == before
        assert not command_log.exists()
        _finalizer_state(finalizer_state, "completed")
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)

    assert process.returncode == 0, (stdout, stderr)
    assert json.loads((state / "state.json").read_text(encoding="utf-8"))[
        "phase"
    ] == "activated"


@pytest.mark.parametrize("terminal_phase", ["failed", "blocked"])
def test_terminal_finalizer_failure_blocks_without_config_or_docker_change(
    activator_environment: tuple[dict[str, str], Path, Path, Path],
    terminal_phase: str,
) -> None:
    environment, state, env_file, command_log = activator_environment
    _finalizer_state(
        Path(environment["FULL_ACTIVATOR_FINALIZER_STATE_FILE"]), terminal_phase
    )
    before = env_file.read_text(encoding="utf-8")

    result = _run(environment, "internal-run")

    assert result.returncode != 0
    assert env_file.read_text(encoding="utf-8") == before
    assert not command_log.exists()
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["phase"] == "blocked"
    assert current["reason"] == f"dense_finalizer_{terminal_phase}"


@pytest.mark.parametrize(
    "report",
    [
        {"chunk_count": 14_549_268, "recall_at_300": 0.99, "max_warm_search_seconds": 1.0},
        {"chunk_count": 14_549_269, "recall_at_300": 0.97, "max_warm_search_seconds": 1.0},
        {"chunk_count": 14_549_269, "recall_at_300": 0.99, "max_warm_search_seconds": 5.01},
    ],
)
def test_invalid_verification_report_cannot_enable_full_basis(
    activator_environment: tuple[dict[str, str], Path, Path, Path],
    report: dict[str, float | int],
) -> None:
    environment, state, env_file, command_log = activator_environment
    Path(environment["FULL_ACTIVATOR_VERIFICATION_REPORT"]).write_text(
        json.dumps(report), encoding="utf-8"
    )
    before = env_file.read_text(encoding="utf-8")

    result = _run(environment, "internal-run")

    assert result.returncode != 0
    assert env_file.read_text(encoding="utf-8") == before
    assert not command_log.exists()
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["phase"] == "blocked"
    assert current["reason"] == "verification_report_invalid"


def test_master_approval_must_already_be_true(
    activator_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, state, env_file, command_log = activator_environment
    env_file.write_text(
        env_file.read_text(encoding="utf-8").replace(
            "REDACTED_ALLOW_PAID_MODEL_CALLS=true",
            "REDACTED_ALLOW_PAID_MODEL_CALLS=false",
        ),
        encoding="utf-8",
    )
    before = env_file.read_text(encoding="utf-8")

    result = _run(environment, "internal-run")

    assert result.returncode != 0
    assert env_file.read_text(encoding="utf-8") == before
    assert not command_log.exists()
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["reason"] == "master_approval_not_enabled"


def test_readiness_probe_timeout_cannot_exceed_the_overall_deadline(
    activator_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, state, env_file, command_log = activator_environment
    environment["FULL_ACTIVATOR_READINESS_PROBE_TIMEOUT_SECONDS"] = "0.13"
    before = env_file.read_text(encoding="utf-8")

    result = _run(environment, "internal-run")

    assert result.returncode == 64
    assert "cannot exceed the overall readiness timeout" in result.stderr
    assert env_file.read_text(encoding="utf-8") == before
    assert not command_log.exists()
    assert not (state / "state.json").exists()


def test_docker_failure_is_recorded_without_readiness_or_secret_disclosure(
    activator_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, state, env_file, command_log = activator_environment
    environment["FULL_ACTIVATOR_TEST_DOCKER_FAIL"] = "1"

    result = _run(environment, "internal-run")

    assert result.returncode == 17
    assert "REDACTED_ALLOW_BEST_AVAILABLE_CALLS=false" in env_file.read_text(
        encoding="utf-8"
    )
    assert command_log.read_text(encoding="utf-8").splitlines() == [
        "docker compose up -d --force-recreate backend worker",
        "docker compose up -d --force-recreate backend worker",
    ]
    assert all(
        not line.startswith("curl ")
        for line in command_log.read_text(encoding="utf-8").splitlines()
    )
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["phase"] == "failed"
    assert current["reason"] == "compose_recreate_failed_rollback_recreate_failed"
    assert "secret-that-must-never-be-logged" not in result.stdout + result.stderr


def test_unready_live_service_never_records_activation(
    activator_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, state, env_file, command_log = activator_environment
    environment["FULL_ACTIVATOR_TEST_READINESS_READY"] = "0"

    result = _run(environment, "internal-run")

    assert result.returncode != 0
    current = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert current["phase"] == "failed"
    assert current["reason"] == "full_basis_readiness_timeout"
    assert "REDACTED_ALLOW_BEST_AVAILABLE_CALLS=false" in env_file.read_text(
        encoding="utf-8"
    )
    assert command_log.read_text(encoding="utf-8").splitlines()[-1] == (
        "docker compose up -d --force-recreate backend worker"
    )


def test_failed_activation_removes_an_approval_key_it_added(
    activator_environment: tuple[dict[str, str], Path, Path, Path],
) -> None:
    environment, _state, env_file, command_log = activator_environment
    env_file.write_text(
        env_file.read_text(encoding="utf-8").replace(
            "REDACTED_ALLOW_BEST_AVAILABLE_CALLS=false\n",
            "",
        ),
        encoding="utf-8",
    )
    environment["FULL_ACTIVATOR_TEST_READINESS_READY"] = "0"

    result = _run(environment, "internal-run")

    assert result.returncode != 0
    assert "REDACTED_ALLOW_BEST_AVAILABLE_CALLS=" not in env_file.read_text(
        encoding="utf-8"
    )
    assert command_log.read_text(encoding="utf-8").splitlines()[-1] == (
        "docker compose up -d --force-recreate backend worker"
    )


def test_managed_activator_removes_its_launchd_job_after_failure(
    activator_environment: tuple[dict[str, str], Path, Path, Path],
    tmp_path: Path,
) -> None:
    environment, state, _env_file, _command_log = activator_environment
    environment["FULL_ACTIVATOR_TEST_DOCKER_FAIL"] = "1"
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = tmp_path / "fake-launchctl"
    _write_executable(
        fake_launchctl,
        """#!/bin/zsh
set -eu
print -r -- "$*" >> "$FULL_ACTIVATOR_TEST_LAUNCHCTL_LOG"
""",
    )
    environment.update(
        {
            "FULL_ACTIVATOR_LAUNCHCTL": str(fake_launchctl),
            "FULL_ACTIVATOR_LAUNCH_LABEL": "com.example.one-shot-activator",
            "FULL_ACTIVATOR_TEST_LAUNCHCTL_LOG": str(launchctl_log),
        }
    )

    result = _run(environment, "internal-run-managed")

    assert result.returncode == 17
    assert launchctl_log.read_text(encoding="utf-8").splitlines() == [
        "remove com.example.one-shot-activator"
    ]
    assert json.loads((state / "state.json").read_text(encoding="utf-8"))[
        "phase"
    ] == "failed"
