from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = PROJECT_ROOT / "compose.yaml"


def _render_compose(environment: dict[str, str]) -> dict[str, object]:
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            os.devnull,
            "-f",
            str(COMPOSE_FILE),
            "--profile",
            "retrieval",
            "config",
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker CLI not installed")
def test_opensearch_heap_override_preserves_the_production_default() -> None:
    default = _render_compose(os.environ.copy())
    overridden_environment = os.environ.copy()
    overridden_environment["REDACTED_OPENSEARCH_JAVA_OPTS"] = "-Xms2g -Xmx2g"
    overridden = _render_compose(overridden_environment)

    default_options = default["services"]["opensearch"]["environment"][
        "OPENSEARCH_JAVA_OPTS"
    ]
    overridden_options = overridden["services"]["opensearch"]["environment"][
        "OPENSEARCH_JAVA_OPTS"
    ]
    assert default_options == "-Xms4g -Xmx4g"
    assert overridden_options == "-Xms2g -Xmx2g"
