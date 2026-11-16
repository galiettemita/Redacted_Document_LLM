from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_importing_worker_tasks_does_not_require_corpus_or_model_configuration(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    for name in (
        "REDACTED_ALLOW_PAID_MODEL_CALLS",
        "REDACTED_BM25_MANIFEST_PATH",
        "REDACTED_CORPUS_MANIFEST_PATH",
        "REDACTED_DENSE_ARTIFACT_DIRECTORY",
        "REDACTED_DENSE_MANIFEST_PATH",
        "REDACTED_OPENAI_API_KEY",
        "REDACTED_OPENSEARCH_URL",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "REDACTED_DATABASE_URL": f"sqlite:///{tmp_path / 'worker.db'}",
            "REDACTED_ENVIRONMENT": "test",
            "REDACTED_REDIS_URL": "redis://127.0.0.1:6379/15",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", "import redacted_backend.jobs.tasks"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
