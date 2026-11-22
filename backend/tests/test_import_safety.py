from __future__ import annotations

import subprocess
import sys


def test_importing_configuration_does_not_construct_the_web_application() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import redacted_backend.config; "
                "assert 'redacted_backend.main' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_importing_web_application_does_not_require_worker_only_faiss() -> None:
    script = """
import builtins
import sys

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "faiss" or name.startswith("faiss."):
        raise ModuleNotFoundError("FAISS is intentionally absent from the API image")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import redacted_backend.main
assert "faiss" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
