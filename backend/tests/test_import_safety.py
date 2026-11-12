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
