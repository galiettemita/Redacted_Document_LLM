from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import pytest

from redacted_backend.corpus.preflight import PreflightStatus, run_preflight

DiskUsage = namedtuple("DiskUsage", "total used free")
GIB = 1024**3


@pytest.mark.parametrize(
    ("free_gib", "status", "sufficient", "recommended"),
    [
        (149, PreflightStatus.INSUFFICIENT, False, False),
        (150, PreflightStatus.MINIMUM, True, False),
        (250, PreflightStatus.RECOMMENDED, True, True),
    ],
)
def test_preflight_enforces_the_149_150_and_250_gib_boundaries(
    free_gib: int,
    status: PreflightStatus,
    sufficient: bool,
    recommended: bool,
) -> None:
    report = run_preflight(
        [Path("/corpus")],
        _disk_usage=lambda _: DiskUsage(
            500 * GIB, (500 - free_gib) * GIB, free_gib * GIB
        ),
        _system_probe=lambda: {
            "ram_bytes": 32 * GIB,
            "cpu_count": 12,
            "accelerator": "none",
        },
    )

    assert report.status is status
    assert report.sufficient is sufficient
    assert report.recommended is recommended
    assert report.minimum_free_gib == 150
    assert report.recommended_free_gib == 250
    assert report.volumes[0].free_bytes == free_gib * GIB


def test_preflight_uses_the_least_free_target_without_writing_to_it() -> None:
    calls: list[Path] = []

    def disk_usage(path: Path) -> DiskUsage:
        calls.append(path)
        free = 300 * GIB if path == Path("/source") else 149 * GIB
        return DiskUsage(500 * GIB, 500 * GIB - free, free)

    report = run_preflight(
        [Path("/source"), Path("/index")],
        _disk_usage=disk_usage,
        _system_probe=lambda: {"ram_bytes": 1, "cpu_count": 1, "accelerator": "none"},
    )

    assert calls == [Path("/source"), Path("/index")]
    assert report.status is PreflightStatus.INSUFFICIENT
