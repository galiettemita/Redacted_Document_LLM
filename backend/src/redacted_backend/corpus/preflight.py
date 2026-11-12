from __future__ import annotations

import os
import platform
import shutil
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

GIB = 1024**3


class PreflightStatus(StrEnum):
    INSUFFICIENT = "insufficient"
    MINIMUM = "minimum"
    RECOMMENDED = "recommended"


class VolumeReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    total_bytes: int = Field(ge=0)
    free_bytes: int = Field(ge=0)


class PreflightReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: PreflightStatus
    sufficient: bool
    recommended: bool
    minimum_free_gib: int
    recommended_free_gib: int
    volumes: list[VolumeReport] = Field(min_length=1)
    ram_bytes: int = Field(ge=0)
    cpu_count: int = Field(ge=1)
    accelerator: str


DiskUsage = Callable[[Path], Any]
SystemProbe = Callable[[], Mapping[str, Any]]


def run_preflight(
    paths: Sequence[Path],
    minimum_free_gib: int = 150,
    recommended_free_gib: int = 250,
    *,
    _disk_usage: DiskUsage = shutil.disk_usage,
    _system_probe: SystemProbe | None = None,
) -> PreflightReport:
    if not paths:
        raise ValueError("At least one corpus or index path is required")
    if minimum_free_gib <= 0 or recommended_free_gib < minimum_free_gib:
        raise ValueError("Disk thresholds are invalid")

    volumes = []
    for path in paths:
        usage = _disk_usage(path)
        volumes.append(
            VolumeReport(
                path=str(path),
                total_bytes=int(usage.total),
                free_bytes=int(usage.free),
            )
        )
    least_free = min(volume.free_bytes for volume in volumes)
    minimum_bytes = minimum_free_gib * GIB
    recommended_bytes = recommended_free_gib * GIB
    if least_free >= recommended_bytes:
        status = PreflightStatus.RECOMMENDED
    elif least_free >= minimum_bytes:
        status = PreflightStatus.MINIMUM
    else:
        status = PreflightStatus.INSUFFICIENT

    system = (_system_probe or _probe_system)()
    return PreflightReport(
        status=status,
        sufficient=least_free >= minimum_bytes,
        recommended=least_free >= recommended_bytes,
        minimum_free_gib=minimum_free_gib,
        recommended_free_gib=recommended_free_gib,
        volumes=volumes,
        ram_bytes=max(0, int(system["ram_bytes"])),
        cpu_count=max(1, int(system["cpu_count"])),
        accelerator=str(system["accelerator"]),
    )


def _probe_system() -> Mapping[str, Any]:
    ram_bytes = 0
    try:
        ram_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pass
    accelerator = "none detected"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        accelerator = "Apple Silicon (availability not benchmarked)"
    return {
        "ram_bytes": ram_bytes,
        "cpu_count": os.cpu_count() or 1,
        "accelerator": accelerator,
    }
