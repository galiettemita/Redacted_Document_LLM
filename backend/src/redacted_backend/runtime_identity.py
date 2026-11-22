from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field


_RUNTIME_DISTRIBUTIONS = (
    "faiss-cpu",
    "numpy",
    "redacted-document-backend",
    "sentence-transformers",
    "torch",
    "transformers",
)
_NATIVE_SUFFIXES = (".dylib", ".dll", ".pyd", ".so")


class WorkerRuntimeIdentity(BaseModel):
    """Non-configurable evidence identifying one worker build and host boot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["worker-runtime-identity-v1"] = (
        "worker-runtime-identity-v1"
    )
    worker_build_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    serving_host_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    platform: str = Field(min_length=1)
    machine: str = Field(min_length=1)


class WorkerHeartbeat(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["worker-heartbeat-v1"] = "worker-heartbeat-v1"
    status: Literal["ready"] = "ready"
    identity: WorkerRuntimeIdentity


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_build_file(name: str) -> Path:
    backend_root = _default_backend_root()
    candidates = (Path("/app") / name, backend_root / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"Worker build evidence is missing: {name}")


def _installed_distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in _RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def _installed_runtime_artifact_hashes() -> dict[str, str]:
    artifacts = {"python-executable": _sha256_file(Path(sys.executable))}
    for distribution_name in _RUNTIME_DISTRIBUTIONS:
        try:
            installed = distribution(distribution_name)
        except PackageNotFoundError:
            continue
        for relative in installed.files or ():
            relative_name = relative.as_posix()
            lower_name = relative_name.lower()
            if not (
                lower_name.endswith(_NATIVE_SUFFIXES) or ".so." in lower_name
            ):
                continue
            artifact_path = Path(installed.locate_file(relative))
            if artifact_path.is_file():
                artifacts[f"{distribution_name}/{relative_name}"] = _sha256_file(
                    artifact_path
                )
    model_root = Path("/opt/huggingface")
    if model_root.is_dir():
        for artifact_path in sorted(
            (path for path in model_root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(model_root).as_posix(),
        ):
            artifacts[
                f"huggingface/{artifact_path.relative_to(model_root).as_posix()}"
            ] = _sha256_file(artifact_path)
    return artifacts


def _read_host_boot_id(path: Path) -> str:
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
    elif sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = result.stdout.strip()
    else:
        raise RuntimeError("Serving host boot identity is unavailable")
    if not value:
        raise RuntimeError("Serving host boot identity is empty")
    return value


def build_worker_runtime_identity(
    *,
    package_root: Path,
    lock_path: Path,
    dockerfile_path: Path,
    os_release_path: Path,
    boot_id_path: Path,
    distribution_versions: Mapping[str, str],
    runtime_artifact_hashes: Mapping[str, str],
    platform_name: str,
    machine_name: str,
) -> WorkerRuntimeIdentity:
    required_files = (lock_path, dockerfile_path)
    if not package_root.is_dir() or any(not path.is_file() for path in required_files):
        raise RuntimeError("Worker build identity evidence is incomplete")
    source_files = sorted(
        (path for path in package_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(package_root).as_posix(),
    )
    if not source_files:
        raise RuntimeError("Worker source identity evidence is empty")
    if not runtime_artifact_hashes or any(
        len(value) != 64
        or any(
            character not in "0123456789abcdef" for character in value
        )
        for value in runtime_artifact_hashes.values()
    ):
        raise RuntimeError("Worker runtime artifact identity evidence is invalid")
    build_evidence = {
        "dockerfile_sha256": _sha256_file(dockerfile_path),
        "distributions": dict(sorted(distribution_versions.items())),
        "os_release_sha256": (
            _sha256_file(os_release_path)
            if os_release_path.is_file()
            else hashlib.sha256(platform_name.encode("utf-8")).hexdigest()
        ),
        "python": sys.version,
        "runtime_artifacts": dict(sorted(runtime_artifact_hashes.items())),
        "sources": [
            {
                "path": path.relative_to(package_root).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in source_files
        ],
        "uv_lock_sha256": _sha256_file(lock_path),
    }
    canonical = json.dumps(
        build_evidence,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    host_boot_id = _read_host_boot_id(boot_id_path)
    return WorkerRuntimeIdentity(
        worker_build_id=hashlib.sha256(canonical).hexdigest(),
        serving_host_id=hashlib.sha256(host_boot_id.encode("utf-8")).hexdigest(),
        platform=platform_name,
        machine=machine_name,
    )


@lru_cache(maxsize=1)
def current_worker_runtime_identity() -> WorkerRuntimeIdentity:
    return build_worker_runtime_identity(
        package_root=Path(__file__).resolve().parent,
        lock_path=_default_build_file("uv.lock"),
        dockerfile_path=_default_build_file("Dockerfile.worker"),
        os_release_path=Path("/etc/os-release"),
        boot_id_path=Path("/proc/sys/kernel/random/boot_id"),
        distribution_versions=_installed_distribution_versions(),
        runtime_artifact_hashes=_installed_runtime_artifact_hashes(),
        platform_name=platform.platform(),
        machine_name=platform.machine(),
    )
