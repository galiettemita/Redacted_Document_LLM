from __future__ import annotations

from pathlib import Path

from redacted_backend.runtime_identity import build_worker_runtime_identity


WORKER_DOCKERFILE = Path(__file__).parents[2] / "backend" / "Dockerfile.worker"


def test_worker_image_bases_are_digest_pinned() -> None:
    from_lines = [
        line
        for line in WORKER_DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if line.startswith("FROM ")
    ]

    assert from_lines
    assert all("@sha256:" in line for line in from_lines)


def test_worker_identity_changes_with_build_or_host_evidence(tmp_path: Path) -> None:
    package = tmp_path / "redacted_backend"
    package.mkdir()
    source = package / "worker.py"
    source.write_text("BUILD = 1\n", encoding="utf-8")
    lock = tmp_path / "uv.lock"
    lock.write_text("lock-v1\n", encoding="utf-8")
    dockerfile = tmp_path / "Dockerfile.worker"
    dockerfile.write_text("FROM python:3.12\n", encoding="utf-8")
    os_release = tmp_path / "os-release"
    os_release.write_text("ID=test\n", encoding="utf-8")
    boot = tmp_path / "boot_id"
    boot.write_text("boot-one\n", encoding="utf-8")

    first = build_worker_runtime_identity(
        package_root=package,
        lock_path=lock,
        dockerfile_path=dockerfile,
        os_release_path=os_release,
        boot_id_path=boot,
        distribution_versions={"faiss-cpu": "1.16.0"},
        runtime_artifact_hashes={"python": "1" * 64},
        platform_name="Linux-test",
        machine_name="x86_64-test",
    )
    source.write_text("BUILD = 2\n", encoding="utf-8")
    second_build = build_worker_runtime_identity(
        package_root=package,
        lock_path=lock,
        dockerfile_path=dockerfile,
        os_release_path=os_release,
        boot_id_path=boot,
        distribution_versions={"faiss-cpu": "1.16.0"},
        runtime_artifact_hashes={"python": "1" * 64},
        platform_name="Linux-test",
        machine_name="x86_64-test",
    )
    boot.write_text("boot-two\n", encoding="utf-8")
    second_host = build_worker_runtime_identity(
        package_root=package,
        lock_path=lock,
        dockerfile_path=dockerfile,
        os_release_path=os_release,
        boot_id_path=boot,
        distribution_versions={"faiss-cpu": "1.16.0"},
        runtime_artifact_hashes={"python": "1" * 64},
        platform_name="Linux-test",
        machine_name="x86_64-test",
    )
    changed_dependency = build_worker_runtime_identity(
        package_root=package,
        lock_path=lock,
        dockerfile_path=dockerfile,
        os_release_path=os_release,
        boot_id_path=boot,
        distribution_versions={"faiss-cpu": "1.17.0"},
        runtime_artifact_hashes={"python": "1" * 64},
        platform_name="Linux-test",
        machine_name="x86_64-test",
    )

    assert first.worker_build_id != second_build.worker_build_id
    assert second_build.serving_host_id == first.serving_host_id
    assert second_host.worker_build_id == second_build.worker_build_id
    assert second_host.serving_host_id != second_build.serving_host_id
    assert changed_dependency.worker_build_id != second_host.worker_build_id

    changed_runtime = build_worker_runtime_identity(
        package_root=package,
        lock_path=lock,
        dockerfile_path=dockerfile,
        os_release_path=os_release,
        boot_id_path=boot,
        distribution_versions={"faiss-cpu": "1.17.0"},
        runtime_artifact_hashes={"python": "2" * 64},
        platform_name="Linux-test",
        machine_name="x86_64-test",
    )
    assert changed_runtime.worker_build_id != changed_dependency.worker_build_id
