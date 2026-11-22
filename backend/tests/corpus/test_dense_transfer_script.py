from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "scripts" / "transfer-compressed-dense.zsh"
HEADROOM_BYTES = 2 * 1024**3


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _remote_fixture(root: Path) -> tuple[Path, dict[str, object]]:
    remote = root / "remote"
    shard = remote / "shard-00000"
    shard.mkdir(parents=True)
    files = {
        "chunks.offsets.u64": b"offsets" * 700,
        "candidate-template.faiss": b"template",
        "recall-report.json": b'{"recall":1}\n',
        "recall-queries.txt": b"query\n",
        "shard-00000/embeddings.f16": b"embeddings",
        "shard-00000/candidate.faiss": b"candidate",
    }
    for relative, value in files.items():
        path = remote / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    shard_manifest = {
        "schema_version": "compressed-dense-shard-manifest-v1",
        "artifact_profile_hash": "1" * 64,
        "candidate_template_sha256": _sha256(files["candidate-template.faiss"]),
        "shard_id": 0,
        "path": "shard-00000",
        "start_row": 0,
        "end_row": 1,
        "chunk_count": 1,
        "embeddings_size_bytes": len(files["shard-00000/embeddings.f16"]),
        "embeddings_sha256": _sha256(files["shard-00000/embeddings.f16"]),
        "candidate_index_size_bytes": len(files["shard-00000/candidate.faiss"]),
        "candidate_index_sha256": _sha256(files["shard-00000/candidate.faiss"]),
    }
    total = (
        len(files["chunks.offsets.u64"])
        + len(files["candidate-template.faiss"])
        + len(files["recall-report.json"])
        + len(files["shard-00000/embeddings.f16"])
        + len(files["shard-00000/candidate.faiss"])
    )
    manifest: dict[str, object] = {
        "schema_version": "sharded-dense-index-manifest-v2",
        "chunk_offsets_path": "chunks.offsets.u64",
        "chunk_offsets_size_bytes": len(files["chunks.offsets.u64"]),
        "chunk_offsets_sha256": _sha256(files["chunks.offsets.u64"]),
        "candidate_template_path": "candidate-template.faiss",
        "candidate_template_size_bytes": len(files["candidate-template.faiss"]),
        "candidate_template_sha256": _sha256(files["candidate-template.faiss"]),
        "recall_report_path": "recall-report.json",
        "recall_report_size_bytes": len(files["recall-report.json"]),
        "recall_report_sha256": _sha256(files["recall-report.json"]),
        "recall_queries_sha256": _sha256(files["recall-queries.txt"]),
        "total_artifact_bytes": total,
        "shards": [shard_manifest],
    }
    (remote / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return remote, manifest


def _fake_commands(root: Path) -> Path:
    commands = root / "bin"
    commands.mkdir()
    (commands / "ssh").write_text(
        """#!/bin/zsh
set -eu
command_text="${@[-1]}"
if [[ "$command_text" == *"manifest.json"* ]]; then
  cat "$FAKE_REMOTE_DENSE/manifest.json"
elif [[ "$command_text" == *"recall-queries.txt"* ]]; then
  wc -c < "$FAKE_REMOTE_DENSE/recall-queries.txt"
else
  exit 64
fi
""",
        encoding="utf-8",
    )
    (commands / "rsync").write_text(
        """#!/bin/zsh
set -eu
if [[ "${FAIL_RSYNC:-0}" == "1" ]]; then
  exit 23
fi
source_path="${@[-2]}"
destination="${@[-1]}"
relative="${source_path#*:$REMOTE_DENSE/}"
print -- "$relative" >> "$FAKE_RSYNC_LOG"
mkdir -p "${destination:h}"
cp "$FAKE_REMOTE_DENSE/$relative" "$destination"
""",
        encoding="utf-8",
    )
    (commands / "df").write_text(
        """#!/bin/zsh
set -eu
blocks=$((FAKE_AVAILABLE_BYTES / 1024))
print 'Filesystem 1024-blocks Used Available Capacity Mounted on'
print "fake 999999 0 $blocks 0% /"
""",
        encoding="utf-8",
    )
    for path in commands.iterdir():
        path.chmod(0o755)
    return commands


def _run(
    root: Path,
    *,
    available_bytes: int,
    fail_rsync: bool = False,
    dense_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    remote, _manifest = _remote_fixture(root) if not (root / "remote").exists() else (
        root / "remote",
        {},
    )
    commands = root / "bin"
    if not commands.exists():
        commands = _fake_commands(root)
    project = root / "project"
    project.mkdir(exist_ok=True)
    key = root / "id_ed25519"
    key.touch()
    environment = os.environ | {
        "PATH": f"{commands}:{os.environ['PATH']}",
        "PROJECT_ROOT": str(project),
        "REMOTE": "root@fake-host",
        "SSH_PORT": "22",
        "SSH_KEY": str(key),
        "REMOTE_DENSE": "/workspace/redacted-build/best-available/dense",
        "FAKE_REMOTE_DENSE": str(remote),
        "FAKE_AVAILABLE_BYTES": str(available_bytes),
        "FAKE_RSYNC_LOG": str(root / "rsync.log"),
        "FAIL_RSYNC": "1" if fail_rsync else "0",
    }
    if dense_dir is not None:
        environment["DENSE_DIR"] = str(dense_dir)
    return subprocess.run(
        ["/bin/zsh", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_native_zsh_transfer_verifies_files_and_promotes_manifest_last(
    tmp_path: Path,
) -> None:
    _remote, manifest = _remote_fixture(tmp_path)

    result = _run(tmp_path, available_bytes=HEADROOM_BYTES + 1024**2)

    assert result.returncode == 0, result.stderr
    dense = tmp_path / "project/var/corpus/best-available/dense"
    assert json.loads((dense / "manifest.json").read_text()) == manifest
    assert json.loads((dense / "shard-00000/manifest.json").read_text()) == (
        manifest["shards"][0]
    )


def test_failed_rsync_never_replaces_the_installed_manifest(tmp_path: Path) -> None:
    _remote_fixture(tmp_path)
    dense = tmp_path / "project/var/corpus/best-available/dense"
    dense.mkdir(parents=True)
    installed = dense / "manifest.json"
    installed.write_text("old verified manifest", encoding="utf-8")

    result = _run(
        tmp_path,
        available_bytes=HEADROOM_BYTES + 1024**2,
        fail_rsync=True,
    )

    assert result.returncode != 0
    assert installed.read_text(encoding="utf-8") == "old verified manifest"


def test_transfer_rejects_unsafe_manifest_paths_before_rsync(tmp_path: Path) -> None:
    remote, manifest = _remote_fixture(tmp_path)
    manifest["shards"][0]["path"] = "../../escape"
    (remote / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = _run(tmp_path, available_bytes=HEADROOM_BYTES + 1024**2)

    assert result.returncode != 0
    assert not (tmp_path / "rsync.log").exists()
    assert not (tmp_path / "project/var/corpus/best-available/dense/manifest.json").exists()


def test_transfer_rejects_multiple_json_documents_before_rsync(
    tmp_path: Path,
) -> None:
    remote, manifest = _remote_fixture(tmp_path)
    malicious = dict(manifest)
    malicious["chunk_offsets_path"] = "../escaped-offsets"
    (remote / "manifest.json").write_text(
        json.dumps(malicious) + "\n" + json.dumps(manifest) + "\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, available_bytes=HEADROOM_BYTES + 1024**2)

    assert result.returncode != 0
    assert not (tmp_path / "rsync.log").exists()
    assert not (tmp_path / "project/var/corpus/best-available/escaped-offsets").exists()
    assert not (tmp_path / "project/var/corpus/best-available/dense/manifest.json").exists()


def test_transfer_rejects_existing_symlink_destination_components(
    tmp_path: Path,
) -> None:
    _remote_fixture(tmp_path)
    dense = tmp_path / "project/var/corpus/best-available/dense"
    outside = tmp_path / "outside"
    dense.mkdir(parents=True)
    outside.mkdir()
    (dense / "shard-00000").symlink_to(outside, target_is_directory=True)

    result = _run(tmp_path, available_bytes=HEADROOM_BYTES + 1024**2)

    assert result.returncode != 0
    assert not (tmp_path / "rsync.log").exists()
    assert list(outside.iterdir()) == []
    assert not (dense / "manifest.json").exists()


def test_transfer_rejects_symlink_above_dense_directory(tmp_path: Path) -> None:
    _remote_fixture(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside-parent"
    outside.mkdir()
    (project / "var").symlink_to(outside, target_is_directory=True)

    result = _run(tmp_path, available_bytes=HEADROOM_BYTES + 1024**2)

    assert result.returncode != 0
    assert not (tmp_path / "rsync.log").exists()
    assert list(outside.iterdir()) == []


def test_transfer_rejects_dense_directory_outside_project(tmp_path: Path) -> None:
    _remote_fixture(tmp_path)
    outside = tmp_path / "outside-dense"

    result = _run(
        tmp_path,
        available_bytes=HEADROOM_BYTES + 1024**2,
        dense_dir=outside,
    )

    assert result.returncode != 0
    assert not outside.exists()
    assert not (tmp_path / "rsync.log").exists()


def test_resume_requires_space_only_for_remaining_bytes(tmp_path: Path) -> None:
    _remote_fixture(tmp_path)
    first = _run(tmp_path, available_bytes=HEADROOM_BYTES + 1024**2)
    assert first.returncode == 0, first.stderr

    resumed = _run(tmp_path, available_bytes=HEADROOM_BYTES + 1024)

    assert resumed.returncode == 0, resumed.stderr
