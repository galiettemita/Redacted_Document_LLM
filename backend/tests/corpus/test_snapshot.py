from __future__ import annotations

from pathlib import Path

import pytest

from redacted_backend.corpus.snapshot import (
    RemoteSnapshot,
    SnapshotChecksumError,
    SnapshotFile,
    SnapshotLock,
    SnapshotLockError,
    resolve_snapshot,
    verify_snapshot_files,
)

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_snapshot_lock_contains_only_sorted_jsonl_files_at_the_requested_commit() -> (
    None
):
    remote = RemoteSnapshot(
        revision=COMMIT,
        license="cc-by-nc-4.0",
        files=[
            SnapshotFile(path="z.jsonl", size_bytes=3, sha256="b" * 64),
            SnapshotFile(path="foiarchive.json.gz", size_bytes=4, sha256="d" * 64),
            SnapshotFile(path="README.md", size_bytes=2, sha256="c" * 64),
            SnapshotFile(path="a.jsonl", size_bytes=1, sha256="a" * 64),
        ],
    )

    lock = resolve_snapshot(revision=COMMIT, metadata_loader=lambda *_: remote)

    assert lock.repo_id == "HistoryLab/foiarchive"
    assert lock.revision == COMMIT
    assert [item.path for item in lock.files] == [
        "a.jsonl",
        "foiarchive.json.gz",
        "z.jsonl",
    ]
    assert lock.license == "cc-by-nc-4.0"


@pytest.mark.parametrize("revision", ["", "latest", "main", "refs/heads/main"])
def test_snapshot_lock_rejects_mutable_or_missing_revisions(revision: str) -> None:
    with pytest.raises(SnapshotLockError, match="immutable"):
        resolve_snapshot(revision=revision, metadata_loader=lambda *_: None)


def test_snapshot_lock_rejects_metadata_resolved_to_another_commit() -> None:
    remote = RemoteSnapshot(
        revision="f" * 40,
        license="cc-by-nc-4.0",
        files=[SnapshotFile(path="data.jsonl", size_bytes=1, sha256="a" * 64)],
    )

    with pytest.raises(SnapshotLockError, match="did not resolve"):
        resolve_snapshot(revision=COMMIT, metadata_loader=lambda *_: remote)


def test_snapshot_verification_rejects_a_checksum_mismatch(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_bytes(b"changed")
    lock = SnapshotLock(
        repo_id="HistoryLab/foiarchive",
        revision=COMMIT,
        license="cc-by-nc-4.0",
        files=[SnapshotFile(path="data.jsonl", size_bytes=7, sha256="0" * 64)],
    )

    with pytest.raises(SnapshotChecksumError, match="data.jsonl"):
        verify_snapshot_files(lock, tmp_path)
