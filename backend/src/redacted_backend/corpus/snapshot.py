from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_REPO_ID = "HistoryLab/foiarchive"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class SnapshotLockError(ValueError):
    pass


class SnapshotChecksumError(ValueError):
    pass


class SnapshotFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError("snapshot paths must be safe relative paths")
        return path.as_posix()


class SnapshotLock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_id: str = DEFAULT_REPO_ID
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str
    files: list[SnapshotFile] = Field(min_length=1)

    def manifest_hash(self) -> str:
        payload = self.model_dump_json(
            by_alias=True,
            exclude_none=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class RemoteSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    revision: str
    license: str
    files: list[SnapshotFile]


MetadataLoader = Callable[[str, str], RemoteSnapshot]


def resolve_snapshot(
    *,
    revision: str,
    repo_id: str = DEFAULT_REPO_ID,
    metadata_loader: MetadataLoader | None = None,
) -> SnapshotLock:
    normalized_revision = revision.strip().lower()
    if not _COMMIT_PATTERN.fullmatch(normalized_revision):
        raise SnapshotLockError(
            "An immutable 40-character repository commit is required; "
            "branches such as main and aliases such as latest are not accepted"
        )

    remote = (metadata_loader or _load_remote_snapshot)(
        repo_id,
        normalized_revision,
    )
    if remote.revision.lower() != normalized_revision:
        raise SnapshotLockError(
            f"Requested revision {normalized_revision} did not resolve to the same commit"
        )
    jsonl_files = sorted(
        (item for item in remote.files if _is_json_lines(item.path)),
        key=lambda item: item.path,
    )
    if not jsonl_files:
        raise SnapshotLockError("The resolved snapshot contains no JSONL files")
    return SnapshotLock(
        repo_id=repo_id,
        revision=normalized_revision,
        license=remote.license,
        files=jsonl_files,
    )


def verify_snapshot_files(lock: SnapshotLock, root: Path) -> None:
    for item in lock.files:
        path = root.joinpath(*PurePosixPath(item.path).parts)
        if not path.is_file():
            raise SnapshotChecksumError(f"Snapshot file is missing: {item.path}")
        if path.stat().st_size != item.size_bytes:
            raise SnapshotChecksumError(f"Snapshot size mismatch: {item.path}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != item.sha256:
            raise SnapshotChecksumError(f"Snapshot checksum mismatch: {item.path}")


def verify_snapshot_inputs(lock: SnapshotLock, source_paths: Sequence[Path]) -> None:
    expected_by_name = {PurePosixPath(item.path).name: item for item in lock.files}
    if len(expected_by_name) != len(lock.files):
        raise SnapshotChecksumError(
            "Snapshot contains duplicate JSONL filenames; preserve its directory tree"
        )
    actual_by_name = {path.name: path for path in source_paths}
    if len(actual_by_name) != len(source_paths):
        raise SnapshotChecksumError("Input paths contain duplicate JSONL filenames")
    if set(actual_by_name) != set(expected_by_name):
        missing = sorted(set(expected_by_name) - set(actual_by_name))
        unexpected = sorted(set(actual_by_name) - set(expected_by_name))
        raise SnapshotChecksumError(
            f"Snapshot input set mismatch; missing={missing}, unexpected={unexpected}"
        )
    for name, item in expected_by_name.items():
        path = actual_by_name[name]
        if not path.is_file():
            raise SnapshotChecksumError(f"Snapshot file is missing: {item.path}")
        if path.stat().st_size != item.size_bytes:
            raise SnapshotChecksumError(f"Snapshot size mismatch: {item.path}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != item.sha256:
            raise SnapshotChecksumError(f"Snapshot checksum mismatch: {item.path}")


def _load_remote_snapshot(repo_id: str, revision: str) -> RemoteSnapshot:
    from huggingface_hub import HfApi

    info = HfApi(token=os.environ.get("HF_TOKEN")).dataset_info(
        repo_id=repo_id,
        revision=revision,
        files_metadata=True,
    )
    files: list[SnapshotFile] = []
    for sibling in info.siblings or []:
        path = sibling.rfilename
        if not _is_json_lines(path):
            continue
        lfs = sibling.lfs
        checksum = getattr(lfs, "sha256", None) if lfs is not None else None
        if not checksum:
            raise SnapshotLockError(f"JSONL file has no SHA-256 metadata: {path}")
        if sibling.size is None:
            raise SnapshotLockError(f"JSONL file has no size metadata: {path}")
        files.append(
            SnapshotFile(
                path=path,
                size_bytes=sibling.size,
                sha256=checksum,
            )
        )
    card_data = info.card_data
    license_name = getattr(card_data, "license", None) if card_data else None
    return RemoteSnapshot(
        revision=info.sha,
        license=str(license_name or "unknown"),
        files=files,
    )


def _is_json_lines(path: str) -> bool:
    normalized = path.lower()
    return normalized.endswith((".jsonl", ".jsonl.gz", ".json.gz"))
