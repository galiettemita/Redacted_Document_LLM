from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from redacted_backend.storage.base import StorageObject


class StorageIntegrityError(ValueError):
    pass


class LocalDocumentStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def put(self, stream: BinaryIO, expected_sha256: str) -> StorageObject:
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise StorageIntegrityError("expected_sha256 must be lowercase SHA-256")

        incoming = self.root / ".incoming"
        incoming.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(dir=incoming, prefix="upload-")
        temporary_path = Path(temporary_name)
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with os.fdopen(descriptor, "wb") as destination:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    destination.write(chunk)
                    size_bytes += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise StorageIntegrityError("stream SHA-256 did not match expected_sha256")

            storage_key = self.key_for(expected_sha256)
            destination_path = self.resolve(storage_key)
            destination_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if destination_path.exists():
                temporary_path.unlink()
            else:
                os.chmod(temporary_path, 0o600)
                os.replace(temporary_path, destination_path)
            return StorageObject(
                storage_key=storage_key,
                sha256=actual_sha256,
                size_bytes=size_bytes,
                local_path=destination_path,
            )
        finally:
            temporary_path.unlink(missing_ok=True)

    def open(self, storage_key: str) -> BinaryIO:
        return self.resolve(storage_key).open("rb")

    def delete(self, storage_key: str) -> None:
        self.resolve(storage_key).unlink(missing_ok=True)

    def resolve(self, storage_key: str) -> Path:
        relative = PurePosixPath(storage_key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("storage key must be a safe relative key")
        path = (self.root / Path(*relative.parts)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("storage key escapes storage root")
        return path

    @staticmethod
    def key_for(sha256: str) -> str:
        return f"sha256/{sha256[:2]}/{sha256[2:4]}/{sha256}.pdf"
