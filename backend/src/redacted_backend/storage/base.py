from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol


@dataclass(frozen=True)
class StorageObject:
    storage_key: str
    sha256: str
    size_bytes: int
    local_path: Path


class DocumentStore(Protocol):
    def put(self, stream: BinaryIO, expected_sha256: str) -> StorageObject: ...

    def open(self, storage_key: str) -> BinaryIO: ...

    def delete(self, storage_key: str) -> None: ...
