from __future__ import annotations

import hashlib
import os
import struct
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.corpus.chunker import CorpusChunk


class ChunkOffsetError(ValueError):
    pass


class ChunkOffsetsReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["chunk-offsets-report-v1"] = "chunk-offsets-report-v1"
    format_version: Literal["shared-chunk-offsets-v1"] = (
        "shared-chunk-offsets-v1"
    )
    chunk_count: int = Field(ge=1)
    chunks_size_bytes: int = Field(ge=1)
    chunks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    offsets_size_bytes: int = Field(ge=8)
    offsets_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_chunk_offsets(
    chunks_path: Path,
    offsets_path: Path,
    *,
    expected_count: int,
    expected_chunks_sha256: str,
) -> ChunkOffsetsReport:
    if expected_count < 1:
        raise ChunkOffsetError("Expected chunk count must be positive")
    offsets_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=offsets_path.parent,
        prefix=f".{offsets_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    chunks_hash = hashlib.sha256()
    offsets_hash = hashlib.sha256()
    chunk_count = 0
    chunks_size = 0
    try:
        try:
            source = chunks_path.open("rb")
        except OSError as error:
            raise ChunkOffsetError(f"Cannot open chunk file: {chunks_path}") from error
        with source, os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                chunks_size += len(line)
                chunks_hash.update(line)
                if not line.endswith(b"\n"):
                    raise ChunkOffsetError("Chunk JSONL must end with a final newline")
                if not line.strip():
                    raise ChunkOffsetError("Chunk JSONL must not contain a blank line")
                try:
                    CorpusChunk.model_validate_json(line)
                except ValueError as error:
                    raise ChunkOffsetError(
                        f"Chunk JSONL contains an invalid chunk at row {chunk_count}"
                    ) from error
                encoded_offset = struct.pack("<Q", offset)
                destination.write(encoded_offset)
                offsets_hash.update(encoded_offset)
                chunk_count += 1

            if chunk_count != expected_count:
                raise ChunkOffsetError(
                    "Chunk count mismatch: "
                    f"expected {expected_count}, found {chunk_count}"
                )
            chunks_sha256 = chunks_hash.hexdigest()
            if chunks_sha256 != expected_chunks_sha256:
                raise ChunkOffsetError("chunk checksum does not match the expected value")
            destination.flush()
            os.fsync(destination.fileno())

        offsets_size = temporary_path.stat().st_size
        if offsets_size != chunk_count * 8:
            raise ChunkOffsetError("Generated offset byte size is invalid")
        os.replace(temporary_path, offsets_path)
        _fsync_directory(offsets_path.parent)
        return ChunkOffsetsReport(
            chunk_count=chunk_count,
            chunks_size_bytes=chunks_size,
            chunks_sha256=chunks_sha256,
            offsets_size_bytes=offsets_size,
            offsets_sha256=offsets_hash.hexdigest(),
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


class ChunkOffsetIndex:
    def __init__(
        self,
        *,
        chunks_descriptor: int,
        offsets_descriptor: int,
        chunk_count: int,
        chunks_size_bytes: int,
    ) -> None:
        self._chunks_descriptor = chunks_descriptor
        self._offsets_descriptor = offsets_descriptor
        self.chunk_count = chunk_count
        self.chunks_size_bytes = chunks_size_bytes
        self._closed = False

    @classmethod
    def open(
        cls,
        chunks_path: Path,
        offsets_path: Path,
        *,
        expected_count: int,
        expected_chunks_sha256: str,
        expected_offsets_sha256: str,
    ) -> ChunkOffsetIndex:
        if expected_count < 1:
            raise ChunkOffsetError("Expected chunk count must be positive")
        chunks_descriptor = -1
        offsets_descriptor = -1
        try:
            chunks_descriptor = os.open(chunks_path, os.O_RDONLY)
            offsets_descriptor = os.open(offsets_path, os.O_RDONLY)
            chunks_size = os.fstat(chunks_descriptor).st_size
            offsets_size = os.fstat(offsets_descriptor).st_size
            if offsets_size != expected_count * 8:
                raise ChunkOffsetError("Chunk offset file has an invalid size")
            (
                chunks_sha256,
                derived_offsets_sha256,
                derived_count,
            ) = _source_identity(chunks_descriptor)
            if chunks_sha256 != expected_chunks_sha256:
                raise ChunkOffsetError("chunk checksum does not match the expected value")
            if derived_count != expected_count:
                raise ChunkOffsetError("Chunk count does not match the expected value")
            if (
                _sha256_descriptor(offsets_descriptor, offsets_size)
                != expected_offsets_sha256
            ):
                raise ChunkOffsetError("Chunk offset checksum does not match expected")
            if derived_offsets_sha256 != expected_offsets_sha256:
                raise ChunkOffsetError(
                    "Chunk offsets do not match the actual JSONL line starts"
                )
            first_offset = _read_offset(offsets_descriptor, 0)
            last_offset = _read_offset(offsets_descriptor, expected_count - 1)
            if first_offset != 0 or last_offset >= chunks_size:
                raise ChunkOffsetError("Chunk offset boundaries are invalid")
            return cls(
                chunks_descriptor=chunks_descriptor,
                offsets_descriptor=offsets_descriptor,
                chunk_count=expected_count,
                chunks_size_bytes=chunks_size,
            )
        except (OSError, ChunkOffsetError) as error:
            if chunks_descriptor >= 0:
                os.close(chunks_descriptor)
            if offsets_descriptor >= 0:
                os.close(offsets_descriptor)
            if isinstance(error, ChunkOffsetError):
                raise
            raise ChunkOffsetError("Cannot open verified chunk offset artifacts") from error

    def read(
        self,
        row: int,
        *,
        expected_chunk_id: str | None = None,
    ) -> CorpusChunk:
        if self._closed:
            raise ChunkOffsetError("Chunk offset index is closed")
        if isinstance(row, bool) or not isinstance(row, int):
            raise ChunkOffsetError("Chunk row must be an integer")
        if row < 0 or row >= self.chunk_count:
            raise ChunkOffsetError("Chunk row is outside the offset index")
        start = _read_offset(self._offsets_descriptor, row)
        end = (
            _read_offset(self._offsets_descriptor, row + 1)
            if row + 1 < self.chunk_count
            else self.chunks_size_bytes
        )
        if start < 0 or end <= start or end > self.chunks_size_bytes:
            raise ChunkOffsetError("Chunk offset range is invalid")
        line = _pread_exact(self._chunks_descriptor, end - start, start)
        if not line.endswith(b"\n") or line.count(b"\n") != 1:
            raise ChunkOffsetError("Chunk offset does not identify one JSONL row")
        try:
            chunk = CorpusChunk.model_validate_json(line)
        except ValueError as error:
            raise ChunkOffsetError("Chunk offset resolved to invalid JSON") from error
        if expected_chunk_id is not None and chunk.id != expected_chunk_id:
            raise ChunkOffsetError("chunk ID does not match the expected result")
        return chunk

    def close(self) -> None:
        if self._closed:
            return
        os.close(self._chunks_descriptor)
        os.close(self._offsets_descriptor)
        self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise ChunkOffsetError("Chunk offset index is closed")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()


def _read_offset(descriptor: int, row: int) -> int:
    value = _pread_exact(descriptor, 8, row * 8)
    return struct.unpack("<Q", value)[0]


def _pread_exact(descriptor: int, size: int, offset: int) -> bytes:
    parts: list[bytes] = []
    remaining = size
    current = offset
    while remaining:
        value = os.pread(descriptor, remaining, current)
        if not value:
            raise ChunkOffsetError("Chunk artifact ended before the expected byte range")
        parts.append(value)
        remaining -= len(value)
        current += len(value)
    return b"".join(parts)


def _sha256_descriptor(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        value = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not value:
            raise ChunkOffsetError("Chunk artifact ended during checksum verification")
        digest.update(value)
        offset += len(value)
    return digest.hexdigest()


def _source_identity(descriptor: int) -> tuple[str, str, int]:
    chunks_digest = hashlib.sha256()
    offsets_digest = hashlib.sha256()
    count = 0
    with os.fdopen(os.dup(descriptor), "rb") as source:
        source.seek(0)
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            chunks_digest.update(line)
            offsets_digest.update(struct.pack("<Q", offset))
            count += 1
    return chunks_digest.hexdigest(), offsets_digest.hexdigest(), count


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
