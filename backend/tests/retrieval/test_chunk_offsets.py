from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from redacted_backend.corpus.chunker import CorpusChunk
from redacted_backend.retrieval.chunk_offsets import (
    ChunkOffsetError,
    ChunkOffsetIndex,
    build_chunk_offsets,
)


def _chunk(index: int) -> CorpusChunk:
    text = f"Public archive passage {index}"
    return CorpusChunk(
        id=hashlib.sha256(f"chunk-{index}".encode()).hexdigest(),
        document_id=f"document-{index}",
        ordinal=index,
        text=text,
        word_count=len(text.split()),
        title=f"Title {index}",
        corpus="historylab-unlabeled",
        source_url=None,
        classification=None,
        authored=None,
        doc_lang=None,
    )


def _write_chunks(path: Path, chunks: list[CorpusChunk]) -> str:
    value = "".join(
        json.dumps(
            chunk.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for chunk in chunks
    ).encode("utf-8")
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def test_builds_verified_offsets_and_reads_chunks_concurrently(tmp_path: Path) -> None:
    chunks_fixture = [_chunk(index) for index in range(3)]
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_sha256 = _write_chunks(chunks_path, chunks_fixture)
    offsets_path = tmp_path / "chunks.offsets.u64"

    report = build_chunk_offsets(
        chunks_path,
        offsets_path,
        expected_count=3,
        expected_chunks_sha256=chunks_sha256,
    )

    assert report.chunk_count == 3
    assert report.offsets_size_bytes == 24
    assert report.chunks_sha256 == chunks_sha256
    assert report.offsets_sha256 == hashlib.sha256(offsets_path.read_bytes()).hexdigest()
    with ChunkOffsetIndex.open(
        chunks_path,
        offsets_path,
        expected_count=3,
        expected_chunks_sha256=chunks_sha256,
        expected_offsets_sha256=report.offsets_sha256,
    ) as index:
        assert index.read(2).id == chunks_fixture[2].id
        with ThreadPoolExecutor(max_workers=8) as executor:
            returned = list(executor.map(index.read, [2, 0, 1] * 100))
    assert [chunk.id for chunk in returned] == [
        chunks_fixture[row].id for row in [2, 0, 1] * 100
    ]


def test_build_rejects_the_wrong_chunk_hash_without_publishing(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, [_chunk(0)])
    offsets_path = tmp_path / "chunks.offsets.u64"

    with pytest.raises(ChunkOffsetError, match="chunk checksum"):
        build_chunk_offsets(
            chunks_path,
            offsets_path,
            expected_count=1,
            expected_chunks_sha256="0" * 64,
        )

    assert not offsets_path.exists()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (lambda valid: valid.removesuffix(b"\n"), "final newline"),
        (lambda valid: valid + b"\n", "blank"),
        (lambda valid: valid + b"{not-json}\n", "invalid chunk"),
    ],
)
def test_build_rejects_malformed_jsonl_without_publishing(
    tmp_path: Path,
    value,
    message: str,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, [_chunk(0)])
    chunks_path.write_bytes(value(chunks_path.read_bytes()))
    chunks_sha256 = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    offsets_path = tmp_path / "chunks.offsets.u64"

    with pytest.raises(ChunkOffsetError, match=message):
        build_chunk_offsets(
            chunks_path,
            offsets_path,
            expected_count=1,
            expected_chunks_sha256=chunks_sha256,
        )

    assert not offsets_path.exists()


def test_build_rejects_the_wrong_chunk_count_without_publishing(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_sha256 = _write_chunks(chunks_path, [_chunk(0), _chunk(1)])
    offsets_path = tmp_path / "chunks.offsets.u64"

    with pytest.raises(ChunkOffsetError, match="count"):
        build_chunk_offsets(
            chunks_path,
            offsets_path,
            expected_count=1,
            expected_chunks_sha256=chunks_sha256,
        )

    assert not offsets_path.exists()


def test_open_rejects_truncated_offsets_before_any_read(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_sha256 = _write_chunks(chunks_path, [_chunk(0), _chunk(1)])
    offsets_path = tmp_path / "chunks.offsets.u64"
    report = build_chunk_offsets(
        chunks_path,
        offsets_path,
        expected_count=2,
        expected_chunks_sha256=chunks_sha256,
    )
    offsets_path.write_bytes(offsets_path.read_bytes()[:-1])

    with pytest.raises(ChunkOffsetError, match="size"):
        ChunkOffsetIndex.open(
            chunks_path,
            offsets_path,
            expected_count=2,
            expected_chunks_sha256=chunks_sha256,
            expected_offsets_sha256=report.offsets_sha256,
        )


def test_open_rejects_a_changed_chunk_file(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_sha256 = _write_chunks(chunks_path, [_chunk(0)])
    offsets_path = tmp_path / "chunks.offsets.u64"
    report = build_chunk_offsets(
        chunks_path,
        offsets_path,
        expected_count=1,
        expected_chunks_sha256=chunks_sha256,
    )
    chunks_path.write_bytes(chunks_path.read_bytes() + b"tamper")

    with pytest.raises(ChunkOffsetError, match="chunk checksum"):
        ChunkOffsetIndex.open(
            chunks_path,
            offsets_path,
            expected_count=1,
            expected_chunks_sha256=chunks_sha256,
            expected_offsets_sha256=report.offsets_sha256,
        )


def test_read_rejects_invalid_rows_and_chunk_id_mismatches(tmp_path: Path) -> None:
    chunks_fixture = [_chunk(0), _chunk(1)]
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_sha256 = _write_chunks(chunks_path, chunks_fixture)
    offsets_path = tmp_path / "chunks.offsets.u64"
    report = build_chunk_offsets(
        chunks_path,
        offsets_path,
        expected_count=2,
        expected_chunks_sha256=chunks_sha256,
    )

    with ChunkOffsetIndex.open(
        chunks_path,
        offsets_path,
        expected_count=2,
        expected_chunks_sha256=chunks_sha256,
        expected_offsets_sha256=report.offsets_sha256,
    ) as index:
        for row in (-1, 2):
            with pytest.raises(ChunkOffsetError, match="row"):
                index.read(row)
        with pytest.raises(ChunkOffsetError, match="chunk ID"):
            index.read(1, expected_chunk_id=chunks_fixture[0].id)
