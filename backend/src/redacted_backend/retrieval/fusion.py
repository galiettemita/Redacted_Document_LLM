from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

from redacted_backend.domain.retrieval import (
    FourRankedLists,
    FusedChunk,
    FusedDocument,
)
from redacted_backend.retrieval.types import RankedChunk
from redacted_backend.scientific_config import ScientificConfig

_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def fuse_documents(
    lists: FourRankedLists,
    *,
    rrf_k: int = 60,
    limit: int = 100,
    chunks_per_document: int = 2,
    scientific: ScientificConfig | None = None,
) -> list[FusedDocument]:
    if rrf_k < 1 or limit < 1 or chunks_per_document < 1:
        raise ValueError("Fusion limits and RRF k must be positive")
    if chunks_per_document > 2:
        raise ValueError("The realistic method permits at most two chunks per document")
    config = scientific or ScientificConfig()
    all_chunks = [chunk for _, ranked in lists.named() for chunk in ranked]
    if not all_chunks:
        return []
    aliases = _merge_document_aliases(all_chunks, config)

    source_ids: dict[str, set[str]] = defaultdict(set)
    document_scores: dict[str, float] = defaultdict(float)
    chunk_scores: dict[tuple[str, str], float] = defaultdict(float)
    chunk_examples: dict[tuple[str, str], RankedChunk] = {}
    for _list_name, ranked in lists.named():
        best_document_ranks: dict[str, int] = {}
        best_chunk_ranks: dict[tuple[str, str], int] = {}
        for chunk in ranked:
            document_id = aliases[chunk.document_id]
            source_ids[document_id].add(chunk.document_id)
            best_document_ranks[document_id] = min(
                chunk.rank,
                best_document_ranks.get(document_id, chunk.rank),
            )
            key = (document_id, chunk.chunk_id)
            current = best_chunk_ranks.get(key)
            if current is None or chunk.rank < current:
                best_chunk_ranks[key] = chunk.rank
                chunk_examples[key] = chunk
        for document_id, rank in best_document_ranks.items():
            document_scores[document_id] += 1 / (rrf_k + rank)
        for key, rank in best_chunk_ranks.items():
            chunk_scores[key] += 1 / (rrf_k + rank)

    ordered_documents = sorted(
        document_scores,
        key=lambda document_id: (-document_scores[document_id], document_id),
    )[:limit]
    output: list[FusedDocument] = []
    for document_id in ordered_documents:
        selected_keys = sorted(
            (key for key in chunk_scores if key[0] == document_id),
            key=lambda key: (-chunk_scores[key], key[1]),
        )[:chunks_per_document]
        output.append(
            FusedDocument(
                document_id=document_id,
                source_document_ids=sorted(source_ids[document_id]),
                rrf_score=document_scores[document_id],
                chunks=[
                    FusedChunk(
                        chunk_id=key[1],
                        text=chunk_examples[key].text,
                        rrf_score=chunk_scores[key],
                    )
                    for key in selected_keys
                ],
            )
        )
    return output


def normalized_filename(chunk: RankedChunk, scientific: ScientificConfig) -> str:
    candidate = chunk.title or ""
    if not candidate and chunk.source_url:
        path = unquote(urlparse(chunk.source_url).path)
        candidate = PurePosixPath(path).name
    value = unicodedata.normalize("NFKC", candidate).casefold().replace("_", " ")
    value = re.sub(r"\.[a-z0-9]{1,8}$", "", value)
    value = re.sub(r"[-.]+", " ", value)
    value = " ".join(value.split())
    changed = True
    while changed and value:
        changed = False
        for suffix in scientific.archive_copy_suffixes:
            normalized_suffix = " ".join(suffix.casefold().split())
            if value == normalized_suffix:
                value = ""
                changed = True
                break
            if value.endswith(f" {normalized_suffix}"):
                value = value[: -(len(normalized_suffix) + 1)].strip()
                changed = True
                break
    return value


def _opening_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = _TOKEN.findall(normalized)[:120]
    if not tokens:
        return ""
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def _merge_document_aliases(
    chunks: list[RankedChunk],
    scientific: ScientificConfig,
) -> dict[str, str]:
    document_ids = sorted({chunk.document_id for chunk in chunks})
    parent = {document_id: document_id for document_id in document_ids}

    def find(document_id: str) -> str:
        while parent[document_id] != document_id:
            parent[document_id] = parent[parent[document_id]]
            document_id = parent[document_id]
        return document_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        canonical = min(left_root, right_root)
        parent[left_root] = canonical
        parent[right_root] = canonical

    by_filename: dict[str, str] = {}
    by_opening: dict[str, str] = {}
    best_by_document: dict[str, RankedChunk] = {}
    for chunk in chunks:
        current = best_by_document.get(chunk.document_id)
        if current is None or chunk.rank < current.rank:
            best_by_document[chunk.document_id] = chunk
    for document_id, chunk in sorted(best_by_document.items()):
        filename = normalized_filename(chunk, scientific)
        opening = _opening_hash(chunk.text)
        if filename:
            previous = by_filename.setdefault(filename, document_id)
            union(previous, document_id)
        if opening:
            previous = by_opening.setdefault(opening, document_id)
            union(previous, document_id)
    return {document_id: find(document_id) for document_id in document_ids}
