from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PreparedContextQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    term_count: int = Field(ge=1)
    query_version: Literal["context-query-v1"] = "context-query-v1"
    stopwords_version: str = Field(min_length=1)


class Bm25IndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index_name: str
    retrieval_profile: Literal[
        "paper_exact_v1",
        "best_available_foiarchive_v1",
    ] = "paper_exact_v1"
    opensearch_version: str
    corpus_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_definition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stopwords_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)

    def manifest_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(canonical).hexdigest()


class DenseIndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["dense-index-manifest-v1"] = "dense-index-manifest-v1"
    model_name: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    embedding_dimension: int = Field(ge=1)
    corpus_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_count: int = Field(ge=0)
    dtype: Literal["float16"]
    metric: Literal["inner_product"]
    candidate_index: Literal["faiss-hnsw-flat-v1"]
    hnsw_m: int = Field(ge=1)
    ef_construction: int = Field(ge=1)
    ef_search: int = Field(ge=1)
    candidate_count: int = Field(ge=300)
    recall_at_300: float = Field(ge=0, le=1)
    recall_sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def manifest_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class DenseShardManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shard_id: int = Field(ge=0)
    path: str = Field(min_length=1)
    start_row: int = Field(ge=0)
    end_row: int = Field(gt=0)
    chunk_count: int = Field(ge=1)
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    offsets_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_range_and_path(self) -> DenseShardManifest:
        value = PurePosixPath(self.path)
        if value.is_absolute() or len(value.parts) != 1 or value.name in {".", ".."}:
            raise ValueError("Dense shard path must be one safe relative directory name")
        if self.end_row - self.start_row != self.chunk_count:
            raise ValueError(
                "Dense shard range must be consecutive and equal its chunk count"
            )
        return self


class ShardedDenseIndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["sharded-dense-index-manifest-v1"] = (
        "sharded-dense-index-manifest-v1"
    )
    retrieval_profile: Literal["best_available_foiarchive_v1"] = (
        "best_available_foiarchive_v1"
    )
    model_name: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    embedding_dimension: int = Field(ge=1)
    corpus_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_count: int = Field(ge=1)
    shard_size_limit: int = Field(ge=1, le=1_000_000)
    shards: list[DenseShardManifest] = Field(min_length=1)
    dtype: Literal["float16"]
    metric: Literal["inner_product"]
    candidate_index: Literal["faiss-hnsw-flat-v1"]
    hnsw_m: int = Field(ge=1)
    ef_construction: int = Field(ge=1)
    ef_search: int = Field(ge=1)
    candidate_count: int = Field(ge=1)
    recall_at_300: float = Field(ge=0, le=1)
    recall_queries_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recall_query_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_consecutive_ordered_shards(self) -> ShardedDenseIndexManifest:
        expected_start = 0
        for expected_id, shard in enumerate(self.shards):
            if shard.shard_id != expected_id:
                raise ValueError("Dense shards must be ordered by consecutive shard ID")
            if shard.start_row != expected_start:
                raise ValueError("Dense shard ranges must be consecutive and ordered")
            if shard.chunk_count > self.shard_size_limit:
                raise ValueError("Dense shard exceeds the declared shard size limit")
            expected_start = shard.end_row
        if expected_start != self.chunk_count:
            raise ValueError("Dense shard ranges must cover the exact chunk count")
        return self

    def manifest_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class RankedChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: str
    ordinal: int = Field(ge=0)
    text: str
    title: str | None
    corpus: str
    source_url: str | None
    classification: str | None
    authored: str | None
    doc_lang: str | None
    rank: int = Field(ge=1)
    score: float
    retriever: Literal["bm25", "dense"]
    query_kind: Literal["context", "own"]
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
