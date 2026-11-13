from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
