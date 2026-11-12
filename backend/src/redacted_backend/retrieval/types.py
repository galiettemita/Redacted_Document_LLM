from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class RankedChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: str
    ordinal: int = Field(ge=0)
    text: str
    title: str | None
    source: str
    classification: str | None
    authored: str | None
    rank: int = Field(ge=1)
    score: float
    retriever: Literal["bm25", "dense"]
    query_kind: Literal["context", "own"]
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
