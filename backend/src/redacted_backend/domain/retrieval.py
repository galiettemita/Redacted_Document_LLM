from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from redacted_backend.retrieval.types import RankedChunk


class FourRankedLists(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context_bm25: list[RankedChunk]
    context_dense: list[RankedChunk]
    own_bm25: list[RankedChunk]
    own_dense: list[RankedChunk]

    def named(self) -> tuple[tuple[str, list[RankedChunk]], ...]:
        return (
            ("context_bm25", self.context_bm25),
            ("context_dense", self.context_dense),
            ("own_bm25", self.own_bm25),
            ("own_dense", self.own_dense),
        )


class FusedChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(min_length=1)
    rrf_score: float = Field(gt=0)


class FusedDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    source_document_ids: list[str] = Field(min_length=1)
    rrf_score: float = Field(gt=0)
    chunks: list[FusedChunk] = Field(min_length=1, max_length=2)


class RetrievalTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    own_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lists: FourRankedLists
    excluded_document_ids: list[str]
    documents: list[FusedDocument] = Field(min_length=1, max_length=100)
    serialized_documents: str = Field(min_length=1)
    fusion_version: str = "document-rrf-v1"
