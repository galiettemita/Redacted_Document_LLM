from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redacted_backend.scientific_config import ScientificConfig


class DenseArtifactProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["compressed-dense-artifact-profile-v1"] = (
        "compressed-dense-artifact-profile-v1"
    )
    candidate_index: Literal["faiss-ivfpq-exact-rescore-v1"] = (
        "faiss-ivfpq-exact-rescore-v1"
    )
    faiss_version: Literal["1.16.0"] = "1.16.0"
    embedding_dimension: int = Field(default=768, ge=1)
    metric: Literal["inner_product"] = "inner_product"
    ivf_nlist: int = Field(default=4_096, ge=1)
    pq_subvectors: int = Field(default=96, ge=1)
    pq_bits: int = Field(default=8, ge=1, le=16)
    nprobe: int = Field(default=256, ge=1)
    candidate_count_per_shard: int = Field(default=4_000, ge=1)
    training_sample_count: int = Field(default=200_000, ge=1)
    training_selection_version: Literal["stable-hash-row-v1"] = (
        "stable-hash-row-v1"
    )
    shard_size_limit: int = Field(default=1_000_000, ge=1, le=1_000_000)
    metadata_lookup_version: Literal["shared-chunk-offsets-v1"] = (
        "shared-chunk-offsets-v1"
    )
    max_parallel_shards: int = Field(default=4, ge=1)
    max_artifact_bytes: int = Field(default=28 * 1024**3, ge=1)
    max_warm_search_seconds: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def validate_index_layout(self) -> DenseArtifactProfile:
        if self.embedding_dimension % self.pq_subvectors != 0:
            raise ValueError(
                "PQ subvectors must divide the embedding dimension exactly"
            )
        if self.nprobe > self.ivf_nlist:
            raise ValueError("Dense nprobe must not exceed the IVF list count")
        if self.training_sample_count < self.ivf_nlist:
            raise ValueError(
                "Dense training sample count must cover every IVF list"
            )
        return self

    def manifest_hash(self) -> str:
        return _canonical_hash(self.model_dump(mode="json"))


def dense_semantics_hash(scientific: ScientificConfig) -> str:
    return _canonical_hash(
        {
            "model_name": scientific.embedding_model,
            "model_revision": scientific.embedding_model_revision,
            "query_instruction": scientific.embedding_query_instruction,
            "normalized": True,
            "embedding_dimension": scientific.embedding_dimension,
            "metric": "inner_product",
            "result_limit": scientific.top_chunks_per_list,
            "recall_minimum": scientific.dense_recall_minimum,
        }
    )


def _canonical_hash(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
