from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redacted_backend.retrieval.dense_profile import DenseArtifactProfile


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


class CompressedDenseShardManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["compressed-dense-shard-manifest-v1"] = (
        "compressed-dense-shard-manifest-v1"
    )
    artifact_profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_id: int = Field(ge=0)
    path: str = Field(min_length=1)
    start_row: int = Field(ge=0)
    end_row: int = Field(gt=0)
    chunk_count: int = Field(ge=1)
    embeddings_size_bytes: int = Field(ge=1)
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_index_size_bytes: int = Field(ge=1)
    candidate_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_range_and_path(self) -> CompressedDenseShardManifest:
        _require_safe_relative_name(self.path, kind="directory")
        if self.end_row - self.start_row != self.chunk_count:
            raise ValueError(
                "Compressed dense shard range must equal its chunk count"
            )
        return self


class CompressedShardedDenseIndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["sharded-dense-index-manifest-v2"] = (
        "sharded-dense-index-manifest-v2"
    )
    retrieval_profile: Literal["best_available_foiarchive_v1"] = (
        "best_available_foiarchive_v1"
    )
    model_name: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    embedding_dimension: int = Field(ge=1)
    corpus_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dense_semantics_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_count: int = Field(ge=1)
    artifact_profile: DenseArtifactProfile
    artifact_profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_offsets_path: str = Field(min_length=1)
    chunk_offsets_format: Literal["shared-chunk-offsets-v1"]
    chunk_offsets_size_bytes: int = Field(ge=1)
    chunk_offsets_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_template_path: str = Field(min_length=1)
    candidate_template_size_bytes: int = Field(ge=1)
    candidate_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_size_limit: int = Field(ge=1, le=1_000_000)
    shards: list[CompressedDenseShardManifest] = Field(min_length=1)
    dtype: Literal["float16"]
    metric: Literal["inner_product"]
    candidate_index: Literal["faiss-ivfpq-exact-rescore-v1"]
    recall_report_path: str = Field(min_length=1)
    recall_report_size_bytes: int = Field(ge=1)
    recall_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recall_at_300: float = Field(ge=0, le=1)
    recall_queries_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recall_query_count: int = Field(ge=1)
    total_artifact_bytes: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_artifact_contract(self) -> CompressedShardedDenseIndexManifest:
        profile = self.artifact_profile
        if self.artifact_profile_hash != profile.manifest_hash():
            raise ValueError("Dense artifact profile hash mismatch")
        if self.embedding_dimension != profile.embedding_dimension:
            raise ValueError("Dense artifact profile dimension mismatch")
        if self.candidate_index != profile.candidate_index:
            raise ValueError("Dense candidate index profile mismatch")
        if self.metric != profile.metric:
            raise ValueError("Dense candidate metric profile mismatch")
        if self.chunk_offsets_format != profile.metadata_lookup_version:
            raise ValueError("Dense offset format profile mismatch")
        if self.chunk_offsets_size_bytes != self.chunk_count * 8:
            raise ValueError("Dense offset byte size must equal chunk count times eight")
        if self.shard_size_limit != profile.shard_size_limit:
            raise ValueError("Dense shard size limit profile mismatch")
        for value in (
            self.chunk_offsets_path,
            self.candidate_template_path,
            self.recall_report_path,
        ):
            _require_safe_relative_name(value, kind="file")

        expected_start = 0
        for expected_id, shard in enumerate(self.shards):
            if shard.shard_id != expected_id:
                raise ValueError("Compressed dense shards need consecutive IDs")
            if shard.start_row != expected_start:
                raise ValueError("Compressed dense shard ranges must be consecutive")
            if shard.chunk_count > self.shard_size_limit:
                raise ValueError("Compressed dense shard exceeds the size limit")
            if shard.artifact_profile_hash != self.artifact_profile_hash:
                raise ValueError("Compressed dense shard profile hash mismatch")
            if shard.candidate_template_sha256 != self.candidate_template_sha256:
                raise ValueError("Compressed dense shard template hash mismatch")
            expected_embedding_bytes = (
                shard.chunk_count * self.embedding_dimension * 2
            )
            if shard.embeddings_size_bytes != expected_embedding_bytes:
                raise ValueError("Compressed dense shard embedding size mismatch")
            expected_start = shard.end_row
        if expected_start != self.chunk_count:
            raise ValueError(
                "Compressed dense shard ranges must cover the exact chunk count"
            )

        listed_bytes = (
            self.chunk_offsets_size_bytes
            + self.candidate_template_size_bytes
            + self.recall_report_size_bytes
            + sum(
                shard.embeddings_size_bytes + shard.candidate_index_size_bytes
                for shard in self.shards
            )
        )
        if self.total_artifact_bytes != listed_bytes:
            raise ValueError("Dense total artifact byte count mismatch")
        if self.total_artifact_bytes > profile.max_artifact_bytes:
            raise ValueError("Dense artifacts exceed the profile storage ceiling")
        return self

    def manifest_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class DenseFaissVerification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["dense-faiss-verification-v1"] = (
        "dense-faiss-verification-v1"
    )
    dense_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    faiss_version: str = Field(min_length=1)
    template_ntotal: Literal[0] = 0
    candidate_shard_count: int = Field(ge=1)
    candidate_row_count: int = Field(ge=1)

    def require_compatible(
        self,
        manifest: CompressedShardedDenseIndexManifest,
    ) -> DenseFaissVerification:
        if self.dense_manifest_hash != manifest.manifest_hash():
            raise ValueError("Dense FAISS verification manifest hash mismatch")
        if self.faiss_version != manifest.artifact_profile.faiss_version:
            raise ValueError("Dense FAISS verification version mismatch")
        if self.candidate_shard_count != len(manifest.shards):
            raise ValueError("Dense FAISS verification shard count mismatch")
        if self.candidate_row_count != manifest.chunk_count:
            raise ValueError("Dense FAISS verification row count mismatch")
        return self


class DenseLatencyReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["dense-latency-report-v2"] = (
        "dense-latency-report-v2"
    )
    dense_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    faiss_verification: DenseFaissVerification
    recall_queries_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=1)
    warm_search_seconds: list[float] = Field(min_length=1)
    max_observed_seconds: float = Field(gt=0)
    maximum_allowed_seconds: float = Field(gt=0)
    platform: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    runtime_environment: str = Field(min_length=1)
    worker_build_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    serving_host_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_timings(self) -> DenseLatencyReport:
        if len(self.warm_search_seconds) != self.query_count:
            raise ValueError("Dense latency query count does not match timings")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.warm_search_seconds
        ):
            raise ValueError("Dense latency timings must be finite and positive")
        if not math.isfinite(self.max_observed_seconds):
            raise ValueError("Dense latency maximum must be finite")
        if not math.isfinite(self.maximum_allowed_seconds):
            raise ValueError("Dense latency allowance must be finite")
        if max(self.warm_search_seconds) != self.max_observed_seconds:
            raise ValueError("Dense latency maximum does not match timings")
        if self.max_observed_seconds > self.maximum_allowed_seconds:
            raise ValueError("Dense latency maximum exceeds the allowed duration")
        return self

    def require_compatible(
        self,
        manifest: CompressedShardedDenseIndexManifest,
    ) -> DenseLatencyReport:
        if self.dense_manifest_hash != manifest.manifest_hash():
            raise ValueError("Dense latency manifest hash mismatch")
        if (
            self.recall_queries_sha256 != manifest.recall_queries_sha256
            or self.query_count != manifest.recall_query_count
        ):
            raise ValueError("Dense latency query identity mismatch")
        if (
            self.maximum_allowed_seconds
            != manifest.artifact_profile.max_warm_search_seconds
        ):
            raise ValueError("Dense latency profile maximum mismatch")
        self.faiss_verification.require_compatible(manifest)
        return self

    def require_runtime_compatible(
        self,
        manifest: CompressedShardedDenseIndexManifest,
        *,
        runtime_environment: str,
        platform: str,
        machine: str,
        worker_build_id: str,
        serving_host_id: str,
    ) -> DenseLatencyReport:
        self.require_compatible(manifest)
        if self.runtime_environment != runtime_environment:
            raise ValueError("Dense latency runtime environment mismatch")
        if self.platform != platform:
            raise ValueError("Dense latency platform mismatch")
        if self.machine != machine:
            raise ValueError("Dense latency machine mismatch")
        if self.worker_build_id != worker_build_id:
            raise ValueError("Dense latency worker build mismatch")
        if self.serving_host_id != serving_host_id:
            raise ValueError("Dense latency serving host mismatch")
        return self


class DenseRecallReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["dense-recall-build-report-v1"] = (
        "dense-recall-build-report-v1"
    )
    artifact_profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recall_queries_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recall_query_count: int = Field(ge=1)
    recall_at_300: float = Field(ge=0, le=1)
    per_query_recall: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_measurements(self) -> DenseRecallReport:
        if len(self.per_query_recall) != self.recall_query_count:
            raise ValueError("Dense recall query count does not match measurements")
        if any(
            not math.isfinite(value) or value < 0 or value > 1
            for value in self.per_query_recall
        ):
            raise ValueError("Dense recall measurements must be finite probabilities")
        if not math.isfinite(self.recall_at_300):
            raise ValueError("Dense recall mean must be finite")
        measured = math.fsum(self.per_query_recall) / len(self.per_query_recall)
        if not math.isclose(measured, self.recall_at_300, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Dense recall mean does not match measurements")
        return self

    def require_compatible(
        self,
        manifest: CompressedShardedDenseIndexManifest,
        *,
        minimum: float,
    ) -> DenseRecallReport:
        if self.artifact_profile_hash != manifest.artifact_profile_hash:
            raise ValueError("Dense recall artifact profile mismatch")
        if self.corpus_manifest_hash != manifest.corpus_manifest_hash:
            raise ValueError("Dense recall corpus identity mismatch")
        if (
            self.recall_queries_sha256 != manifest.recall_queries_sha256
            or self.recall_query_count != manifest.recall_query_count
        ):
            raise ValueError("Dense recall query identity mismatch")
        if not math.isclose(
            self.recall_at_300,
            manifest.recall_at_300,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Dense recall manifest mean mismatch")
        if self.recall_at_300 < minimum:
            raise ValueError("Dense recall is below the required minimum")
        return self

def _require_safe_relative_name(value: str, *, kind: Literal["directory", "file"]) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise ValueError(f"Dense artifact path must be one safe relative {kind} name")


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
