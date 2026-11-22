from pydantic import BaseModel, ConfigDict

from redacted_backend.corpus.best_available import (
    BEST_AVAILABLE_BM25_INDEX_NAME,
    PRODUCTION_BEST_AVAILABLE_RETRIEVAL_CONTRACT,
    BestAvailableCorpusManifest,
    BestAvailableRetrievalContract,
)
from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.domain.analysis import PredictionBasis
from redacted_backend.retrieval.bm25 import index_definition_hash
from redacted_backend.retrieval.dense_profile import (
    DenseArtifactProfile,
    dense_semantics_hash,
)
from redacted_backend.retrieval.types import (
    Bm25IndexManifest,
    CompressedShardedDenseIndexManifest,
    DenseIndexManifest,
)
from redacted_backend.scientific_config import ScientificConfig


class ComponentReadiness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ready: bool
    detail: str


class BasisReadiness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    operational: bool
    approved: bool
    ready: bool
    detail: str


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    http: ComponentReadiness
    database: ComponentReadiness
    worker: ComponentReadiness
    corpus: ComponentReadiness
    model: ComponentReadiness
    prediction_ready: bool
    prediction_bases: dict[PredictionBasis, BasisReadiness]
    strongest_ready_basis: PredictionBasis | None


def verify_corpus_readiness(
    corpus: CorpusManifest,
    index: Bm25IndexManifest,
    dense: DenseIndexManifest,
    scientific: ScientificConfig,
) -> ComponentReadiness:
    if not corpus.coverage_ready:
        return ComponentReadiness(
            ready=False,
            detail="corpus exclusion coverage is incomplete",
        )
    if corpus.scientific_hash != scientific.manifest_hash():
        return ComponentReadiness(
            ready=False,
            detail="corpus scientific manifest mismatch",
        )
    if index.scientific_hash != scientific.manifest_hash():
        return ComponentReadiness(
            ready=False,
            detail="BM25 scientific manifest mismatch",
        )
    if index.stopwords_hash != scientific.context_stopwords_hash():
        return ComponentReadiness(
            ready=False,
            detail="BM25 stop-word manifest mismatch",
        )
    if index.index_definition_hash != index_definition_hash(scientific):
        return ComponentReadiness(
            ready=False,
            detail="BM25 index definition mismatch",
        )
    if index.corpus_manifest_hash != corpus.manifest_hash():
        return ComponentReadiness(
            ready=False,
            detail="BM25 corpus manifest mismatch",
        )
    if (
        index.document_count != corpus.document_count
        or index.chunk_count != corpus.chunk_count
    ):
        return ComponentReadiness(
            ready=False,
            detail="BM25 corpus count mismatch",
        )
    if (
        index.index_name != scientific.bm25_index_name
        or index.opensearch_version != scientific.opensearch_version
    ):
        return ComponentReadiness(
            ready=False,
            detail="BM25 runtime version mismatch",
        )
    if dense.model_name != scientific.embedding_model:
        return ComponentReadiness(ready=False, detail="dense model name mismatch")
    if dense.model_revision != scientific.embedding_model_revision:
        return ComponentReadiness(ready=False, detail="dense model revision mismatch")
    if dense.embedding_dimension != scientific.embedding_dimension:
        return ComponentReadiness(ready=False, detail="dense dimension mismatch")
    if dense.scientific_hash != scientific.manifest_hash():
        return ComponentReadiness(
            ready=False,
            detail="dense scientific manifest mismatch",
        )
    if dense.corpus_manifest_hash != corpus.manifest_hash():
        return ComponentReadiness(ready=False, detail="dense corpus manifest mismatch")
    if dense.chunks_sha256 != corpus.chunks_sha256:
        return ComponentReadiness(ready=False, detail="dense chunk checksum mismatch")
    if dense.chunk_count != corpus.chunk_count:
        return ComponentReadiness(ready=False, detail="dense corpus count mismatch")
    if (
        dense.candidate_index != scientific.dense_index_version
        or dense.hnsw_m != scientific.dense_hnsw_m
        or dense.ef_construction != scientific.dense_ef_construction
        or dense.ef_search != scientific.dense_ef_search
        or dense.candidate_count != scientific.dense_candidate_count
    ):
        return ComponentReadiness(ready=False, detail="dense index parameters mismatch")
    if dense.recall_at_300 < scientific.dense_recall_minimum:
        return ComponentReadiness(
            ready=False,
            detail=(
                "dense recall@300 is below "
                f"{scientific.dense_recall_minimum:.4f}"
            ),
        )
    return ComponentReadiness(
        ready=True,
        detail="BM25 and dense indexes verified",
    )


def verify_best_available_corpus_readiness(
    corpus: BestAvailableCorpusManifest,
    index: Bm25IndexManifest,
    dense: CompressedShardedDenseIndexManifest,
    scientific: ScientificConfig,
    *,
    contract: BestAvailableRetrievalContract = (
        PRODUCTION_BEST_AVAILABLE_RETRIEVAL_CONTRACT
    ),
) -> ComponentReadiness:
    scientific_hash = scientific.manifest_hash()
    corpus_hash = corpus.manifest_hash()
    if corpus.paper_coverage_ready:
        return ComponentReadiness(
            ready=False,
            detail="best-available corpus cannot claim exact-paper coverage",
        )
    if corpus.retrieval_profile != "best_available_foiarchive_v1":
        return ComponentReadiness(
            ready=False,
            detail="best-available corpus profile mismatch",
        )
    if corpus.scientific_hash != scientific_hash:
        return ComponentReadiness(
            ready=False,
            detail="best-available corpus scientific manifest mismatch",
        )
    if (
        corpus.manifest_hash() != contract.corpus_manifest_hash
        or corpus.chunks_sha256 != contract.chunks_sha256
        or corpus.parsed_row_count != contract.parsed_row_count
        or corpus.document_count != contract.document_count
        or corpus.chunk_count != contract.chunk_count
    ):
        return ComponentReadiness(
            ready=False,
            detail="best-available production corpus identity mismatch",
        )
    if (
        index.retrieval_profile != "best_available_foiarchive_v1"
        or index.index_name != BEST_AVAILABLE_BM25_INDEX_NAME
    ):
        return ComponentReadiness(
            ready=False,
            detail="best-available BM25 profile mismatch",
        )
    if index.scientific_hash != scientific_hash:
        return ComponentReadiness(
            ready=False,
            detail="BM25 scientific manifest mismatch",
        )
    if index.stopwords_hash != scientific.context_stopwords_hash():
        return ComponentReadiness(
            ready=False,
            detail="BM25 stop-word manifest mismatch",
        )
    if index.index_definition_hash != index_definition_hash(scientific):
        return ComponentReadiness(
            ready=False,
            detail="BM25 index definition mismatch",
        )
    if index.corpus_manifest_hash != corpus_hash:
        return ComponentReadiness(
            ready=False,
            detail="BM25 corpus manifest mismatch",
        )
    if (
        index.document_count != corpus.document_count
        or index.chunk_count != corpus.chunk_count
    ):
        return ComponentReadiness(
            ready=False,
            detail="BM25 corpus count mismatch",
        )
    if index.opensearch_version != scientific.opensearch_version:
        return ComponentReadiness(
            ready=False,
            detail="BM25 runtime version mismatch",
        )
    if dense.retrieval_profile != "best_available_foiarchive_v1":
        return ComponentReadiness(
            ready=False,
            detail="best-available dense profile mismatch",
        )
    if dense.model_name != scientific.embedding_model:
        return ComponentReadiness(ready=False, detail="dense model name mismatch")
    if dense.model_revision != scientific.embedding_model_revision:
        return ComponentReadiness(ready=False, detail="dense model revision mismatch")
    if dense.embedding_dimension != scientific.embedding_dimension:
        return ComponentReadiness(ready=False, detail="dense dimension mismatch")
    if dense.dense_semantics_hash != dense_semantics_hash(scientific):
        return ComponentReadiness(
            ready=False,
            detail="dense semantic manifest mismatch",
        )
    if dense.corpus_manifest_hash != corpus_hash:
        return ComponentReadiness(ready=False, detail="dense corpus manifest mismatch")
    if dense.chunks_sha256 != corpus.chunks_sha256:
        return ComponentReadiness(ready=False, detail="dense chunk checksum mismatch")
    if dense.chunk_count != corpus.chunk_count:
        return ComponentReadiness(ready=False, detail="dense corpus count mismatch")
    if dense.artifact_profile_hash != dense.artifact_profile.manifest_hash():
        return ComponentReadiness(
            ready=False,
            detail="dense artifact profile hash mismatch",
        )
    supported_profile = DenseArtifactProfile()
    if (
        dense.artifact_profile != supported_profile
        or dense.artifact_profile_hash != supported_profile.manifest_hash()
        or dense.candidate_index != supported_profile.candidate_index
    ):
        return ComponentReadiness(
            ready=False,
            detail="unsupported dense artifact profile",
        )
    if dense.chunk_offsets_size_bytes != dense.chunk_count * 8:
        return ComponentReadiness(ready=False, detail="dense offset size mismatch")
    expected_start = 0
    for expected_id, shard in enumerate(dense.shards):
        if (
            shard.shard_id != expected_id
            or shard.start_row != expected_start
            or shard.end_row - shard.start_row != shard.chunk_count
            or shard.chunk_count > dense.shard_size_limit
        ):
            return ComponentReadiness(ready=False, detail="dense shard range mismatch")
        if shard.artifact_profile_hash != dense.artifact_profile_hash:
            return ComponentReadiness(
                ready=False,
                detail="dense artifact profile hash mismatch",
            )
        if shard.candidate_template_sha256 != dense.candidate_template_sha256:
            return ComponentReadiness(
                ready=False,
                detail="dense candidate template mismatch",
            )
        if (
            shard.embeddings_size_bytes
            != shard.chunk_count * dense.embedding_dimension * 2
        ):
            return ComponentReadiness(
                ready=False,
                detail="dense shard embedding size mismatch",
            )
        expected_start = shard.end_row
    if expected_start != dense.chunk_count:
        return ComponentReadiness(ready=False, detail="dense shard range mismatch")
    if (
        dense.recall_query_count != contract.recall_query_count
        or dense.recall_queries_sha256 != contract.recall_queries_sha256
    ):
        return ComponentReadiness(
            ready=False,
            detail="dense recall query identity mismatch",
        )
    if dense.recall_at_300 < scientific.dense_recall_minimum:
        return ComponentReadiness(
            ready=False,
            detail=(
                "dense recall@300 is below "
                f"{scientific.dense_recall_minimum:.4f}"
            ),
        )
    return ComponentReadiness(
        ready=True,
        detail="best-available BM25 and dense indexes verified",
    )
