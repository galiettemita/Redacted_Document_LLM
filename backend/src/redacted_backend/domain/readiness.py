from pydantic import BaseModel, ConfigDict

from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.retrieval.bm25 import index_definition_hash
from redacted_backend.retrieval.types import Bm25IndexManifest, DenseIndexManifest
from redacted_backend.scientific_config import ScientificConfig


class ComponentReadiness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

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
