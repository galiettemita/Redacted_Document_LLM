from pydantic import BaseModel, ConfigDict

from redacted_backend.corpus.manifest import CorpusManifest
from redacted_backend.retrieval.bm25 import index_definition_hash
from redacted_backend.retrieval.types import Bm25IndexManifest
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
    return ComponentReadiness(ready=True, detail="BM25 index verified")
