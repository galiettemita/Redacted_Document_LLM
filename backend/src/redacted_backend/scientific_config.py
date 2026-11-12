import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ScientificConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_words: int = 300
    chunk_overlap: int = 0
    corpus_chunker_version: str = "paragraph-sentence-word-v1"
    corpus_exclusion_version: str = "foiarchive-exclusions-v1"
    corpus_snapshot_repo: str = "HistoryLab/foiarchive"
    opensearch_version: str = "3.9.0"
    bm25_index_name: str = "foiarchive-bm25-v1"
    bm25_bulk_batch_size: int = 1_000
    context_stopwords_version: str = "visible-english-v1"
    context_stopwords: tuple[str, ...] = (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "between",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    )
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    top_chunks_per_list: int = 300
    rrf_k: int = 60
    top_documents: int = 100
    chunks_per_document: int = 2
    pdf_render_dpi: int = 144
    visible_transcript_version: str = "visible-transcript-v1"
    mask_detection_version: str = "mask-discovery-v1"
    length_cue_version: str = "visible-geometry-v1"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    model: Literal["gpt-6-astra"] = "gpt-6-astra"
    reasoning_effort: Literal["medium"] = "medium"

    def context_stopwords_hash(self) -> str:
        canonical = json.dumps(
            list(self.context_stopwords),
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def manifest_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
