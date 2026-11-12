import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ScientificConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_words: int = 300
    chunk_overlap: int = 0
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    top_chunks_per_list: int = 300
    rrf_k: int = 60
    top_documents: int = 100
    chunks_per_document: int = 2
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    model: Literal["gpt-6-astra"] = "gpt-6-astra"
    reasoning_effort: Literal["medium"] = "medium"

    def manifest_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
