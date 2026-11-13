from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Literal, Protocol
from urllib.parse import unquote, urlparse

from redacted_backend.domain.retrieval import FourRankedLists, RetrievalTrace
from redacted_backend.domain.visible_examples import VisibleExample
from redacted_backend.retrieval.context_query import prepare_context_query
from redacted_backend.retrieval.fusion import fuse_documents
from redacted_backend.retrieval.self_exclusion import (
    CorpusDocumentCandidate,
    SelfDocumentIdentityError,
    SelfDocumentMatcher,
    UploadIdentity,
)
from redacted_backend.retrieval.serialization import serialize_prompt_documents
from redacted_backend.retrieval.types import RankedChunk
from redacted_backend.scientific_config import ScientificConfig


class Retriever(Protocol):
    def search(
        self,
        query: str,
        limit: int,
        *,
        query_kind: Literal["context", "own"],
    ) -> list[RankedChunk]: ...


class RetrievalError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class RealisticRetrievalService:
    def __init__(
        self,
        *,
        bm25: Retriever,
        dense: Retriever,
        self_matcher: SelfDocumentMatcher | None = None,
        scientific: ScientificConfig | None = None,
    ) -> None:
        self.scientific = scientific or ScientificConfig()
        self.bm25 = bm25
        self.dense = dense
        self.self_matcher = self_matcher or SelfDocumentMatcher(self.scientific)

    def retrieve_realistic(
        self,
        example: VisibleExample,
        own_query: str,
    ) -> RetrievalTrace:
        if not own_query.strip():
            raise RetrievalError("invalid_own_query", "Own query must not be empty")
        context = prepare_context_query(example, self.scientific)
        limit = self.scientific.top_chunks_per_list
        lists = FourRankedLists(
            context_bm25=self.bm25.search(
                context.text,
                limit,
                query_kind="context",
            ),
            context_dense=self.dense.search(
                context.text,
                limit=limit,
                query_kind="context",
            ),
            own_bm25=self.bm25.search(
                own_query,
                limit,
                query_kind="own",
            ),
            own_dense=self.dense.search(
                own_query,
                limit=limit,
                query_kind="own",
            ),
        )
        try:
            decision = self.self_matcher.match(
                UploadIdentity(
                    filename=f"{example.document_id}.pdf",
                    visible_text=(
                        f"{example.before_target} {example.after_target}"
                    ).strip(),
                ),
                _corpus_candidates(lists),
            )
        except SelfDocumentIdentityError as error:
            raise RetrievalError(error.code, str(error)) from error
        excluded = set(decision.excluded_document_ids)
        filtered = FourRankedLists(
            **{
                name: [chunk for chunk in ranked if chunk.document_id not in excluded]
                for name, ranked in lists.named()
            }
        )
        documents = fuse_documents(
            filtered,
            rrf_k=self.scientific.rrf_k,
            limit=self.scientific.top_documents,
            chunks_per_document=self.scientific.chunks_per_document,
            scientific=self.scientific,
        )
        if not documents:
            raise RetrievalError(
                "no_evidence_retrieved",
                "All four real retrieval lists were empty after self-exclusion",
            )
        return RetrievalTrace(
            context_query_sha256=context.sha256,
            own_query_sha256=hashlib.sha256(own_query.encode("utf-8")).hexdigest(),
            lists=lists,
            excluded_document_ids=sorted(excluded),
            documents=documents,
            serialized_documents=serialize_prompt_documents(documents),
        )


def _corpus_candidates(lists: FourRankedLists) -> list[CorpusDocumentCandidate]:
    chunks_by_document: dict[str, list[RankedChunk]] = defaultdict(list)
    for _name, ranked in lists.named():
        for chunk in ranked:
            chunks_by_document[chunk.document_id].append(chunk)
    candidates: list[CorpusDocumentCandidate] = []
    for document_id, chunks in sorted(chunks_by_document.items()):
        ordered = sorted(chunks, key=lambda chunk: (chunk.ordinal, chunk.chunk_id))
        best = min(chunks, key=lambda chunk: (chunk.rank, chunk.chunk_id))
        filename = best.title
        if not filename and best.source_url:
            path = unquote(urlparse(best.source_url).path)
            filename = PurePosixPath(path).name or None
        candidates.append(
            CorpusDocumentCandidate(
                document_id=document_id,
                filename=filename,
                archive_identifiers=(document_id,),
                opening_text=" ".join(chunk.text for chunk in ordered[:2]),
            )
        )
    return candidates
