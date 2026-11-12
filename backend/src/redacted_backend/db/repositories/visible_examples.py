from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from redacted_backend.db.models import VisibleExampleArtifact
from redacted_backend.documents.preparation import PreparedVisibleContext
from redacted_backend.domain.visible_examples import VisibleExample
from redacted_backend.retrieval.types import PreparedContextQuery


class ContextQueryArtifactConflictError(ValueError):
    pass


class VisibleExampleRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_current(
        self,
        passage_id: UUID | str,
        passage_revision: int,
    ) -> VisibleExampleArtifact | None:
        return self._session.scalar(
            select(VisibleExampleArtifact).where(
                VisibleExampleArtifact.passage_id == str(passage_id),
                VisibleExampleArtifact.passage_revision == passage_revision,
                VisibleExampleArtifact.invalidated_at.is_(None),
            )
        )

    def save(
        self,
        prepared: PreparedVisibleContext,
        context_query: PreparedContextQuery,
        *,
        passage_revision: int,
    ) -> VisibleExampleArtifact:
        artifact = VisibleExampleArtifact(
            id=str(uuid4()),
            passage_id=str(prepared.example.passage_id),
            document_id=str(prepared.example.document_id),
            passage_revision=passage_revision,
            payload=prepared.example.model_dump(mode="json"),
            visible_word_count=prepared.visible_word_count,
            confirmed_mask_count=prepared.confirmed_mask_count,
            context_query=context_query.text,
            context_query_sha256=context_query.sha256,
            context_query_term_count=context_query.term_count,
            context_query_version=context_query.query_version,
            context_stopwords_version=context_query.stopwords_version,
        )
        self._session.add(artifact)
        self._session.flush()
        self._session.refresh(artifact)
        return artifact

    def ensure_context_query(
        self,
        artifact: VisibleExampleArtifact,
        prepared: PreparedContextQuery,
    ) -> VisibleExampleArtifact:
        current = (
            artifact.context_query,
            artifact.context_query_sha256,
            artifact.context_query_term_count,
            artifact.context_query_version,
            artifact.context_stopwords_version,
        )
        expected = (
            prepared.text,
            prepared.sha256,
            prepared.term_count,
            prepared.query_version,
            prepared.stopwords_version,
        )
        if all(value is None for value in current):
            (
                artifact.context_query,
                artifact.context_query_sha256,
                artifact.context_query_term_count,
                artifact.context_query_version,
                artifact.context_stopwords_version,
            ) = expected
            self._session.flush()
            return artifact
        if any(value is None for value in current):
            raise ContextQueryArtifactConflictError(
                "stored context query metadata is incomplete"
            )
        if current != expected:
            raise ContextQueryArtifactConflictError(
                "stored context query does not match the visible example"
            )
        return artifact


def artifact_example(artifact: VisibleExampleArtifact) -> VisibleExample:
    return VisibleExample.model_validate(artifact.payload)
