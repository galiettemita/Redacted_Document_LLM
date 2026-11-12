from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from redacted_backend.db.models import VisibleExampleArtifact
from redacted_backend.documents.preparation import PreparedVisibleContext
from redacted_backend.domain.visible_examples import VisibleExample


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
        )
        self._session.add(artifact)
        self._session.flush()
        self._session.refresh(artifact)
        return artifact


def artifact_example(artifact: VisibleExampleArtifact) -> VisibleExample:
    return VisibleExample.model_validate(artifact.payload)
