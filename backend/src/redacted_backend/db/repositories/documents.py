from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from redacted_backend.db.models import Document
from redacted_backend.domain.documents import (
    DateProvenance,
    DocumentRecord,
    DocumentStatus,
    VisibleDateEvidence,
)


class VisibleDateConflictError(ValueError):
    pass


class DocumentRepository:
    def __init__(self, session: Session):
        self._session = session

    def create(
        self,
        *,
        filename: str,
        sha256: str,
        media_type: str,
        page_count: int,
        storage_key: str,
        expires_at: datetime,
    ) -> DocumentRecord:
        document = Document(
            id=str(uuid4()),
            filename=filename,
            sha256=sha256,
            media_type=media_type,
            page_count=page_count,
            status=DocumentStatus.UPLOADED.value,
            storage_key=storage_key,
            expires_at=expires_at,
        )
        self._session.add(document)
        self._session.flush()
        self._session.refresh(document)
        return _to_record(document)

    def get(self, document_id: UUID | str) -> DocumentRecord | None:
        document = self._session.get(Document, str(document_id))
        return _to_record(document) if document is not None else None

    def get_for_update(self, document_id: UUID | str) -> DocumentRecord | None:
        document = self._session.get(
            Document,
            str(document_id),
            with_for_update=True,
        )
        return _to_record(document) if document is not None else None

    def set_detected_visible_date(
        self,
        document_id: UUID | str,
        evidence: VisibleDateEvidence,
    ) -> DocumentRecord | None:
        document = self._session.get(
            Document,
            str(document_id),
            with_for_update=True,
        )
        if document is None or document.status == DocumentStatus.DELETED.value:
            return None
        if document.visible_date is not None:
            if (
                document.visible_date != evidence.value
                or document.date_provenance != DateProvenance.VISIBLE_TEXT.value
            ):
                raise VisibleDateConflictError(
                    "Visible date evidence conflicts with the stored document date"
                )
            if document.visible_date_evidence is None:
                document.visible_date_evidence = evidence.model_dump(mode="json")
                self._session.flush()
            return _to_record(document)
        document.visible_date = evidence.value
        document.date_provenance = DateProvenance.VISIBLE_TEXT.value
        document.visible_date_evidence = evidence.model_dump(mode="json")
        self._session.flush()
        return _to_record(document)

    def mark_deleted(self, document_id: UUID | str) -> DocumentRecord | None:
        document = self._session.get(
            Document,
            str(document_id),
            with_for_update=True,
        )
        if document is None or document.status == DocumentStatus.DELETED.value:
            return None
        document.status = DocumentStatus.DELETED.value
        document.deleted_at = datetime.now(UTC)
        self._session.flush()
        return _to_record(document)

    def live_reference_count(self, storage_key: str) -> int:
        count = self._session.scalar(
            select(func.count())
            .select_from(Document)
            .where(
                Document.storage_key == storage_key,
                Document.status != DocumentStatus.DELETED.value,
            )
        )
        return int(count or 0)


def _to_record(document: Document) -> DocumentRecord:
    return DocumentRecord(
        id=document.id,
        filename=document.filename,
        sha256=document.sha256,
        media_type=document.media_type,
        page_count=document.page_count,
        status=document.status,
        storage_key=document.storage_key,
        visible_date=document.visible_date,
        date_provenance=document.date_provenance,
        visible_date_evidence=document.visible_date_evidence,
        mask_set_revision=document.mask_set_revision,
        created_at=document.created_at,
        expires_at=document.expires_at,
    )
