from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import delete, exists, func, select, update
from sqlalchemy.orm import Session, aliased

from redacted_backend.db.models import (
    AnalysisEvent,
    AnalysisJob,
    Document,
    VisibleExampleArtifact,
)
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

    def get_live(
        self,
        document_id: UUID | str,
        *,
        now: datetime | None = None,
    ) -> DocumentRecord | None:
        document = self._session.scalar(
            select(Document).where(
                Document.id == str(document_id),
                Document.status != DocumentStatus.DELETED.value,
                Document.expires_at > (now or datetime.now(UTC)),
            )
        )
        return _to_record(document) if document is not None else None

    def get_for_update(self, document_id: UUID | str) -> DocumentRecord | None:
        document = self._session.get(
            Document,
            str(document_id),
            with_for_update=True,
        )
        return _to_record(document) if document is not None else None

    def get_live_for_update(
        self,
        document_id: UUID | str,
        *,
        now: datetime | None = None,
    ) -> DocumentRecord | None:
        document = self._session.scalar(
            select(Document)
            .where(
                Document.id == str(document_id),
                Document.status != DocumentStatus.DELETED.value,
                Document.expires_at > (now or datetime.now(UTC)),
            )
            .with_for_update()
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
        if (
            document is None
            or document.status == DocumentStatus.DELETED.value
            or _as_utc(document.expires_at) <= datetime.now(UTC)
        ):
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

    def mark_deleted(
        self,
        document_id: UUID | str,
        *,
        reason: str = "user_deleted",
    ) -> DocumentRecord | None:
        document = self._session.get(
            Document,
            str(document_id),
            with_for_update=True,
        )
        if document is None:
            return None
        if document.status == DocumentStatus.DELETED.value:
            return _to_record(document)
        document.status = DocumentStatus.DELETED.value
        document.deleted_at = datetime.now(UTC)
        document.deletion_reason = reason
        document.visible_date = None
        document.date_provenance = None
        document.visible_date_evidence = None
        self._session.flush()
        return _to_record(document)

    def expired_for_update(
        self,
        now: datetime,
        *,
        limit: int = 100,
    ) -> list[DocumentRecord]:
        documents = self._session.scalars(
            select(Document)
            .where(
                Document.status != DocumentStatus.DELETED.value,
                Document.expires_at <= now,
            )
            .order_by(Document.expires_at, Document.id)
            .limit(limit)
            .with_for_update()
        ).all()
        return [_to_record(document) for document in documents]

    def delete_sensitive_artifacts(self, document_id: UUID | str) -> None:
        identifier = str(document_id)
        self._session.execute(
            delete(AnalysisEvent).where(AnalysisEvent.document_id == identifier)
        )
        self._session.execute(
            delete(AnalysisJob).where(AnalysisJob.document_id == identifier)
        )
        self._session.execute(
            delete(VisibleExampleArtifact).where(
                VisibleExampleArtifact.document_id == identifier
            )
        )
        self._session.flush()

    def pending_storage_cleanup_keys(self, *, limit: int = 100) -> list[str]:
        live_document = aliased(Document)
        keys = self._session.scalars(
            select(Document.storage_key)
            .where(
                Document.status == DocumentStatus.DELETED.value,
                Document.storage_deleted_at.is_(None),
                ~exists(
                    select(live_document.id).where(
                        live_document.storage_key == Document.storage_key,
                        live_document.status != DocumentStatus.DELETED.value,
                    )
                ),
            )
            .distinct()
            .order_by(Document.storage_key)
            .limit(limit)
        ).all()
        return list(keys)

    def mark_storage_deleted(
        self,
        storage_key: str,
        *,
        deleted_at: datetime | None = None,
    ) -> None:
        self._session.execute(
            update(Document)
            .where(
                Document.storage_key == storage_key,
                Document.status == DocumentStatus.DELETED.value,
                Document.storage_deleted_at.is_(None),
            )
            .values(storage_deleted_at=deleted_at or datetime.now(UTC))
        )
        self._session.flush()

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
        deletion_reason=document.deletion_reason,
        storage_deleted_at=document.storage_deleted_at,
    )


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
