from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from redacted_backend.db.models import (
    Document,
    IdempotencyRecord,
    Passage,
    PassageInvalidation,
)
from redacted_backend.domain.documents import DocumentStatus
from redacted_backend.domain.passages import (
    PassageRecord,
    PassageRect,
    PassageSource,
    PassageStatus,
)


class PassageNotFoundError(LookupError):
    pass


class PassageConflictError(ValueError):
    pass


class PassagePageOutOfRangeError(ValueError):
    pass


class PassageRepository:
    def __init__(self, session: Session):
        self._session = session

    def create(
        self,
        *,
        document_id: UUID | str,
        label: str,
        page_number: int,
        rects: list[PassageRect],
        source: PassageSource = PassageSource.MANUAL,
        split_from: UUID | str | None = None,
    ) -> PassageRecord:
        document = self._session.get(Document, str(document_id))
        if document is None or document.status == DocumentStatus.DELETED.value:
            raise PassageNotFoundError("document not found")
        self._validate_page(document, page_number)
        model = Passage(
            id=str(uuid4()),
            document_id=str(document_id),
            label=label,
            page_number=page_number,
            rects=[rect.model_dump(mode="json") for rect in rects],
            source=source.value,
            status=PassageStatus.MARKED.value,
            revision=1,
            split_from=str(split_from) if split_from is not None else None,
        )
        self._session.add(model)
        self._session.flush()
        self._session.refresh(model)
        return _to_record(model)

    def get(self, passage_id: UUID | str) -> PassageRecord | None:
        model = self._session.get(Passage, str(passage_id))
        return _to_record(model) if model is not None else None

    def get_idempotent(
        self,
        *,
        scope: str,
        key: str,
        request_hash: str,
    ) -> PassageRecord | None:
        record = self._session.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.scope == scope,
                IdempotencyRecord.key == key,
            )
        )
        if record is None:
            return None
        if record.request_hash != request_hash:
            raise PassageConflictError(
                "idempotency key was already used with a different request"
            )
        passage = self._session.get(Passage, record.resource_id)
        if passage is None:
            raise PassageConflictError("idempotent resource no longer exists")
        return _to_record(passage)

    def save_idempotency(
        self,
        *,
        scope: str,
        key: str,
        request_hash: str,
        resource_id: UUID | str,
    ) -> None:
        self._session.add(
            IdempotencyRecord(
                id=str(uuid4()),
                scope=scope,
                key=key,
                request_hash=request_hash,
                resource_id=str(resource_id),
            )
        )
        self._session.flush()

    def patch(
        self,
        passage_id: UUID | str,
        *,
        label: str | None,
        page_number: int | None,
        rects: list[PassageRect] | None,
    ) -> PassageRecord:
        model = self._get_live(passage_id)
        if label is not None:
            model.label = label
        if page_number is not None:
            document = self._session.get(Document, model.document_id)
            if document is None:
                raise PassageNotFoundError("document not found")
            self._validate_page(document, page_number)
            model.page_number = page_number
        if rects is not None:
            model.rects = [rect.model_dump(mode="json") for rect in rects]
        self._invalidate(model, "passage_corrected")
        self._session.flush()
        return _to_record(model)

    def mark_deleted(self, passage_id: UUID | str) -> PassageRecord:
        model = self._get_live(passage_id)
        self._invalidate(model, "passage_deleted", deleted=True)
        self._session.flush()
        return _to_record(model)

    def merge(self, passage_ids: list[UUID]) -> tuple[PassageRecord, list[UUID]]:
        models = [self._get_live(passage_id) for passage_id in passage_ids]
        if len({model.document_id for model in models}) != 1:
            raise PassageConflictError("passages must belong to one document")
        if len({model.page_number for model in models}) != 1:
            raise PassageConflictError("passages on different pages cannot be merged")
        ordered = sorted(models, key=_document_order_key)
        survivor = ordered[0]
        survivor.rects = sorted(
            [rect for model in ordered for rect in model.rects],
            key=lambda rect: (rect["y"], rect["x"]),
        )
        self._invalidate(survivor, "passages_merged")
        merged_ids: list[UUID] = []
        for merged in ordered[1:]:
            merged.merged_into = survivor.id
            self._invalidate(merged, "merged_into_passage", deleted=True)
            merged_ids.append(UUID(merged.id))
        self._session.flush()
        return _to_record(survivor), merged_ids

    def split(
        self,
        passage_id: UUID | str,
        *,
        parts: list[tuple[str, int, list[PassageRect]]],
    ) -> list[PassageRecord]:
        original = self._get_live(passage_id)
        first_label, first_page, first_rects = parts[0]
        document = self._session.get(Document, original.document_id)
        if document is None:
            raise PassageNotFoundError("document not found")
        self._validate_page(document, first_page)
        original.label = first_label
        original.page_number = first_page
        original.rects = [rect.model_dump(mode="json") for rect in first_rects]
        self._invalidate(original, "passage_split")
        records = [_to_record(original)]
        for label, page_number, rects in parts[1:]:
            records.append(
                self.create(
                    document_id=original.document_id,
                    label=label,
                    page_number=page_number,
                    rects=rects,
                    source=PassageSource(original.source),
                    split_from=original.id,
                )
            )
        self._session.flush()
        return records

    def invalidate_document_date(self, document_id: UUID | str) -> None:
        models = self._session.scalars(
            select(Passage).where(
                Passage.document_id == str(document_id),
                Passage.status != PassageStatus.DELETED.value,
            )
        ).all()
        for model in models:
            self._invalidate(model, "document_date_changed")
        self._session.flush()

    def delete_for_document(self, document_id: UUID | str) -> None:
        models = self._session.scalars(
            select(Passage).where(
                Passage.document_id == str(document_id),
                Passage.status != PassageStatus.DELETED.value,
            )
        )
        for model in models:
            self._invalidate(model, "document_deleted", deleted=True)
        self._session.flush()

    def _get_live(self, passage_id: UUID | str) -> Passage:
        model = self._session.get(Passage, str(passage_id))
        if model is None or model.status == PassageStatus.DELETED.value:
            raise PassageNotFoundError("passage not found")
        return model

    @staticmethod
    def _validate_page(document: Document, page_number: int) -> None:
        if page_number > document.page_count:
            raise PassagePageOutOfRangeError(
                f"page {page_number} exceeds document page count {document.page_count}"
            )

    def _invalidate(
        self, model: Passage, reason: str, *, deleted: bool = False
    ) -> None:
        previous_revision = model.revision
        model.revision += 1
        model.status = (
            PassageStatus.DELETED.value if deleted else PassageStatus.MARKED.value
        )
        model.updated_at = datetime.now(UTC)
        if deleted:
            model.deleted_at = model.updated_at
        self._session.add(
            PassageInvalidation(
                id=str(uuid4()),
                passage_id=model.id,
                from_revision=previous_revision,
                to_revision=model.revision,
                reason=reason,
            )
        )


def _document_order_key(model: Passage) -> tuple[int, float, float, str]:
    first_rect = min(model.rects, key=lambda rect: (rect["y"], rect["x"]))
    return model.page_number, first_rect["y"], first_rect["x"], model.id


def _to_record(model: Passage) -> PassageRecord:
    return PassageRecord(
        id=model.id,
        document_id=model.document_id,
        label=model.label,
        page_number=model.page_number,
        rects=model.rects,
        source=model.source,
        status=model.status,
        revision=model.revision,
        merged_into=model.merged_into,
        split_from=model.split_from,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
