from __future__ import annotations

import hashlib
import importlib
import importlib.util
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import pytest

from redacted_backend.db.base import Base, Database
from redacted_backend.db.models import Document
from redacted_backend.db.repositories.documents import DocumentRepository
from redacted_backend.db.repositories.passages import (
    PassageNotFoundError,
    PassageRepository,
)
from redacted_backend.domain.passages import PassageRect
from redacted_backend.storage.local import LocalDocumentStore


def test_expiry_removes_only_unreferenced_bytes_and_records_the_reason(
    tmp_path: Path,
) -> None:
    spec = importlib.util.find_spec("redacted_backend.jobs.retention")
    assert spec is not None, "the production retention service is missing"
    retention = importlib.import_module("redacted_backend.jobs.retention")

    database = Database.connect(f"sqlite+pysqlite:///{tmp_path / 'retention.db'}")
    Base.metadata.create_all(database.engine)
    store = LocalDocumentStore(tmp_path / "documents")
    pdf_bytes = b"%PDF-shared-retention-fixture"
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    stored = store.put(BytesIO(pdf_bytes), sha256)
    now = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)

    with database.session_factory() as session, session.begin():
        repository = DocumentRepository(session)
        expired = repository.create(
            filename="expired.pdf",
            sha256=sha256,
            media_type="application/pdf",
            page_count=1,
            storage_key=stored.storage_key,
            expires_at=now - timedelta(seconds=1),
        )
        expired_model = session.get(Document, str(expired.id))
        expired_model.visible_date = date(1945, 5, 8)
        expired_model.date_provenance = "visible_text"
        expired_model.visible_date_evidence = {
            "value": "1945-05-08",
            "page_number": 1,
            "line_text": "PRIVATE DATE LINE",
            "words": [
                {
                    "text": "PRIVATE",
                    "page_number": 1,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.1,
                    "height": 0.02,
                    "block_id": 0,
                    "line_id": 0,
                    "source": "embedded",
                    "confidence": 1.0,
                }
            ],
            "parser_version": "visible-date-v1",
        }
        active = repository.create(
            filename="active.pdf",
            sha256=sha256,
            media_type="application/pdf",
            page_count=1,
            storage_key=stored.storage_key,
            expires_at=now + timedelta(hours=1),
        )

    first = retention.purge_expired_documents(database, store, now=now)

    assert first.expired_document_ids == [expired.id]
    assert first.deleted_storage_keys == []
    assert stored.local_path.is_file()
    with database.session_factory() as session:
        expired_model = session.get(Document, str(expired.id))
        active_model = session.get(Document, str(active.id))
        assert expired_model.status == "deleted"
        assert expired_model.deletion_reason == "retention_expired"
        assert expired_model.visible_date is None
        assert expired_model.date_provenance is None
        assert expired_model.visible_date_evidence is None
        assert active_model.status == "uploaded"

    with database.session_factory() as session, session.begin():
        active_model = session.get(Document, str(active.id))
        active_model.expires_at = now - timedelta(seconds=1)

    second = retention.purge_expired_documents(database, store, now=now)

    assert second.expired_document_ids == [active.id]
    assert second.deleted_storage_keys == [stored.storage_key]
    assert not stored.local_path.exists()


def test_failed_storage_cleanup_is_retried_after_the_document_is_tombstoned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retention = importlib.import_module("redacted_backend.jobs.retention")
    database = Database.connect(f"sqlite+pysqlite:///{tmp_path / 'retry.db'}")
    Base.metadata.create_all(database.engine)
    store = LocalDocumentStore(tmp_path / "retry-documents")
    pdf_bytes = b"%PDF-retry-retention-fixture"
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    stored = store.put(BytesIO(pdf_bytes), sha256)
    now = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
    with database.session_factory() as session, session.begin():
        expired = DocumentRepository(session).create(
            filename="retry.pdf",
            sha256=sha256,
            media_type="application/pdf",
            page_count=1,
            storage_key=stored.storage_key,
            expires_at=now - timedelta(seconds=1),
        )

    real_delete = store.delete
    calls = 0

    def fail_once(storage_key: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary storage failure")
        real_delete(storage_key)

    monkeypatch.setattr(store, "delete", fail_once)
    with pytest.raises(OSError, match="temporary storage failure"):
        retention.purge_expired_documents(database, store, now=now)

    with database.session_factory() as session:
        tombstone = session.get(Document, str(expired.id))
        assert tombstone.status == "deleted"
        assert tombstone.storage_deleted_at is None
    assert stored.local_path.is_file()

    retried = retention.purge_expired_documents(database, store, now=now)

    assert retried.deleted_storage_keys == [stored.storage_key]
    assert not stored.local_path.exists()
    with database.session_factory() as session:
        tombstone = session.get(Document, str(expired.id))
        assert tombstone.storage_deleted_at is not None


def test_expired_document_cannot_accept_new_passage_work(tmp_path: Path) -> None:
    database = Database.connect(f"sqlite+pysqlite:///{tmp_path / 'expiry.db'}")
    Base.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    with database.session_factory() as session, session.begin():
        expired = DocumentRepository(session).create(
            filename="expired.pdf",
            sha256="a" * 64,
            media_type="application/pdf",
            page_count=1,
            storage_key="sha256/aa/aa/expired.pdf",
            expires_at=now - timedelta(seconds=1),
        )

    with (
        database.session_factory() as session,
        session.begin(),
        pytest.raises(PassageNotFoundError),
    ):
        PassageRepository(session).create(
            document_id=expired.id,
            label="Passage 1",
            page_number=1,
            rects=[PassageRect(x=0.1, y=0.1, width=0.2, height=0.1)],
        )


def test_retention_drains_every_expired_batch_in_one_sweep(tmp_path: Path) -> None:
    retention = importlib.import_module("redacted_backend.jobs.retention")
    database = Database.connect(f"sqlite+pysqlite:///{tmp_path / 'batches.db'}")
    Base.metadata.create_all(database.engine)
    store = LocalDocumentStore(tmp_path / "batch-documents")
    now = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
    expected_ids = []
    with database.session_factory() as session, session.begin():
        repository = DocumentRepository(session)
        for index in range(5):
            pdf_bytes = f"%PDF-expired-{index}".encode()
            sha256 = hashlib.sha256(pdf_bytes).hexdigest()
            stored = store.put(BytesIO(pdf_bytes), sha256)
            record = repository.create(
                filename=f"expired-{index}.pdf",
                sha256=sha256,
                media_type="application/pdf",
                page_count=1,
                storage_key=stored.storage_key,
                expires_at=now - timedelta(seconds=index + 1),
            )
            expected_ids.append(record.id)

    report = retention.purge_expired_documents(database, store, now=now, limit=2)

    assert set(report.expired_document_ids) == set(expected_ids)
    assert len(report.deleted_storage_keys) == 5
    with database.session_factory() as session:
        assert DocumentRepository(session).expired_for_update(now, limit=10) == []
