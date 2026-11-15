from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import text

from redacted_backend.db.base import Database
from redacted_backend.db.repositories.documents import DocumentRepository
from redacted_backend.db.repositories.passages import PassageRepository
from redacted_backend.storage.local import LocalDocumentStore

DOCUMENT_LIFECYCLE_LOCK = threading.RLock()
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionReport:
    expired_document_ids: list[UUID]
    deleted_storage_keys: list[str]


@contextmanager
def storage_lifecycle_lock(
    database: Database,
    storage_key: str,
) -> Iterator[None]:
    """Serialize reference transitions and unlink decisions across API processes."""

    if database.engine.dialect.name != "postgresql":
        with DOCUMENT_LIFECYCLE_LOCK:
            yield
        return

    lock_key = f"document-storage:{storage_key}"
    with database.engine.connect() as connection:
        connection.execute(
            text("SELECT pg_advisory_lock(hashtext(:lock_key))"),
            {"lock_key": lock_key},
        )
        try:
            yield
        finally:
            connection.execute(
                text("SELECT pg_advisory_unlock(hashtext(:lock_key))"),
                {"lock_key": lock_key},
            )


def cleanup_unreferenced_storage(
    database: Database,
    store: LocalDocumentStore,
    storage_key: str,
) -> bool:
    """Unlink one unreferenced object and durably record cleanup completion."""

    with storage_lifecycle_lock(database, storage_key):
        with database.session_factory() as session:
            if DocumentRepository(session).live_reference_count(storage_key) > 0:
                return False
        store.delete(storage_key)
        with database.session_factory() as session, session.begin():
            DocumentRepository(session).mark_storage_deleted(storage_key)
        return True


def purge_expired_documents(
    database: Database,
    store: LocalDocumentStore,
    *,
    now: datetime | None = None,
    limit: int = 100,
) -> RetentionReport:
    cutoff = now or datetime.now(UTC)
    expired_ids: list[UUID] = []

    while True:
        with database.session_factory() as session, session.begin():
            repository = DocumentRepository(session)
            expired = repository.expired_for_update(cutoff, limit=limit)
            for record in expired:
                deleted = repository.mark_deleted(
                    record.id,
                    reason="retention_expired",
                )
                if deleted is None:
                    continue
                PassageRepository(session).delete_for_document(record.id)
                repository.delete_sensitive_artifacts(record.id)
                expired_ids.append(record.id)
        if len(expired) < limit:
            break

    deleted_storage_keys: list[str] = []
    while True:
        with database.session_factory() as session:
            pending_keys = DocumentRepository(session).pending_storage_cleanup_keys(
                limit=limit
            )
        if not pending_keys:
            break
        progress = False
        for storage_key in pending_keys:
            if cleanup_unreferenced_storage(database, store, storage_key):
                deleted_storage_keys.append(storage_key)
                progress = True
        if len(pending_keys) < limit or not progress:
            break

    if expired_ids or deleted_storage_keys:
        _LOGGER.info(
            "retention sweep expired %d documents and removed %d storage objects",
            len(expired_ids),
            len(deleted_storage_keys),
        )
    return RetentionReport(
        expired_document_ids=expired_ids,
        deleted_storage_keys=deleted_storage_keys,
    )


class RetentionScheduler:
    def __init__(
        self,
        database: Database,
        store: LocalDocumentStore,
        *,
        interval_seconds: int,
    ) -> None:
        self._database = database
        self._store = store
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            purge_expired_documents(self._database, self._store)
        except Exception:
            _LOGGER.exception("initial retention sweep failed; cleanup will retry")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="document-retention",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                purge_expired_documents(self._database, self._store)
            except Exception:
                _LOGGER.exception("retention sweep failed")
