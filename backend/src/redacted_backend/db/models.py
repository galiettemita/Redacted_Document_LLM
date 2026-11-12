from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from redacted_backend.db.base import Base


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_storage_key_status", "storage_key", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    media_type: Mapped[str] = mapped_column(String(100), nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    storage_key: Mapped[str] = mapped_column(String(255), nullable=False)
    visible_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    date_provenance: Mapped[str | None] = mapped_column(String(64), nullable=True)
    visible_date_evidence: Mapped[dict[str, object] | None] = mapped_column(
        JSON,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class Passage(Base):
    __tablename__ = "passages"
    __table_args__ = (Index("ix_passages_document_status", "document_id", "status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    rects: Mapped[list[dict[str, float]]] = mapped_column(JSON, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    merged_into: Mapped[str | None] = mapped_column(
        ForeignKey("passages.id"),
        nullable=True,
    )
    split_from: Mapped[str | None] = mapped_column(
        ForeignKey("passages.id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class PassageInvalidation(Base):
    __tablename__ = "passage_invalidations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    passage_id: Mapped[str] = mapped_column(
        ForeignKey("passages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    to_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class VisibleExampleArtifact(Base):
    __tablename__ = "visible_example_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "passage_id",
            "passage_revision",
            name="uq_visible_example_passage_revision",
        ),
        Index(
            "ix_visible_example_passage_active",
            "passage_id",
            "invalidated_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    passage_id: Mapped[str] = mapped_column(
        ForeignKey("passages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    passage_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    visible_word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    confirmed_mask_count: Mapped[int] = mapped_column(Integer, nullable=False)
    context_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_query_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    context_query_term_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    context_query_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    context_stopwords_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    invalidated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        UniqueConstraint("scope", "key", name="uq_idempotency_scope_key"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scope: Mapped[str] = mapped_column(String(255), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
