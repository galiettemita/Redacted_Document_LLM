"""Create stable passages and invalidation records.

Revision ID: 0002_passages
Revises: 0001_documents
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0002_passages"
down_revision: str | None = "0001_documents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "passages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("rects", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("merged_into", sa.String(length=36), nullable=True),
        sa.Column("split_from", sa.String(length=36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["merged_into"], ["passages.id"]),
        sa.ForeignKeyConstraint(["split_from"], ["passages.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_passages_document_id", "passages", ["document_id"])
    op.create_index("ix_passages_status", "passages", ["status"])
    op.create_index(
        "ix_passages_document_status",
        "passages",
        ["document_id", "status"],
    )
    op.create_table(
        "passage_invalidations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("passage_id", sa.String(length=36), nullable=False),
        sa.Column("from_revision", sa.Integer(), nullable=False),
        sa.Column("to_revision", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["passage_id"], ["passages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_passage_invalidations_passage_id",
        "passage_invalidations",
        ["passage_id"],
    )
    op.create_table(
        "idempotency_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("scope", sa.String(length=255), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("resource_id", sa.String(length=36), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scope", "key", name="uq_idempotency_scope_key"),
    )


def downgrade() -> None:
    op.drop_table("idempotency_records")
    op.drop_index(
        "ix_passage_invalidations_passage_id",
        table_name="passage_invalidations",
    )
    op.drop_table("passage_invalidations")
    op.drop_index("ix_passages_document_status", table_name="passages")
    op.drop_index("ix_passages_status", table_name="passages")
    op.drop_index("ix_passages_document_id", table_name="passages")
    op.drop_table("passages")
