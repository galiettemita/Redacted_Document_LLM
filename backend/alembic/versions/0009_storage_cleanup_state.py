"""Track durable completion of document byte cleanup.

Revision ID: 0009_storage_cleanup_state
Revises: 0008_document_retention
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0009_storage_cleanup_state"
down_revision: str | None = "0008_document_retention"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("storage_deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    documents = sa.table(
        "documents",
        sa.column("id", sa.String()),
        sa.column("status", sa.String()),
        sa.column("deletion_reason", sa.String()),
        sa.column("visible_date", sa.Date()),
        sa.column("date_provenance", sa.String()),
        sa.column("visible_date_evidence", sa.JSON()),
    )
    visible_artifacts = sa.table(
        "visible_example_artifacts",
        sa.column("document_id", sa.String()),
    )
    analysis_events = sa.table(
        "analysis_events",
        sa.column("document_id", sa.String()),
    )
    analysis_jobs = sa.table(
        "analysis_jobs",
        sa.column("document_id", sa.String()),
    )
    deleted_document_ids = sa.select(documents.c.id).where(
        documents.c.status == "deleted"
    )
    op.execute(
        analysis_events.delete().where(
            analysis_events.c.document_id.in_(deleted_document_ids)
        )
    )
    op.execute(
        analysis_jobs.delete().where(
            analysis_jobs.c.document_id.in_(deleted_document_ids)
        )
    )
    op.execute(
        visible_artifacts.delete().where(
            visible_artifacts.c.document_id.in_(deleted_document_ids)
        )
    )
    op.execute(
        documents.update()
        .where(documents.c.status == "deleted")
        .values(
            visible_date=None,
            date_provenance=None,
            visible_date_evidence=sa.null(),
            deletion_reason=sa.func.coalesce(
                documents.c.deletion_reason,
                "legacy_deleted",
            ),
        )
    )


def downgrade() -> None:
    op.drop_column("documents", "storage_deleted_at")
