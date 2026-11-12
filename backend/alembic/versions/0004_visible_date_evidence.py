"""Persist evidence for automatically detected visible document dates.

Revision ID: 0004_visible_date_evidence
Revises: 0003_visible_examples
"""

from collections.abc import Sequence
from uuid import uuid4

import sqlalchemy as sa

from alembic import op

revision: str = "0004_visible_date_evidence"
down_revision: str | None = "0003_visible_examples"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("visible_date_evidence", sa.JSON(), nullable=True),
    )
    connection = op.get_bind()
    affected_passages = connection.execute(
        sa.text(
            "SELECT id, revision FROM passages "
            "WHERE document_id IN ("
            "SELECT id FROM documents "
            "WHERE date_provenance = 'manual_visible_document'"
            ") AND status != 'deleted'"
        )
    ).mappings()
    invalidation_table = sa.table(
        "passage_invalidations",
        sa.column("id", sa.String()),
        sa.column("passage_id", sa.String()),
        sa.column("from_revision", sa.Integer()),
        sa.column("to_revision", sa.Integer()),
        sa.column("reason", sa.String()),
    )
    invalidations = [
        {
            "id": str(uuid4()),
            "passage_id": row["id"],
            "from_revision": row["revision"],
            "to_revision": row["revision"] + 1,
            "reason": "manual_visible_date_removed",
        }
        for row in affected_passages
    ]
    if invalidations:
        connection.execute(sa.insert(invalidation_table), invalidations)
    connection.execute(
        sa.text(
            "UPDATE visible_example_artifacts "
            "SET invalidated_at = CURRENT_TIMESTAMP "
            "WHERE document_id IN ("
            "SELECT id FROM documents "
            "WHERE date_provenance = 'manual_visible_document'"
            ") AND invalidated_at IS NULL"
        )
    )
    connection.execute(
        sa.text(
            "UPDATE passages SET revision = revision + 1, status = 'marked', "
            "updated_at = CURRENT_TIMESTAMP "
            "WHERE document_id IN ("
            "SELECT id FROM documents "
            "WHERE date_provenance = 'manual_visible_document'"
            ") AND status != 'deleted'"
        )
    )
    connection.execute(
        sa.text(
            "UPDATE documents SET visible_date = NULL, date_provenance = NULL "
            "WHERE date_provenance = 'manual_visible_document'"
        )
    )


def downgrade() -> None:
    op.drop_column("documents", "visible_date_evidence")
