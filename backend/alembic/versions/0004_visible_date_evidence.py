"""Persist evidence for automatically detected visible document dates.

Revision ID: 0004_visible_date_evidence
Revises: 0003_visible_examples
"""

from collections.abc import Sequence

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
    manual_documents = sa.text(
        "SELECT id FROM documents "
        "WHERE date_provenance = 'manual_visible_document'"
    )
    ids = [row[0] for row in connection.execute(manual_documents)]
    if not ids:
        return

    parameters = {f"document_{index}": value for index, value in enumerate(ids)}
    placeholders = ", ".join(f":document_{index}" for index in range(len(ids)))
    connection.execute(
        sa.text(
            "UPDATE visible_example_artifacts "
            "SET invalidated_at = CURRENT_TIMESTAMP "
            f"WHERE document_id IN ({placeholders}) AND invalidated_at IS NULL"
        ),
        parameters,
    )
    connection.execute(
        sa.text(
            "UPDATE passages SET revision = revision + 1, status = 'marked', "
            "updated_at = CURRENT_TIMESTAMP "
            f"WHERE document_id IN ({placeholders}) AND status != 'deleted'"
        ),
        parameters,
    )
    connection.execute(
        sa.text(
            "UPDATE documents SET visible_date = NULL, date_provenance = NULL "
            f"WHERE id IN ({placeholders})"
        ),
        parameters,
    )


def downgrade() -> None:
    op.drop_column("documents", "visible_date_evidence")
