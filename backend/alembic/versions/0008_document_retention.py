"""Record auditable document deletion reasons.

Revision ID: 0008_document_retention
Revises: 0007_document_mask_set_revision
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0008_document_retention"
down_revision: str | None = "0007_document_mask_set_revision"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("deletion_reason", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "deletion_reason")
