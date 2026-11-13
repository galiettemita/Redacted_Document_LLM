"""Track document mask-set changes across visible-context preparation.

Revision ID: 0007_document_mask_set_revision
Revises: 0006_analysis_jobs
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0007_document_mask_set_revision"
down_revision: str | None = "0006_analysis_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "mask_set_revision",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("documents", "mask_set_revision")
