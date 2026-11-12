"""Persist revision-bound visible examples.

Revision ID: 0003_visible_examples
Revises: 0002_passages
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0003_visible_examples"
down_revision: str | None = "0002_passages"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "visible_example_artifacts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("passage_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("passage_revision", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("visible_word_count", sa.Integer(), nullable=False),
        sa.Column("confirmed_mask_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("invalidated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["passage_id"],
            ["passages.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "passage_id",
            "passage_revision",
            name="uq_visible_example_passage_revision",
        ),
    )
    op.create_index(
        "ix_visible_example_artifacts_document_id",
        "visible_example_artifacts",
        ["document_id"],
    )
    op.create_index(
        "ix_visible_example_artifacts_passage_id",
        "visible_example_artifacts",
        ["passage_id"],
    )
    op.create_index(
        "ix_visible_example_passage_active",
        "visible_example_artifacts",
        ["passage_id", "invalidated_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_visible_example_passage_active",
        table_name="visible_example_artifacts",
    )
    op.drop_index(
        "ix_visible_example_artifacts_passage_id",
        table_name="visible_example_artifacts",
    )
    op.drop_index(
        "ix_visible_example_artifacts_document_id",
        table_name="visible_example_artifacts",
    )
    op.drop_table("visible_example_artifacts")
