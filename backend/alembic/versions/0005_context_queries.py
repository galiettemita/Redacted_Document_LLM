"""Persist revision-bound visible-only context queries.

Revision ID: 0005_context_queries
Revises: 0004_visible_date_evidence
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005_context_queries"
down_revision: str | None = "0004_visible_date_evidence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "visible_example_artifacts",
        sa.Column("context_query", sa.Text(), nullable=True),
    )
    op.add_column(
        "visible_example_artifacts",
        sa.Column("context_query_sha256", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "visible_example_artifacts",
        sa.Column("context_query_term_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "visible_example_artifacts",
        sa.Column("context_query_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "visible_example_artifacts",
        sa.Column("context_stopwords_version", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("visible_example_artifacts", "context_stopwords_version")
    op.drop_column("visible_example_artifacts", "context_query_version")
    op.drop_column("visible_example_artifacts", "context_query_term_count")
    op.drop_column("visible_example_artifacts", "context_query_sha256")
    op.drop_column("visible_example_artifacts", "context_query")
