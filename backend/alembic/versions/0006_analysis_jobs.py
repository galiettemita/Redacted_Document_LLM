"""Persist revision-bound analysis jobs and ordered events.

Revision ID: 0006_analysis_jobs
Revises: 0005_context_queries
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0006_analysis_jobs"
down_revision: str | None = "0005_context_queries"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("passage_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("passage_revision", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=512), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("blockers", sa.JSON(), nullable=False),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("query_payload", sa.JSON(), nullable=True),
        sa.Column("retrieval_trace", sa.JSON(), nullable=True),
        sa.Column("prediction_payload", sa.JSON(), nullable=True),
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
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["passage_id"], ["passages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "passage_id",
            "idempotency_key",
            name="uq_analysis_job_passage_idempotency",
        ),
    )
    op.create_index("ix_analysis_jobs_document_id", "analysis_jobs", ["document_id"])
    op.create_index("ix_analysis_jobs_passage_id", "analysis_jobs", ["passage_id"])
    op.create_index("ix_analysis_jobs_status", "analysis_jobs", ["status"])
    op.create_index(
        "ix_analysis_jobs_document_status",
        "analysis_jobs",
        ["document_id", "status"],
    )
    op.create_table(
        "analysis_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("passage_id", sa.String(length=36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["analysis_jobs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["passage_id"], ["passages.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_id", "sequence", name="uq_analysis_event_job_sequence"
        ),
    )
    op.create_index("ix_analysis_events_job_id", "analysis_events", ["job_id"])
    op.create_index(
        "ix_analysis_events_passage_id", "analysis_events", ["passage_id"]
    )
    op.create_index(
        "ix_analysis_events_document_id_id",
        "analysis_events",
        ["document_id", "id"],
    )


def downgrade() -> None:
    op.drop_index("ix_analysis_events_document_id_id", table_name="analysis_events")
    op.drop_index("ix_analysis_events_passage_id", table_name="analysis_events")
    op.drop_index("ix_analysis_events_job_id", table_name="analysis_events")
    op.drop_table("analysis_events")
    op.drop_index("ix_analysis_jobs_document_status", table_name="analysis_jobs")
    op.drop_index("ix_analysis_jobs_status", table_name="analysis_jobs")
    op.drop_index("ix_analysis_jobs_passage_id", table_name="analysis_jobs")
    op.drop_index("ix_analysis_jobs_document_id", table_name="analysis_jobs")
    op.drop_table("analysis_jobs")
