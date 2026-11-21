"""Persist the immutable prediction basis on analysis jobs.

Revision ID: 0010_prediction_basis
Revises: 0009_storage_cleanup_state
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0010_prediction_basis"
down_revision: str | None = "0009_storage_cleanup_state"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LEGACY_BASIS = "paper_exact_v1"


def upgrade() -> None:
    op.add_column(
        "analysis_jobs",
        sa.Column("prediction_basis", sa.String(length=64), nullable=True),
    )
    bind = op.get_bind()
    jobs = sa.table(
        "analysis_jobs",
        sa.column("id", sa.String()),
        sa.column("prediction_basis", sa.String()),
        sa.column("prediction_payload", sa.JSON()),
    )
    events = sa.table(
        "analysis_events",
        sa.column("id", sa.Integer()),
        sa.column("payload", sa.JSON()),
    )
    bind.execute(jobs.update().values(prediction_basis=_LEGACY_BASIS))
    for row in bind.execute(
        sa.select(jobs.c.id, jobs.c.prediction_payload).where(
            jobs.c.prediction_payload.is_not(None)
        )
    ):
        payload = dict(row.prediction_payload)
        payload["prediction_basis"] = _LEGACY_BASIS
        bind.execute(
            jobs.update().where(jobs.c.id == row.id).values(prediction_payload=payload)
        )
    for row in bind.execute(sa.select(events.c.id, events.c.payload)):
        payload = dict(row.payload)
        payload["prediction_basis"] = _LEGACY_BASIS
        bind.execute(
            events.update().where(events.c.id == row.id).values(payload=payload)
        )
    with op.batch_alter_table("analysis_jobs") as batch:
        batch.alter_column(
            "prediction_basis",
            existing_type=sa.String(length=64),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("analysis_jobs") as batch:
        batch.drop_column("prediction_basis")
