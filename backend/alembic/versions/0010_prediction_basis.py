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
_VALID_BASES = {
    "model_only_v1",
    "best_available_foiarchive_bm25_v1",
    "best_available_foiarchive_v1",
    "paper_exact_v1",
}


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
        sa.column("job_id", sa.String()),
        sa.column("payload", sa.JSON()),
    )
    job_rows = list(
        bind.execute(sa.select(jobs.c.id, jobs.c.prediction_payload))
    )
    event_rows = list(
        bind.execute(sa.select(events.c.id, events.c.job_id, events.c.payload))
    )
    events_by_job: dict[str, list[object]] = {}
    for event in event_rows:
        events_by_job.setdefault(str(event.job_id), []).append(event)

    for row in job_rows:
        evidence = _basis_evidence(row.prediction_payload)
        for event in events_by_job.get(str(row.id), []):
            evidence.update(_basis_evidence(event.payload))
        if len(evidence) > 1:
            raise RuntimeError(
                f"conflicting prediction basis evidence for analysis job {row.id}"
            )
        basis = next(iter(evidence), _LEGACY_BASIS)
        bind.execute(
            jobs.update()
            .where(jobs.c.id == row.id)
            .values(prediction_basis=basis)
        )
        if isinstance(row.prediction_payload, dict):
            payload = dict(row.prediction_payload)
            payload.setdefault("prediction_basis", basis)
            bind.execute(
                jobs.update()
                .where(jobs.c.id == row.id)
                .values(prediction_payload=payload)
            )
        for event in events_by_job.get(str(row.id), []):
            if not isinstance(event.payload, dict):
                continue
            payload = dict(event.payload)
            payload.setdefault("prediction_basis", basis)
            bind.execute(
                events.update().where(events.c.id == event.id).values(payload=payload)
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


def _basis_evidence(payload: object) -> set[str]:
    if not isinstance(payload, dict) or "prediction_basis" not in payload:
        return set()
    basis = payload["prediction_basis"]
    if basis not in _VALID_BASES:
        raise RuntimeError(f"unknown prediction basis evidence: {basis!r}")
    return {str(basis)}
