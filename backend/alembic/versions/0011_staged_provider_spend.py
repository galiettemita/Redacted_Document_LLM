"""Add the durable staged-provider spending ledger.

Revision ID: 0011_staged_provider_spend
Revises: 0010_prediction_basis
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0011_staged_provider_spend"
down_revision: str | None = "0010_prediction_basis"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_STAGED_BASIS = "best_available_foiarchive_bm25_v1"


def upgrade() -> None:
    op.create_table(
        "staged_spend_ledgers",
        sa.Column("prediction_basis", sa.String(length=64), primary_key=True),
        sa.Column(
            "committed_microusd",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "actual_microusd",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "committed_microusd >= 0",
            name="ck_staged_spend_ledgers_committed_nonnegative",
        ),
        sa.CheckConstraint(
            "actual_microusd >= 0",
            name="ck_staged_spend_ledgers_actual_nonnegative",
        ),
    )
    op.create_table(
        "staged_spend_claims",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "prediction_basis",
            sa.String(length=64),
            sa.ForeignKey(
                "staged_spend_ledgers.prediction_basis",
                ondelete="RESTRICT",
            ),
            nullable=False,
        ),
        sa.Column("scope", sa.String(length=512), nullable=False),
        sa.Column("idempotency_key", sa.String(length=512), nullable=False),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("maximum_cost_microusd", sa.BigInteger(), nullable=False),
        sa.Column("actual_cost_microusd", sa.BigInteger(), nullable=True),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "scope",
            "idempotency_key",
            name="uq_staged_spend_claim_scope_key",
        ),
        sa.CheckConstraint(
            "maximum_cost_microusd > 0",
            name="ck_staged_spend_claims_maximum_positive",
        ),
        sa.CheckConstraint(
            "actual_cost_microusd IS NULL OR actual_cost_microusd >= 0",
            name="ck_staged_spend_claims_actual_nonnegative",
        ),
    )
    ledgers = sa.table(
        "staged_spend_ledgers",
        sa.column("prediction_basis", sa.String()),
        sa.column("committed_microusd", sa.BigInteger()),
        sa.column("actual_microusd", sa.BigInteger()),
    )
    op.bulk_insert(
        ledgers,
        [
            {
                "prediction_basis": _STAGED_BASIS,
                "committed_microusd": 0,
                "actual_microusd": 0,
            }
        ],
    )


def downgrade() -> None:
    op.drop_table("staged_spend_claims")
    op.drop_table("staged_spend_ledgers")
