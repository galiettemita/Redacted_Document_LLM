from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import select

from redacted_backend.config import Settings
from redacted_backend.db.models import StagedSpendClaim, StagedSpendLedger
from redacted_backend.jobs.spend_budget import StagedSpendBudget, SpendBudgetError
from tests.api.test_context import create_test_app_and_passage


def test_settings_parse_the_exact_decimal_staged_spend_cap() -> None:
    settings = Settings(staged_bm25_spend_cap_usd="10.00")

    assert settings.staged_bm25_spend_cap_usd == Decimal("10.00")


def test_budget_reserves_atomically_replays_and_rejects_cap_exhaustion(
    tmp_path: Path,
) -> None:
    app, _document, _passage = create_test_app_and_passage(tmp_path)
    budget = StagedSpendBudget(app.state.database, cap_usd=Decimal("1.00"))

    assert budget.reserve("query:passage", "job-1:query", "a" * 64, 0.60)
    assert not budget.reserve("query:passage", "job-1:query", "a" * 64, 0.60)
    with pytest.raises(SpendBudgetError, match="spend_cap_exceeded"):
        budget.reserve("query:passage", "job-2:query", "b" * 64, 0.400001)

    with app.state.database.session_factory() as session:
        ledger = session.get(
            StagedSpendLedger,
            "best_available_foiarchive_bm25_v1",
        )
        claims = session.scalars(select(StagedSpendClaim)).all()
    assert ledger is not None
    assert ledger.committed_microusd == 600_000
    assert ledger.actual_microusd == 0
    assert len(claims) == 1
    assert claims[0].state == "claimed"


def test_completed_settlement_releases_only_unused_maximum(
    tmp_path: Path,
) -> None:
    app, _document, _passage = create_test_app_and_passage(tmp_path)
    budget = StagedSpendBudget(app.state.database, cap_usd=Decimal("1.00"))
    assert budget.reserve("query:passage", "job-1:query", "a" * 64, 0.60)

    budget.settle("query:passage", "job-1:query", "a" * 64, 0.20)
    budget.settle("query:passage", "job-1:query", "a" * 64, 0.20)
    assert budget.reserve("prediction:passage", "job-1:prediction", "b" * 64, 0.80)

    with app.state.database.session_factory() as session:
        ledger = session.get(
            StagedSpendLedger,
            "best_available_foiarchive_bm25_v1",
        )
        claim = session.get(StagedSpendClaim, StagedSpendBudget.claim_id(
            "query:passage", "job-1:query"
        ))
    assert ledger is not None
    assert ledger.committed_microusd == 1_000_000
    assert ledger.actual_microusd == 200_000
    assert claim is not None
    assert claim.state == "settled"
    assert claim.actual_cost_microusd == 200_000


def test_uncertain_attempt_keeps_maximum_and_lowered_cap_fails_closed(
    tmp_path: Path,
) -> None:
    app, _document, _passage = create_test_app_and_passage(tmp_path)
    original = StagedSpendBudget(app.state.database, cap_usd=Decimal("1.00"))
    assert original.reserve("query:passage", "job-1:query", "a" * 64, 0.60)

    lowered = StagedSpendBudget(app.state.database, cap_usd=Decimal("0.50"))
    with pytest.raises(SpendBudgetError, match="spend_cap_exceeded"):
        lowered.reserve("query:passage", "job-2:query", "b" * 64, 0.01)

    with app.state.database.session_factory() as session:
        ledger = session.get(
            StagedSpendLedger,
            "best_available_foiarchive_bm25_v1",
        )
    assert ledger is not None
    assert ledger.committed_microusd == 600_000
    assert ledger.actual_microusd == 0


def test_claim_identity_and_settlement_are_immutable(tmp_path: Path) -> None:
    app, _document, _passage = create_test_app_and_passage(tmp_path)
    budget = StagedSpendBudget(app.state.database, cap_usd=Decimal("1.00"))
    assert budget.reserve("query:passage", "job-1:query", "a" * 64, 0.60)

    with pytest.raises(SpendBudgetError, match="spend_claim_conflict"):
        budget.reserve("query:passage", "job-1:query", "b" * 64, 0.60)
    with pytest.raises(SpendBudgetError, match="spend_claim_conflict"):
        budget.settle("query:passage", "job-1:query", "b" * 64, 0.20)
    budget.settle("query:passage", "job-1:query", "a" * 64, 0.20)
    with pytest.raises(SpendBudgetError, match="spend_claim_conflict"):
        budget.settle("query:passage", "job-1:query", "a" * 64, 0.21)
