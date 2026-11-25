from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal

from redacted_backend.db.base import Database
from redacted_backend.db.models import StagedSpendClaim, StagedSpendLedger

_MICRO_USD = Decimal("1000000")
_PROVIDER_LEDGER_KEY = "best_available_foiarchive_bm25_v1"


class SpendBudgetError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class ProviderSpendBudget:
    def __init__(self, database: Database, *, cap_usd: Decimal) -> None:
        self.database = database
        self.cap_microusd = _microusd(cap_usd)

    @staticmethod
    def claim_id(scope: str, idempotency_key: str) -> str:
        return hashlib.sha256(
            f"{scope}\0{idempotency_key}".encode("utf-8")
        ).hexdigest()

    def reserve(
        self,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
        maximum_cost_usd: Decimal | float,
    ) -> bool:
        maximum = _microusd(maximum_cost_usd)
        claim_id = self.claim_id(scope, idempotency_key)
        now = datetime.now(UTC)
        with self.database.session_factory() as session, session.begin():
            ledger = session.get(
                StagedSpendLedger,
                _PROVIDER_LEDGER_KEY,
                with_for_update=True,
            )
            if ledger is None:
                ledger = StagedSpendLedger(
                    prediction_basis=_PROVIDER_LEDGER_KEY,
                    committed_microusd=0,
                    actual_microusd=0,
                    updated_at=now,
                )
                session.add(ledger)
                session.flush()
            existing = session.get(
                StagedSpendClaim,
                claim_id,
                with_for_update=True,
            )
            if existing is not None:
                self._require_matching(
                    existing,
                    scope=scope,
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                    maximum=maximum,
                )
                return False
            if ledger.committed_microusd + maximum > self.cap_microusd:
                raise SpendBudgetError(
                    "spend_cap_exceeded",
                    "The provider spending ceiling has been reached",
                )
            session.add(
                StagedSpendClaim(
                    id=claim_id,
                    prediction_basis=_PROVIDER_LEDGER_KEY,
                    scope=scope,
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                    maximum_cost_microusd=maximum,
                    actual_cost_microusd=None,
                    state="claimed",
                    created_at=now,
                    updated_at=now,
                )
            )
            ledger.committed_microusd += maximum
            ledger.updated_at = now
            session.flush()
            return True

    def settle(
        self,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
        actual_cost_usd: Decimal | float,
    ) -> None:
        actual = _microusd(actual_cost_usd, allow_zero=True)
        claim_id = self.claim_id(scope, idempotency_key)
        now = datetime.now(UTC)
        with self.database.session_factory() as session, session.begin():
            ledger = session.get(
                StagedSpendLedger,
                _PROVIDER_LEDGER_KEY,
                with_for_update=True,
            )
            claim = session.get(
                StagedSpendClaim,
                claim_id,
                with_for_update=True,
            )
            if ledger is None or claim is None:
                raise SpendBudgetError(
                    "spend_claim_conflict",
                    "The provider spending claim does not exist",
                )
            self._require_matching(
                claim,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                maximum=claim.maximum_cost_microusd,
            )
            if actual > claim.maximum_cost_microusd:
                raise SpendBudgetError(
                    "spend_claim_conflict",
                    "Actual provider cost exceeds its maximum reservation",
                )
            if claim.state == "settled":
                if claim.actual_cost_microusd != actual:
                    raise SpendBudgetError(
                        "spend_claim_conflict",
                        "The provider spending settlement is immutable",
                    )
                return
            ledger.committed_microusd -= claim.maximum_cost_microusd - actual
            ledger.actual_microusd += actual
            ledger.updated_at = now
            claim.actual_cost_microusd = actual
            claim.state = "settled"
            claim.updated_at = now
            session.flush()

    @staticmethod
    def _require_matching(
        claim: StagedSpendClaim,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
        maximum: int,
    ) -> None:
        if (
            claim.scope != scope
            or claim.idempotency_key != idempotency_key
            or claim.request_sha256 != request_sha256
            or claim.maximum_cost_microusd != maximum
        ):
            raise SpendBudgetError(
                "spend_claim_conflict",
                "The provider spending claim is immutable",
            )


def _microusd(value: Decimal | float, *, allow_zero: bool = False) -> int:
    amount = Decimal(str(value))
    if amount < 0 or (amount == 0 and not allow_zero):
        raise SpendBudgetError(
            "invalid_spend_amount",
            "Provider spending amounts must be positive",
        )
    return int((amount * _MICRO_USD).to_integral_value(rounding=ROUND_CEILING))


# Backward-compatible import for staged-only callers while the durable table keeps
# its original migration name. All paid prediction bases now share this one ledger.
StagedSpendBudget = ProviderSpendBudget
