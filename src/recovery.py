"""Stage 3 — Recovery: dollar math + idempotent, audited ledger.

Fully implemented because the concept, not the code, is the interview point.
The output Resolvd measures is DOLLARS, so this is where the demo "pays off."

Two properties that separate this from a toy:
  - Idempotency: a recovery claim is keyed on order_id. Re-running never
    double-counts a recovery. (A double recovery claim against a vendor is worse
    than missing one.)
  - Audit trail: every claim records the matched SKU, the source document, both
    prices, and whether a human confirmed it. "Why are you clawing back $1,320
    on PO-5001?" always has a documented answer.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .schema import ReverseMapResult

LEDGER = Path(__file__).resolve().parent.parent / "data" / "mock_systems" / "recovery_ledger.json"


def _load() -> list[dict]:
    if not LEDGER.exists():
        return []
    return json.loads(LEDGER.read_text())


def record_recovery(result: ReverseMapResult) -> dict:
    """Post a confirmed recovery to the ledger. Idempotent on order_id."""
    ledger = _load()
    for entry in ledger:
        if entry["order_id"] == result.order_id:
            return {"status": "noop_already_claimed", "order_id": result.order_id}

    entry = {
        "order_id": result.order_id,
        "matched_sku": result.matched_sku,
        "matched_source": result.matched_source,
        "list_unit_price": result.list_unit_price,
        "contracted_unit_price": result.contracted_unit_price,
        "quantity": result.quantity,
        "recoverable": result.recoverable,
        "confidence": result.confidence,
        "human_confirmed": result.human_confirmed,
        "claimed_at": datetime.now(timezone.utc).isoformat(),
    }
    ledger.append(entry)
    LEDGER.write_text(json.dumps(ledger, indent=2))
    return {"status": "claimed", **entry}


def total_recovered() -> float:
    return round(sum(e["recoverable"] for e in _load()), 2)
