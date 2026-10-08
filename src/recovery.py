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
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .schema import ContractPrice, OrderLine, ReverseMapResult

LEDGER = Path(__file__).resolve().parent.parent / "data" / "mock_systems" / "recovery_ledger.json"


def _load() -> list[dict]:
    if not LEDGER.exists():
        return []
    return json.loads(LEDGER.read_text())


_lock = threading.Lock()


def record_recovery(result: ReverseMapResult) -> dict:
    """Post a confirmed recovery to the ledger. Idempotent on order_id.

    The lock makes read-check-append atomic. Without it, a concurrent scan can
    interleave two claims for the same order between the duplicate check and the
    write, and idempotency-by-order_id silently stops holding — which is the one
    property this module exists to guarantee.
    """
    with _lock:
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
            # --- Provenance: how to reconstruct this decision under dispute ---
            "decided_by": "human" if result.human_confirmed else "agent",
            "model": result.model,
            "tier": result.tier,
            "prompt_version": result.prompt_version,
            "corpus_version": result.corpus_version,
            "candidates_considered": result.candidates_considered,
            "candidate_keys": result.candidate_keys,
            "guardrail_flags": result.guardrail_flags,
            "rationale": result.rationale,
        }
        ledger.append(entry)
        LEDGER.write_text(json.dumps(ledger, indent=2))
        return {"status": "claimed", **entry}


def total_recovered() -> float:
    return round(sum(e["recoverable"] for e in _load()), 2)


# --- Which contract price a claim uses -----------------------------------
#
# The model decides WHAT was bought. This decides WHAT IT SHOULD HAVE COST, and
# it is plain Python on purpose: a price policy has to be reproducible from the
# ledger alone. Settled in docs/WORKING_STATE.md §5.2 (Decision 1). Pure — no
# clock, no I/O — so it can be replayed like agent.decide().

CLAIMABLE = "claimable"
NEEDS_REVIEW = "needs_review"
EXPIRED = "expired"
NO_VALID_CONTRACT = "no_valid_contract"
SAVINGS_OPPORTUNITY = "savings_opportunity"   # a cheaper contract exists, but binds someone else


@dataclass
class PriceSelection:
    status: str
    chosen: Optional[ContractPrice] = None   # the price an automatic claim uses
    lowest: Optional[ContractPrice] = None   # the most the hospital may be owed
    valid: list[ContractPrice] = field(default_factory=list)
    reason: str = ""
    gap: float = 0.0   # chosen - lowest, in the unit they were compared in

    @property
    def has_additional(self) -> bool:
        """True when a lower valid price exists — a top-up for a human to approve."""
        return self.gap > 0


def _same_party(a: Optional[str], b: Optional[str]) -> bool:
    # Exact match after case and whitespace folding. Real vendor names need entity
    # resolution ("Medline Industries, LP" vs "MEDLINE IND"); until that exists a
    # near-miss must fail closed and go to a human, not fuzzy-match into a claim.
    return bool(a and b) and " ".join(a.split()).casefold() == " ".join(b.split()).casefold()


def _unit_key(row: ContractPrice):
    return (row.uom.casefold() if row.uom else None, row.units_per_pack)


def select_contract_price(order: OrderLine, rows: list[ContractPrice],
                          as_of: date) -> PriceSelection:
    """Pick the contract price for an order from every row pricing the matched product.

    1. Keep rows whose holder is bound to this sale (settled 2026-10-07):
         A  the holder SOLD it (holder == order.supplier) — it broke its own promise;
         B  the holder MADE it (holder == order.manufacturer) — the distributor that
            delivered must honour the maker's price.
       Any other holder made no promise to this seller: SAVINGS_OPPORTUNITY, never a
       claim — "buy from that holder next time". The registry manufacturer on a
       reseller's row does not count; that would claim against a party that made
       no promise.
    2. Keep rows in force on the order's effective date.
    3. If several remain, the automatic claim uses the HIGHEST: the smallest claim,
       the one a vendor cannot dispute. The gap to the lowest is reported, not
       claimed. A false claim is worse than a missed one.
    4. If the claim window has closed by `as_of`, the money is EXPIRED, not
       recoverable.

    Anything this cannot establish — an order naming neither seller nor maker, an
    unknown date against a dated contract, prices in units that cannot be compared
    — returns NEEDS_REVIEW. Unknown never becomes a claim and never becomes
    "no contract".
    """
    if not order.manufacturer and not order.supplier:
        return PriceSelection(NEEDS_REVIEW, reason="order names neither seller nor maker")

    def bound(r: ContractPrice) -> bool:
        holder = r.holder or r.vendor
        return _same_party(holder, order.supplier) or _same_party(holder, order.manufacturer)

    held = [r for r in rows if bound(r)]
    if not held:
        # Names match exactly until vendor entity resolution exists, so a variant
        # ("Medline Industries" vs "Medline") lands here too. That costs a missed
        # claim, never a false one, and it stays visible on the savings list.
        others = sorted({r.holder or r.vendor for r in rows})
        cheapest = min(rows, key=lambda r: (r.per_each_price is None,
                                            r.per_each_price or r.contracted_unit_price))
        return PriceSelection(SAVINGS_OPPORTUNITY, lowest=cheapest, valid=[],
                              reason=f"no contract held by the seller {order.supplier!r} "
                                     f"or maker {order.manufacturer!r}; held by {others}")

    if order.effective_date is None:
        if any(r.effective_start or r.effective_end for r in held):
            return PriceSelection(NEEDS_REVIEW, valid=held,
                                  reason="order has no date; contracts are dated")
        valid = held
    else:
        valid = [r for r in held if r.in_force_on(order.effective_date)]
        if not valid:
            return PriceSelection(NO_VALID_CONTRACT,
                                  reason=f"no contract in force on {order.effective_date}")

    if len(valid) == 1:
        price = lambda r: r.contracted_unit_price  # noqa: E731
    elif all(r.per_each_price is not None for r in valid):
        price = lambda r: r.per_each_price  # noqa: E731
    elif len({_unit_key(r) for r in valid}) == 1:
        price = lambda r: r.contracted_unit_price  # noqa: E731
    else:
        return PriceSelection(NEEDS_REVIEW, valid=valid,
                              reason="contract prices are in units that cannot be compared")

    chosen = max(valid, key=price)
    lowest = min(valid, key=price)
    gap = price(chosen) - price(lowest)

    if (order.effective_date and chosen.claim_window_days is not None
            and as_of > order.effective_date + timedelta(days=chosen.claim_window_days)):
        return PriceSelection(EXPIRED, chosen=chosen, lowest=lowest, valid=valid, gap=gap,
                              reason=f"claim window of {chosen.claim_window_days} days closed")

    return PriceSelection(CLAIMABLE, chosen=chosen, lowest=lowest, valid=valid, gap=gap)
