"""Schema: the data contract between every stage of Backtrace.

Fully implemented — the learning is in the reverse-map agent, not here. But know
it cold. The key modeling idea: a ContractPrice can come from ANY messy source
(GPO overlay, local agreement, email addendum), and an OrderLine often has NO
clean SKU — just a vague free-text description. The whole game is connecting the
second to the first.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ContractPrice(BaseModel):
    """One agreed price for an item, extracted from some contract source.

    Note `source` — provenance matters. When you claim a recovery against a
    vendor, you must be able to point at WHICH document set the contracted price.
    """
    sku: str
    description: str
    contracted_unit_price: float
    vendor: str
    source: str  # e.g. "GPO overlay 2025", "Local agreement - Acme", "Email addendum 4/12"


class OrderLine(BaseModel):
    """A non-catalog order line as it appears in the PO — deliberately messy.

    Often no clean SKU, just a free-text description and the list price paid.
    """
    order_id: str
    raw_description: str
    quantity: float
    list_unit_price: float          # what the hospital actually paid (off-contract)
    sku_hint: Optional[str] = None  # sometimes a partial/garbled code is present


class MatchDecision(str, Enum):
    MATCH = "match"                 # confident this order == a contracted item
    UNCERTAIN = "uncertain"         # plausible, needs a human to confirm
    NO_MATCH = "no_match"           # nothing in the corpus fits


class ReverseMapResult(BaseModel):
    """The agent's output for one non-catalog order line."""
    order_id: str
    decision: MatchDecision
    matched_sku: Optional[str] = None
    matched_source: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    rationale: str = ""
    # Filled by the recovery stage when there's a confirmed match.
    list_unit_price: Optional[float] = None
    contracted_unit_price: Optional[float] = None
    quantity: Optional[float] = None
    human_confirmed: Optional[bool] = None

    # --- Provenance ------------------------------------------------------
    # A recovery claim is a financial assertion against a vendor. When it is
    # disputed, "the model said so" is not an answer — you have to be able to
    # name which model, which prompt, and which contract lines were on the table
    # at the moment the decision was made. These fields are written straight
    # through to the ledger by recovery.record_recovery().
    model: Optional[str] = None
    tier: Optional[str] = None
    prompt_version: Optional[str] = None
    corpus_version: Optional[str] = None
    candidates_considered: list[str] = Field(default_factory=list)
    guardrail_flags: list[str] = Field(default_factory=list)
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None

    @property
    def needs_human_review(self) -> bool:
        """UNCERTAIN lines that no human has ruled on yet."""
        return self.decision == MatchDecision.UNCERTAIN and self.human_confirmed is None

    @property
    def blocked_by_guardrail(self) -> bool:
        """True when a guardrail (not the model's own judgement) forced escalation.

        Separates 'the agent was genuinely unsure' from 'something went wrong' —
        a transport failure and a near-duplicate glove contract both land in
        UNCERTAIN, but they need different follow-up.
        """
        from .guardrails import ESCALATING_FLAGS
        return any(f in ESCALATING_FLAGS for f in self.guardrail_flags)

    @property
    def recoverable(self) -> float:
        """Dollars recoverable = (list - contracted) * qty, if we have a match."""
        if (self.list_unit_price is None or self.contracted_unit_price is None
                or self.quantity is None):
            return 0.0
        delta = self.list_unit_price - self.contracted_unit_price
        return round(max(delta, 0.0) * self.quantity, 2)


class CandidateMatch(BaseModel):
    """A retrieval candidate: a contract price + how similar it looked. Feeds the agent."""
    contract: ContractPrice
    similarity: float  # retrieval score, 0..1 — NOT the final confidence
