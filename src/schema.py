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
