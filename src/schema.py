"""Schema: the data contract between every stage of Backtrace.

Fully implemented — the learning is in the reverse-map agent, not here. But know
it cold. The key modeling idea: a ContractPrice can come from ANY messy source
(GPO overlay, local agreement, email addendum), and an OrderLine often has NO
clean SKU — just a vague free-text description. The whole game is connecting the
second to the first.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ContractPrice(BaseModel):
    """One agreed price for an item, extracted from some contract source.

    Note `source` — provenance matters. When you claim a recovery against a
    vendor, you must be able to point at WHICH document set the contracted price.

    Every field below `source` is optional and defaults to "unknown". The
    hand-written synthetic corpus cannot supply most of them; real contract data
    (docs/WORKING_STATE.md §5) can, and the price policy in
    recovery.select_contract_price() treats "unknown" conservatively rather than
    guessing. None of them reaches a prompt.
    """
    sku: str
    description: str
    contracted_unit_price: float
    vendor: str
    source: str  # e.g. "GPO overlay 2025", "Local agreement - Acme", "Email addendum 4/12"

    # --- Contract identity ----------------------------------------------
    # One SKU can sit on several contracts at different prices, and two vendors
    # can reuse one part number. A row is identified by (holder, sku, contract),
    # never by SKU alone — see identity_key.
    contract_id: Optional[str] = None
    holder: Optional[str] = None     # who the price binds: the manufacturer, not
                                     # necessarily who invoiced (a distributor)
    manufacturer: Optional[str] = None   # who made it, per the device registry (GUDID);
                                         # differs from holder when a reseller holds the contract
    product_group: Optional[str] = None  # device nomenclature term (GMDN); identical
                                         # products from different sellers share it

    # --- When the price applies -----------------------------------------
    effective_start: Optional[date] = None   # None = open-ended
    effective_end: Optional[date] = None
    claim_window_days: Optional[int] = None  # how long after invoice a claim is allowed

    # --- Unit of measure ------------------------------------------------
    # $9.10 per box of 100 and $0.15 each are not comparable until normalised.
    uom: Optional[str] = None                # "each", "box", "case", ...
    units_per_pack: Optional[int] = Field(None, gt=0)

    # --- Ingest-side trust ----------------------------------------------
    source_text: Optional[str] = None        # verbatim, before any cleaning; audit only
    needs_verification: bool = False         # set by ingest-side sanitising

    @property
    def identity_key(self) -> tuple[str, str, str]:
        """(holder, sku, contract) — rows sharing a key are amendments of each other."""
        return ((self.holder or self.vendor).casefold(), self.sku,
                self.contract_id or self.source)

    @property
    def per_each_price(self) -> Optional[float]:
        """Contract price per single unit, or None when the unit is unknown.

        None is the honest answer for a box price with no pack size: dividing by a
        guess is how a $9.10 box becomes a $9.10 glove and a 100x false claim.
        """
        if self.units_per_pack:
            return self.contracted_unit_price / self.units_per_pack
        if self.uom and self.uom.casefold() in ("each", "ea"):
            return self.contracted_unit_price
        return None

    def in_force_on(self, day: date) -> bool:
        """True when `day` falls inside the effective window. Unknown bounds are open."""
        if self.effective_start and day < self.effective_start:
            return False
        if self.effective_end and day > self.effective_end:
            return False
        return True


class OrderLine(BaseModel):
    """A non-catalog order line as it appears in the PO — deliberately messy.

    Often no clean SKU, just a free-text description and the list price paid.
    """
    order_id: str
    raw_description: str
    quantity: float
    list_unit_price: float          # what the hospital actually paid (off-contract)
    sku_hint: Optional[str] = None  # sometimes a partial/garbled code is present

    # Hospitals usually buy a manufacturer's product through a distributor. The
    # contract binds the manufacturer; the claim goes to whoever invoiced.
    supplier: Optional[str] = None       # who billed the hospital — the claim's addressee
    manufacturer: Optional[str] = None   # who made it — matched against ContractPrice.holder
    effective_date: Optional[date] = None  # the date that prices this order


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
