"""Ingest: contract sources -> ContractPrice rows, behind one interface.

Every place a contracted price can come from — a GPO table, a prose letter, an
email, and later the VA catalog and GUDID — is a ContractSource: a name plus a
load() that returns rows. corpus.build_corpus() asks for the configured set of
sources and merges them; it does not know or care which formats exist.

Why an interface before the real data exists: the next change swaps 7 hand-written
lines for ~50k real ones (docs/WORKING_STATE.md §5.3). Putting the seam in first,
with the synthetic files behind it and no behaviour change, means that swap is a
new source class, not an edit to the corpus or the agent — and a bad ingest rolls
back by changing config.CORPUS_SOURCE.

Merging is where SKU identity used to go wrong. Rows used to be keyed on SKU alone,
newest source winning. That is right for an amendment to one contract and wrong
across contracts: the same SKU on two contract vehicles is two prices, and two
vendors can reuse one part number. merge_amendments() keys on
ContractPrice.identity_key — (holder, sku, contract) — so only a genuine amendment
replaces a row.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Protocol

from .schema import ContractPrice

CONTRACTS = Path(__file__).resolve().parent.parent / "data" / "contracts"

# The synthetic files' contract vehicles. The email addendum is Cardinal amending
# its own GPO price, so it carries the GPO contract's id — which is what makes it
# replace that row instead of sitting beside it.
GPO_2025 = "GPO-OVERLAY-2025"
LOCAL_ST_MARKS = "LOCAL-ST-MARKS-ACME"


class ContractSource(Protocol):
    name: str

    def load(self) -> list[ContractPrice]:
        """Return every row this source yields, in the source's own order."""
        ...


# --- Synthetic sources: the three hand-written files ---------------------

def _parse_gpo_overlay(text: str) -> list[ContractPrice]:
    rows: list[ContractPrice] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 4 and parts[0].lower() != "vendor" and not parts[0].startswith("GPO"):
            vendor, sku, desc, price = parts
            try:
                rows.append(ContractPrice(sku=sku, description=desc, vendor=vendor,
                                          contracted_unit_price=float(price),
                                          source="GPO overlay 2025",
                                          contract_id=GPO_2025, holder=vendor,
                                          source_text=line.strip()))
            except ValueError:
                continue
    return rows


def _parse_local_agreement(text: str) -> list[ContractPrice]:
    """Prose letter: pull '(Vendor #SKU): $price' style lines."""
    rows: list[ContractPrice] = []
    pattern = re.compile(r"-\s*(.+?)\s*\((\w[\w\s]*?)\s*#([\w\-]+)\):\s*\$([\d.]+)")
    for m in pattern.finditer(text):
        desc, vendor, sku, price = m.groups()
        rows.append(ContractPrice(sku=sku.strip(), description=desc.strip(),
                                  vendor=vendor.strip(),
                                  contracted_unit_price=float(price),
                                  source="Local agreement - St. Mark's",
                                  contract_id=LOCAL_ST_MARKS, holder=vendor.strip(),
                                  source_text=m.group(0).strip()))
    return rows


def _parse_email_addendum(text: str) -> list[ContractPrice]:
    """Email that changes a price: find a SKU code + a $price near it."""
    rows: list[ContractPrice] = []
    sku_m = re.search(r"\b([A-Z]{3}-[A-Z0-9]+)\b", text)
    price_m = re.search(r"\$([\d.]+)", text)
    if sku_m and price_m:
        rows.append(ContractPrice(
            sku=sku_m.group(1),
            description="Foley Catheter 16Fr 2-way (email price update)",
            vendor="Cardinal", contracted_unit_price=float(price_m.group(1)),
            source="Email addendum 4/12",
            contract_id=GPO_2025, holder="Cardinal",
            source_text=text.strip()))
    return rows


class FileSource:
    """A source backed by one file in data/contracts and a parser for its format."""

    def __init__(self, name: str, filename: str, parser) -> None:
        self.name = name
        self.path = CONTRACTS / filename
        self._parser = parser

    def load(self) -> list[ContractPrice]:
        return self._parser(self.path.read_text())


def synthetic_sources() -> list[ContractSource]:
    """The hand-written corpus, oldest first — load order decides amendments."""
    return [
        FileSource("gpo_overlay", "gpo_overlay.txt", _parse_gpo_overlay),
        FileSource("local_agreement", "local_agreement.txt", _parse_local_agreement),
        FileSource("email_addendum", "email_addendum.txt", _parse_email_addendum),
    ]


_REGISTRY = {
    "synthetic": synthetic_sources,
}


def sources_for(name: str) -> list[ContractSource]:
    """The configured source set. An unknown name fails loudly — never falls back.

    Silently building the synthetic corpus when someone asked for real data would
    produce a scan that looks fine and measures nothing.
    """
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"Unknown corpus source {name!r}; expected one of {sorted(_REGISTRY)}"
        ) from None


# --- Merge ---------------------------------------------------------------

def merge_amendments(rows: Iterable[ContractPrice]) -> list[ContractPrice]:
    """Collapse amendments of one contract line; keep distinct contracts distinct.

    Rows arrive oldest first. A later row with the same identity_key replaces the
    earlier one in place (so corpus order, and with it retrieval tie-breaking and
    the embedding cache, is stable). Rows with different keys all survive, even
    when they share a SKU — choosing between them is a price-policy decision for
    recovery.select_contract_price(), not something to settle by load order.
    """
    merged: dict[tuple[str, str, str], ContractPrice] = {}
    for row in rows:
        merged[row.identity_key] = row
    return list(merged.values())
