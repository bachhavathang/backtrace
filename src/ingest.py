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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from .config import MAX_VENDOR_CHARS, RAW_DATA, SNAPSHOT
from .guardrails import sanitize_contract_text, sku_is_clean
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
        return self._parser(read_document(self.path))


def read_document(path: Path) -> str:
    """Decode a contract document: UTF-8 if it is, else Windows-1252.

    The hand-written files carry a Windows-1252 em dash (byte 0x97). read_text()
    with no encoding uses the platform default, so they parsed on Windows and
    crashed on Linux; CI's first run on ubuntu caught it. Real documents arrive
    in both encodings, so the fallback is deliberate, not a patch for one file.
    """
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def synthetic_sources() -> list[ContractSource]:
    """The hand-written corpus, oldest first — load order decides amendments."""
    return [
        FileSource("gpo_overlay", "gpo_overlay.txt", _parse_gpo_overlay),
        FileSource("local_agreement", "local_agreement.txt", _parse_local_agreement),
        FileSource("email_addendum", "email_addendum.txt", _parse_email_addendum),
    ]


def snapshot_dir() -> Path:
    """The snapshot to read: BACKTRACE_SNAPSHOT if set, else the newest in data/raw."""
    if SNAPSHOT:
        return Path(SNAPSHOT)
    dated = sorted(p for p in RAW_DATA.glob("*") if p.is_dir()) if RAW_DATA.exists() else []
    if not dated:
        raise FileNotFoundError(
            f"No snapshot under {RAW_DATA}. Run: python -m src.sources.harvest")
    return dated[-1]


def real_sources() -> list[ContractSource]:
    """VA contract prices, enriched from the FDA device registry, from one snapshot."""
    from .sources.va_nac import VaNacSource   # deferred: synthetic runs never need it
    return [VaNacSource(snapshot_dir())]


_REGISTRY = {
    "synthetic": synthetic_sources,
    "real": real_sources,
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


# --- Vet: contract text is untrusted --------------------------------------

def vet(row: ContractPrice) -> ContractPrice:
    """Clean a row's prompt-bound text and flag it if any of it reads as a payload.

    Description, vendor and SKU all reach the prompt, so all three are checked.
    The row is never dropped: a flagged contract may still be the true match, and
    dropping it would lose that money silently. It is marked needs_verification,
    which stops any order whose shortlist shows it from auto-claiming
    (guardrails.shortlist_flags). The verbatim text is kept in source_text.
    """
    desc = sanitize_contract_text(row.description)
    vendor = sanitize_contract_text(row.vendor, MAX_VENDOR_CHARS)
    holder = sanitize_contract_text(row.holder, MAX_VENDOR_CHARS) if row.holder else None
    sku_ok = sku_is_clean(row.sku)

    suspicious = (desc.suspicious or vendor.suspicious
                  or (holder is not None and holder.suspicious) or not sku_ok)
    return row.model_copy(update={
        "description": desc.text,
        "vendor": vendor.text,
        "holder": holder.text if holder else None,
        "source_text": row.source_text if row.source_text is not None else row.description,
        "needs_verification": row.needs_verification or suspicious,
    })


# --- Load, with a report -------------------------------------------------

@dataclass
class IngestReport:
    """What happened to every row, so nothing disappears without a count (G17).

    `sources` holds each source's own account (rows read, kept, rejected by
    reason, registry join) where the source keeps one; hand-written sources
    report only what they returned.
    """
    sources: dict[str, dict] = field(default_factory=dict)
    loaded: int = 0
    flagged: int = 0
    merged_away: int = 0
    total: int = 0

    def summary(self) -> str:
        lines = [f"{name}: {info}" for name, info in self.sources.items()]
        lines.append(f"loaded {self.loaded}, flagged {self.flagged}, "
                     f"merged away {self.merged_away} (amendments/duplicates), "
                     f"corpus {self.total}")
        return "\n".join(lines)


def load_all(sources: list[ContractSource]) -> tuple[list[ContractPrice], IngestReport]:
    report = IngestReport()
    rows: list[ContractPrice] = []
    for source in sources:
        loaded = [vet(row) for row in source.load()]
        info = source.report() if hasattr(source, "report") else {"kept": len(loaded)}
        report.sources[source.name] = info
        rows.extend(loaded)
    merged = merge_amendments(rows)
    report.loaded = len(rows)
    report.flagged = sum(r.needs_verification for r in merged)
    report.merged_away = len(rows) - len(merged)
    report.total = len(merged)
    return merged, report


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
