"""VA National Acquisition Center catalog (CCST): real federal contract prices.

The only public source of line-item contract pricing for med-surg supplies
(docs/WORKING_STATE.md §3B). It has no published API. Its public search page is a
plain GET that returns an HTML table, which this module treats as one:

    /NAC/MedSurg/List?txtCriteriams=<term>&Count=400&Page=<n>
        -> catalog number, contract number, BPA contract number, description,
           contractor, price, BPA price, SIN — 400 rows a page
    /NAC/MedSurg/Details?lognumber=<n>&type=fss
        -> the price's unit ("$115.58 / CA") and its effective/expiration dates

Two things the list page does not carry — the unit a price is quoted in and the
dates it applies — are exactly the two that decide whether a claim's dollar figure
is right (gaps G1 and G4). They come from the detail page, one request per item,
so the harvester spends a bounded detail budget and leaves the rest unknown.
Unknown is not guessed: select_contract_price() sends it to a human.

The detail page also lists named contacts with phone numbers and email addresses.
None of that is stored. parse_detail_page() extracts the four fields Backtrace
uses and the raw detail HTML is never written to disk.

Because this is a web page and not a contract-stable API, a redesign breaks the
parser. That failure is loud by construction: rows that do not parse are counted
by reason in the ingest report, never silently skipped.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode

from ..schema import ContractPrice
from .openfda import DeviceIndex, JoinStats, enrich

BASE = "https://www.vendorportal.ecms.va.gov"
LIST_PATH = "/NAC/MedSurg/List"
DETAIL_PATH = "/NAC/MedSurg/Details"
PAGE_SIZE = 400   # the largest the site offers

# Columns of the results table, in page order. Matched against the header text so
# a reordered or renamed column fails the parse instead of shifting every field.
COLUMNS = ("catalog_number", "contract_number", "bpa_contract_number", "gsa", "pv",
           "description", "contractor", "price", "bpa_price", "sin")
_HEADER_TEXT = ("catalog number", "contract number", "bpa contract number", "gsa",
                "pv", "product long description", "contractor name", "price",
                "bpa price", "sin")

# Unit-of-measure codes as the VA prints them, mapped to one vocabulary.
UOM_CODES = {
    "EA": "each", "CA": "case", "CS": "case", "BX": "box", "PK": "pack",
    "PG": "pack", "CT": "carton", "CTN": "carton", "DZ": "dozen", "BG": "bag",
    "RL": "roll", "KT": "kit", "ST": "set", "PR": "pair", "TR": "tray",
}


def list_url(term: str, page: int, count: int = PAGE_SIZE) -> str:
    query = {"txtCriteriams": term, "Count": count, "Page": page, "Search": "Search"}
    return f"{BASE}{LIST_PATH}?{urlencode(query)}"


def detail_url(lognumber: str) -> str:
    return f"{BASE}{DETAIL_PATH}?{urlencode({'lognumber': lognumber, 'type': 'fss'})}"


# --- List page ------------------------------------------------------------

class _TableParser(HTMLParser):
    """Collect the rows of the table with id="Export": cell text plus the detail link."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0            # >0 while inside the Export table
        self.headers: list[str] = []
        self.rows: list[dict] = []
        self._cell: list[str] | None = None
        self._row: list[str] | None = None
        self._link: str | None = None
        self._in_th = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            if self.depth or a.get("id") == "Export":
                self.depth += 1
            return
        if not self.depth:
            return
        if tag == "tr":
            self._row, self._link = [], None
        elif tag in ("td", "th"):
            self._cell, self._in_th = [], tag == "th"
        elif tag == "a" and self._cell is not None and "Details" in (a.get("href") or ""):
            m = re.search(r"lognumber=(\d+)", a["href"])
            self._link = m.group(1) if m else None
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if not self.depth:
            return
        if tag == "table":
            self.depth -= 1
        elif tag in ("td", "th") and self._cell is not None:
            text = " ".join("".join(self._cell).split())
            if self._in_th:
                self.headers.append(text.lower())
            elif self._row is not None:
                self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row:
            self.rows.append({"cells": self._row, "lognumber": self._link})
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


@dataclass
class ListPage:
    rows: list[dict]
    total: int | None
    has_next: bool


class PageFormatError(ValueError):
    """The page no longer looks the way this parser expects. Never swallowed."""


def parse_list_page(html: str, page: int = 1) -> ListPage:
    parser = _TableParser()
    parser.feed(html)
    if not parser.headers and "Found 0 records" in html:
        return ListPage([], 0, False)
    if tuple(parser.headers) != _HEADER_TEXT:
        raise PageFormatError(f"results table headers changed: {parser.headers}")

    rows = []
    for raw in parser.rows:
        if len(raw["cells"]) != len(COLUMNS):
            raise PageFormatError(f"row has {len(raw['cells'])} cells, expected {len(COLUMNS)}")
        row = dict(zip(COLUMNS, raw["cells"]))
        row["lognumber"] = raw["lognumber"]
        rows.append(row)

    m = re.search(r"Found\s+([\d,]+)\s+records", html)
    total = int(m.group(1).replace(",", "")) if m else None
    has_next = bool(re.search(rf"[?&;]Page={page + 1}\b", html))
    return ListPage(rows, total, has_next)


# --- Detail page ----------------------------------------------------------

def _text_lines(html: str) -> list[str]:
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    text = unescape(re.sub(r"<[^>]+>", "\n", html)).replace("\xa0", " ")
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def _after(lines: list[str], label: str, start: int = 0) -> tuple[str | None, int]:
    for i in range(start, len(lines) - 1):
        if lines[i].rstrip(":").strip().lower() == label.lower():
            return lines[i + 1], i + 1
    return None, start


def parse_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return datetime.strptime(text.strip(), "%m/%d/%Y").date()
    except ValueError:
        return None


def parse_detail_page(html: str) -> dict:
    """The price unit and the dates that price applies. Nothing else — see module doc."""
    lines = _text_lines(html)
    price_text, _ = _after(lines, "Price")
    uom = None
    if price_text:
        m = re.search(r"/\s*([A-Za-z]{1,4})\s*$", price_text)
        uom = m.group(1).upper() if m else None

    item_start, _ = _after(lines, "Date Effective")
    item_end, _ = _after(lines, "Expiration Date")
    _, at = _after(lines, "Contract Dates")
    contract_start, _ = _after(lines, "Effective", at)
    contract_end, _ = _after(lines, "Expiration", at)
    return {
        "uom": uom,
        "effective_start": (parse_date(item_start) or parse_date(contract_start)),
        "effective_end": (parse_date(item_end) or parse_date(contract_end)),
        # Contract-wide dates, kept separately so one detail page can date every
        # row on that contract — the detail budget cannot cover every item.
        "contract_start": parse_date(contract_start),
        "contract_end": parse_date(contract_end),
    }


# --- Rows -> ContractPrice ------------------------------------------------

def parse_price(text: str | None) -> float | None:
    if not text:
        return None
    m = re.fullmatch(r"\$?\s*([\d,]+(?:\.\d+)?)", text.strip())
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    return value if value > 0 else None


_PACK_IN_TEXT = [
    re.compile(r"\b(\d{1,5})\s*(?:/|per|\s)\s*(bx|box|cs|case|pk|pack|ct|ctn|carton|bg|bag)\b", re.I),
    re.compile(r"\b(bx|box|cs|case|pk|pack|ct|ctn|carton|bg|bag)\s*(?:of|/)\s*(\d{1,5})\b", re.I),
]
_PACK_WORDS = {"bx": "box", "box": "box", "cs": "case", "case": "case", "pk": "pack",
               "pack": "pack", "ct": "carton", "ctn": "carton", "carton": "carton",
               "bg": "bag", "bag": "bag"}


def pack_from_description(description: str, uom: str | None) -> int | None:
    """Units per pack stated in the description — used only when it names the SAME
    unit the price is quoted in. "100/bx" says nothing about a price per case."""
    if not uom:
        return None
    for pattern in _PACK_IN_TEXT:
        for m in pattern.finditer(description):
            a, b = m.groups()
            count, word = (a, b) if a.isdigit() else (b, a)
            if _PACK_WORDS[word.lower()] == uom and int(count) > 0:
                return int(count)
    return None


@dataclass
class LoadResult:
    rows: list[ContractPrice] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)   # dropped, by reason
    warnings: dict[str, int] = field(default_factory=dict)   # kept, but degraded
    read: int = 0

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def warn(self, reason: str) -> None:
        self.warnings[reason] = self.warnings.get(reason, 0) + 1


def rows_to_prices(raw_rows: list[dict], details: dict[str, dict],
                   snapshot: str) -> LoadResult:
    """Turn parsed list rows (+ any harvested detail fields) into ContractPrice rows.

    A row with a BPA price yields two rows: the BPA is a separate contract vehicle
    at a separate price, which is exactly the "same SKU, several contracts" case
    the price policy exists for.
    """
    out = LoadResult()
    contract_dates: dict[str, tuple] = {}
    for r in raw_rows:
        d = details.get(r.get("lognumber") or "")
        if d and (d.get("contract_start") or d.get("contract_end")):
            contract_dates.setdefault(r.get("contract_number", "").strip(),
                                      (d.get("contract_start"), d.get("contract_end")))
    for r in raw_rows:
        out.read += 1
        sku = r.get("catalog_number", "").strip()
        contract = r.get("contract_number", "").strip()
        if not sku:
            out.reject("missing_catalog_number")
            continue
        if not contract:
            out.reject("missing_contract_number")
            continue
        price = parse_price(r.get("price"))
        if price is None:
            out.reject("unparseable_or_zero_price")
            continue

        detail = details.get(r.get("lognumber") or "", {})
        code = detail.get("uom")
        uom = UOM_CODES.get(code) if code else None
        if code and uom is None:
            # Kept, with the unit left unknown: the row may still be the true
            # match, and an unknown unit already routes the claim to a human.
            out.warn(f"unknown_uom_code:{code}")
        description = r.get("description", "")
        common = dict(
            sku=sku, description=description, vendor=r.get("contractor", ""),
            holder=r.get("contractor", ""), source_text=description,
            uom=uom, units_per_pack=(1 if uom == "each" else
                                     pack_from_description(description, uom)),
            effective_start=_as_date(detail.get("effective_start")
                                     or contract_dates.get(contract, (None, None))[0]),
            effective_end=_as_date(detail.get("effective_end")
                                   or contract_dates.get(contract, (None, None))[1]),
        )
        out.rows.append(ContractPrice(
            contracted_unit_price=price, contract_id=contract,
            source=f"VA FSS {contract} (snapshot {snapshot})", **common))

        bpa, bpa_price = r.get("bpa_contract_number", "").strip(), parse_price(r.get("bpa_price"))
        if bpa and bpa_price is not None:
            out.rows.append(ContractPrice(
                contracted_unit_price=bpa_price, contract_id=bpa,
                source=f"VA BPA {bpa} (snapshot {snapshot})", **common))
    return out


def _as_date(value) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)


# --- Snapshot source -----------------------------------------------------

class VaNacSource:
    """Reads a harvested snapshot, enriched from the device registry. No network.

    Layout (written by src.sources.harvest):
        <snapshot>/va_nac/list/<term-slug>_p<n>.html   raw list pages
        <snapshot>/va_nac/details.json                  {lognumber: {uom, dates}}
        <snapshot>/openfda/devices.jsonl                trimmed registry records
    """
    name = "va_nac"

    def __init__(self, snapshot_dir: Path) -> None:
        self.root = Path(snapshot_dir)
        self.dir = self.root / "va_nac"
        self.snapshot = self.root.name
        self.result: LoadResult | None = None
        self.join: JoinStats | None = None

    def load(self) -> list[ContractPrice]:
        pages = sorted((self.dir / "list").glob("*.html"))
        if not pages:
            raise FileNotFoundError(f"no VA list pages under {self.dir / 'list'}")
        raw_rows: list[dict] = []
        for path in pages:
            page_no = int(re.search(r"_p(\d+)\.html$", path.name).group(1))
            raw_rows.extend(parse_list_page(path.read_text(encoding="utf-8"), page_no).rows)
        details_path = self.dir / "details.json"
        details = json.loads(details_path.read_text()) if details_path.exists() else {}
        self.result = rows_to_prices(raw_rows, details, self.snapshot)
        rows, self.join = enrich(self.result.rows, DeviceIndex.from_snapshot(self.root))
        return rows

    def report(self) -> dict:
        r, j = self.result, self.join
        return {"read": r.read, "kept": len(r.rows), "rejected": r.rejected,
                "warnings": r.warnings,
                "registry_join": {"joined": j.joined, "refused": j.refused} if j else None}
