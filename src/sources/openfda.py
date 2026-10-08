"""openFDA device registry (GUDID): who makes a product, and how it is packed.

A real, documented API (api.fda.gov/device/udi.json). It supplies what a contract
catalog does not:

  company_name        the manufacturer — distinct from the contract holder when a
                      reseller holds the contract (gap G5)
  identifiers         the packaging hierarchy, e.g. 100 gloves per inner carton,
                      10 cartons per case — what turns "$115.58 / CA" into a
                      per-glove price (gap G1)
  gmdn_terms          a device nomenclature term; different sellers' identical
                      products share it, which is what product grouping and the
                      eval's equivalence sets need (gaps G6, G11)

The join to a contract row is the risky part (gap G7). openFDA matches part
numbers as words, so "5501" also returns "CI-5501-150" from an unrelated maker.
A wrong join puts a stranger's pack size into a dollar figure, so the index
accepts a join only when ALL of these hold, and counts why it refused otherwise:

  - the normalised part number matches exactly,
  - every exact match belongs to ONE company,
  - the registry description shares product vocabulary with the contract's.

Snapshots store a trimmed record. customer_contacts (names, phones, emails) is
dropped at harvest and never written to disk.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..schema import ContractPrice

API = "https://api.fda.gov/device/udi.json"

KEEP_FIELDS = ("public_device_record_key", "brand_name", "company_name", "catalog_number",
               "version_or_model_number", "device_description",
               "device_count_in_base_package", "identifiers", "gmdn_terms")

# How each unit word appears in GUDID package_type text.
_PACKAGE_WORDS = {
    "case": ("CASE", "SHIPPER"), "box": ("BOX",), "carton": ("CARTON", "CTN"),
    "pack": ("PACK",), "bag": ("BAG",), "tray": ("TRAY",), "kit": ("KIT",),
}

# Join thresholds. Both were set by a false join in the first live harvest: part
# "150" matched one other company exactly, and the two descriptions shared only
# "catheter". A join needs a distinctive part number AND two product words.
MIN_PART_CHARS = 5
MIN_SHARED_WORDS = 2

_STOP = {"and", "the", "with", "for", "of", "non", "use", "single", "each", "size",
         "pack", "box", "case", "per", "ea", "bx", "cs", "pk", "ct"}


def trim(record: dict) -> dict:
    """Keep only what Backtrace uses. Drops customer_contacts and everything else."""
    out = {k: record.get(k) for k in KEEP_FIELDS}
    out["gmdn_terms"] = [g.get("name") for g in record.get("gmdn_terms") or [] if g.get("name")]
    return out


def normalise_part(part: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (part or "").upper())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower()) if w not in _STOP}


def units_per_package(device: dict, unit: str) -> int | None:
    """Single items in one `unit` (case, box, ...) of this device, or None if unclear.

    Walks GUDID's package chain — each package says which identifier it contains
    and how many — so a case of 10 cartons of 100 is 1,000. Returns None unless
    exactly one package level names the unit: two "box" levels, or none, is a
    question for a human, not a guess.
    """
    words = _PACKAGE_WORDS.get(unit)
    if not words:
        return None
    try:
        base = int(device.get("device_count_in_base_package") or 1)
    except (TypeError, ValueError):
        return None
    ids = device.get("identifiers") or []
    primary = {i["id"] for i in ids if i.get("type") == "Primary"}
    packages = {i["id"]: i for i in ids if i.get("type") == "Package"}

    def units(di: str, seen: frozenset = frozenset()) -> int | None:
        if di in primary:
            return base
        pkg = packages.get(di)
        if pkg is None or di in seen:
            return None
        inner = units(pkg.get("unit_of_use_id", ""), seen | {di})
        try:
            qty = int(pkg.get("quantity_per_package") or 0)
        except ValueError:
            return None
        return inner * qty if inner and qty > 0 else None

    matches = [di for di, p in packages.items()
               if any(w in (p.get("package_type") or "").upper() for w in words)]
    if len(matches) != 1:
        return None
    return units(matches[0])


@dataclass
class JoinStats:
    joined: int = 0
    refused: dict[str, int] = field(default_factory=dict)

    def refuse(self, reason: str) -> None:
        self.refused[reason] = self.refused.get(reason, 0) + 1


class DeviceIndex:
    def __init__(self, devices: list[dict]) -> None:
        self.by_part: dict[str, list[dict]] = {}
        for d in devices:
            for part in {normalise_part(d.get("catalog_number")),
                         normalise_part(d.get("version_or_model_number"))}:
                if part:
                    self.by_part.setdefault(part, []).append(d)

    @classmethod
    def from_snapshot(cls, snapshot_dir: Path) -> "DeviceIndex":
        path = Path(snapshot_dir) / "openfda" / "devices.jsonl"
        if not path.exists():
            return cls([])
        return cls([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()])

    def match(self, row: ContractPrice, stats: JoinStats) -> dict | None:
        part = normalise_part(row.sku)
        if len(part) < MIN_PART_CHARS:
            # "150" is somebody's part number at dozens of companies. The first live
            # harvest joined a Foley catheter to a sclerotherapy catheter this way.
            stats.refuse("part_number_too_generic")
            return None
        found = self.by_part.get(part, [])
        if not found:
            stats.refuse("no_exact_part_match")
            return None
        companies = {(d.get("company_name") or "").strip().upper() for d in found}
        if len(companies) != 1:
            stats.refuse("part_number_shared_by_several_makers")
            return None
        device = found[0]
        registry_text = f"{device.get('brand_name')} {device.get('device_description')}"
        if len(_words(row.description) & _words(registry_text)) < MIN_SHARED_WORDS:
            stats.refuse("descriptions_share_no_product_words")
            return None
        stats.joined += 1
        return device


def enrich(rows: list[ContractPrice], index: DeviceIndex) -> tuple[list[ContractPrice], JoinStats]:
    """Add manufacturer, product group and — where the unit is known — pack size."""
    stats = JoinStats()
    out = []
    for row in rows:
        device = index.match(row, stats)
        if device is None:
            out.append(row)
            continue
        update = {"manufacturer": device.get("company_name"),
                  "product_group": (device.get("gmdn_terms") or [None])[0]}
        if row.uom and row.units_per_pack is None and row.uom != "each":
            update["units_per_pack"] = units_per_package(device, row.uom)
        out.append(row.model_copy(update=update))
    return out, stats
