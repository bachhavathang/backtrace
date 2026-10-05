"""Harvest a dated snapshot of real contract and device data. The only network code.

    python -m src.sources.harvest                       # default categories
    python -m src.sources.harvest --terms "foley catheter" --max-pages 2 --details 50

Writes data/raw/<YYYY-MM-DD>/ (gitignored):

    va_nac/list/<term>_p<n>.html   raw VA list pages — re-parseable if the parser changes
    va_nac/details.json            unit + dates per item; never contacts, never raw HTML
    openfda/devices.jsonl          trimmed registry records for the harvested part numbers
    manifest.json                  every URL fetched, when, and the sha256 of what was saved

Polite by default: one request at a time, a pause between them, an honest
User-Agent, bounded pages and a bounded detail budget. openFDA allows more with a
free key (OPENFDA_API_KEY); without one it is limited per day, which a default
harvest stays inside.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from . import openfda, va_nac

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
USER_AGENT = "backtrace-research/0.1 (+https://github.com/bachhavathang/backtrace)"

# Single words, not phrases. The VA search matches phrases narrowly: six phrases
# for the synthetic corpus's product families returned 205 of 510,681 lines, while
# "glove" alone returns 2,200. These cover those families and their neighbours —
# the near-duplicates that make retrieval hard.
DEFAULT_TERMS = ("glove", "catheter", "syringe", "gauze", "drape", "sponge",
                 "electrosurgical", "dressing", "needle", "tubing", "suture", "gown")
FDA_BATCH = 10   # part numbers per openFDA query; word matching makes big batches noisy
FDA_SAFE = ':()+"'   # openFDA's query syntax, left unescaped in the URL


class Fetcher:
    def __init__(self, delay: float, manifest: list[dict]) -> None:
        self.delay, self.manifest, self._last = delay, manifest, 0.0

    def get(self, url: str, ok_404: bool = False) -> bytes | None:
        for attempt in range(3):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and ok_404:
                    return None
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
            except urllib.error.URLError:
                if attempt == 2:
                    raise
            time.sleep(2 ** attempt * 5)
        return None

    def record(self, url: str, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.manifest.append({
            "url": url, "path": path.as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })


def _slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")


def harvest_va(fetch: Fetcher, out: Path, terms, max_pages: int, detail_budget: int) -> list[dict]:
    rows: list[dict] = []
    for term in terms:
        for page in range(1, max_pages + 1):
            url = va_nac.list_url(term, page)
            html = fetch.get(url)
            path = out / "va_nac" / "list" / f"{_slug(term)}_p{page}.html"
            fetch.record(url, path, html)
            parsed = va_nac.parse_list_page(html.decode("utf-8", "replace"), page)
            rows.extend(parsed.rows)
            print(f"  VA  {term!r} p{page}: {len(parsed.rows)} rows (of {parsed.total})")
            if not parsed.has_next:
                break

    # Detail budget: one item per contract first (dates are contract-wide), then
    # further items for their price units, in harvest order.
    seen_contracts, first, rest = set(), [], []
    for r in rows:
        if not r.get("lognumber"):
            continue
        (rest if r["contract_number"] in seen_contracts else first).append(r["lognumber"])
        seen_contracts.add(r["contract_number"])
    wanted = list(dict.fromkeys(first + rest))[:detail_budget]

    details: dict[str, dict] = {}
    for i, log in enumerate(wanted, 1):
        url = va_nac.detail_url(log)
        html = fetch.get(url)
        fields = va_nac.parse_detail_page(html.decode("utf-8", "replace"))
        details[log] = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in fields.items()}
        fetch.manifest.append({"url": url, "path": None, "stored": "parsed fields only",
                               "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        if i % 50 == 0:
            print(f"  VA  details {i}/{len(wanted)}")
    data = json.dumps(details, indent=1, sort_keys=True).encode()
    fetch.record("(parsed from detail pages)", out / "va_nac" / "details.json", data)
    print(f"  VA  details: {len(details)} items")
    return rows


def harvest_fda(fetch: Fetcher, out: Path, parts: list[str], max_queries: int) -> None:
    key = os.environ.get("OPENFDA_API_KEY")
    wanted = {openfda.normalise_part(p) for p in parts} - {""}
    # Quoted terms only; a part number carrying a quote is skipped, not escaped.
    # Generic part numbers are not looked up at all: the join refuses them anyway
    # (openfda.MIN_PART_CHARS), so querying them only spends the daily quota.
    queryable = sorted({p for p in parts if p and '"' not in p
                        and len(openfda.normalise_part(p)) >= openfda.MIN_PART_CHARS})
    limit = len(queryable) if key else min(len(queryable), max_queries * FDA_BATCH)
    if limit < len(queryable):
        print(f"  FDA no OPENFDA_API_KEY: looking up {limit} of {len(queryable)} part numbers "
              f"({max_queries} queries); the rest stay unjoined")
    queryable = queryable[:limit]
    devices: dict[str, dict] = {}
    for i in range(0, len(queryable), FDA_BATCH):
        batch = " ".join(f'"{p}"' for p in queryable[i:i + FDA_BATCH]).replace(" ", "+")
        params = {"search": f"catalog_number:({batch})+version_or_model_number:({batch})",
                  "limit": 1000}
        if key:
            params["api_key"] = key
        url = f"{openfda.API}?{urlencode(params, safe=FDA_SAFE)}"
        body = fetch.get(url, ok_404=True)
        for rec in (json.loads(body).get("results", []) if body else []):
            parts_here = {openfda.normalise_part(rec.get("catalog_number")),
                          openfda.normalise_part(rec.get("version_or_model_number"))}
            if parts_here & wanted:     # exact matches only; word-matched noise dropped
                devices[rec["public_device_record_key"]] = openfda.trim(rec)
        if (i // FDA_BATCH) % 25 == 0:
            print(f"  FDA {i + FDA_BATCH}/{len(queryable)} part numbers, {len(devices)} devices")
    lines = "\n".join(json.dumps(d, sort_keys=True) for d in devices.values()).encode()
    url_shown = f"{openfda.API} (batched part-number queries; key {'used' if key else 'not used'})"
    fetch.record(url_shown, out / "openfda" / "devices.jsonl", lines)
    print(f"  FDA devices: {len(devices)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--terms", nargs="+", default=list(DEFAULT_TERMS))
    ap.add_argument("--max-pages", type=int, default=15, help="per term, 400 rows a page")
    ap.add_argument("--details", type=int, default=800,
                    help="VA detail pages: one per contract first (dates), then items (units)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--out", type=Path, default=RAW / date.today().isoformat())
    ap.add_argument("--skip-fda", action="store_true")
    ap.add_argument("--fda-max-queries", type=int, default=900,
                    help="cap without OPENFDA_API_KEY; openFDA allows ~1,000 a day keyless")
    args = ap.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} already exists; snapshots are immutable. Pick another --out.")
        return 1
    manifest: list[dict] = []
    fetch = Fetcher(args.delay, manifest)
    print(f"Harvesting into {args.out}")
    rows = harvest_va(fetch, args.out, args.terms, args.max_pages, args.details)
    if not args.skip_fda:
        harvest_fda(fetch, args.out, [r["catalog_number"] for r in rows], args.fda_max_queries)
    (args.out / "manifest.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "terms": args.terms, "max_pages": args.max_pages, "detail_budget": args.details,
        "fda_max_queries": args.fda_max_queries,
        "files": manifest}, indent=1))
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
