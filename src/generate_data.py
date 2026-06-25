"""Generate messy contract sources + non-catalog orders for Backtrace.

Fully implemented — plumbing, not the point. But the DESIGN of this data is what
makes the demo interesting, so read it. The contract prices live in THREE
different messy formats (a GPO overlay table, a local agreement letter, an email
addendum). The non-catalog orders are written so that matching them is genuinely
hard — vague descriptions, no clean SKUs, abbreviations — which is exactly the
real problem Resolvd solves.

The "answer key" (which order should map to which SKU) is encoded in comments so
you can sanity-check your agent, but the agent must NOT see it.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
CONTRACTS = DATA / "contracts"
ORDERS = DATA / "orders"
SYSTEMS = DATA / "mock_systems"

# --- Contract source 1: a GPO overlay, pipe-delimited table dump ---------
GPO_OVERLAY = """GPO NATIONAL OVERLAY PRICING — eff. 2025
vendor|sku|description|price
Medline|GLV-N100|Nitrile Exam Gloves, Powder-Free, Large, box/100|9.10
Medline|GAU-4404|Gauze Sponge 4x4 12-ply sterile, pk/25|2.85
Cardinal|SYR-L10|Luer-Lok Syringe 10 mL sterile|0.39
Cardinal|CTH-F16|Foley Catheter 16Fr 2-way latex|4.20
"""

# --- Contract source 2: a local agreement, prose letter ------------------
LOCAL_AGREEMENT = """LOCAL PURCHASING AGREEMENT — St. Mark's Health System & Acme Surgical
Effective this year. The following negotiated prices apply to St. Mark's:

  - Sterile surgical drape, fenestrated, large (Acme #DRP-LG-2): $6.75 each
  - Electrosurgical pencil, hand control, disposable (Acme #ESU-PEN): $11.40 each
  - Nitrile exam gloves, large, box of 100 (Acme #ACM-GLV-L): $8.40 per box

Signed, Procurement Office.
"""

# --- Contract source 3: an email addendum that CHANGES one price ---------
EMAIL_ADDENDUM = """From: rep@cardinal.com
Subject: Price update — Foley catheters
Date: 4/12

Hi team — effective immediately, the Foley Catheter 16Fr 2-way (our SYR... sorry,
CTH-F16) drops to $3.60 under the new tier. Please use this going forward.
Thanks!
"""

# Non-catalog orders. Comments show the intended answer (agent must not see them).
ORDERS_DATA = [
    # order_id, raw_description, qty, list_price, sku_hint
    # -> should MATCH Medline GLV-N100 @ 9.10  (paid 13.50 -> recover 4.40/ea)
    ("PO-5001", "nitrile gloves large pf box 100", 300, 13.50, None),
    # -> should MATCH Cardinal SYR-L10 @ 0.39   (paid 0.55 -> recover 0.16/ea)
    ("PO-5002", "10ml luer lock syringes sterile", 2000, 0.55, "SYR"),
    # -> should MATCH Cardinal CTH-F16 @ 3.60 (email addendum price, NOT 4.20!)
    ("PO-5003", "foley cath 16fr two way", 150, 6.00, None),
    # -> AMBIGUOUS: two glove contracts exist (Medline 9.10, Acme 8.40). UNCERTAIN.
    ("PO-5004", "exam gloves nitrile lg", 500, 12.00, None),
    # -> NO_MATCH: nothing like this in any contract.
    ("PO-5005", "portable ultrasound gel warmer unit", 2, 240.00, None),
    # -> should MATCH Acme DRP-LG-2 @ 6.75 (only in the prose local agreement)
    ("PO-5006", "fenestrated surgical drape large sterile", 400, 9.25, None),
]


def main() -> None:
    for d in (CONTRACTS, ORDERS, SYSTEMS):
        d.mkdir(parents=True, exist_ok=True)

    (CONTRACTS / "gpo_overlay.txt").write_text(GPO_OVERLAY)
    (CONTRACTS / "local_agreement.txt").write_text(LOCAL_AGREEMENT)
    (CONTRACTS / "email_addendum.txt").write_text(EMAIL_ADDENDUM)

    orders = [
        {"order_id": o, "raw_description": d, "quantity": q,
         "list_unit_price": p, "sku_hint": s}
        for (o, d, q, p, s) in ORDERS_DATA
    ]
    (ORDERS / "non_catalog_orders.json").write_text(json.dumps(orders, indent=2))

    ledger = SYSTEMS / "recovery_ledger.json"
    if not ledger.exists():
        ledger.write_text(json.dumps([], indent=2))

    print(f"Wrote 3 messy contract sources to {CONTRACTS}")
    print(f"Wrote {len(orders)} non-catalog orders to {ORDERS}")
    print(f"Initialized empty recovery ledger at {ledger}")


if __name__ == "__main__":
    main()
