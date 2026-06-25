"""Run Backtrace over all non-catalog orders and report recovered dollars.

    python -m src.generate_data   # first time
    python main.py

This is the "backward scan" mode from Resolvd's case study: sweep historical
non-catalog spend, reverse-map each line, total the recoverable dollars.
"""
import json
from pathlib import Path

from src.agent import run_one
from src.recovery import total_recovered
from src.schema import OrderLine

ORDERS = Path(__file__).resolve().parent / "data" / "orders" / "non_catalog_orders.json"


def main() -> None:
    orders = [OrderLine(**o) for o in json.loads(ORDERS.read_text())]
    print(f"Scanning {len(orders)} non-catalog orders...\n")
    for o in orders:
        r = run_one(o)
        line = f"[{r.decision.value:9}] {o.order_id}  {o.raw_description[:40]:40}"
        if r.recoverable:
            line += f"  -> ${r.recoverable:,.2f} recoverable ({r.matched_sku})"
        print(line)
    print(f"\nTOTAL RECOVERED: ${total_recovered():,.2f}")


if __name__ == "__main__":
    main()
