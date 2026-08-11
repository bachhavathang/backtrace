"""Run Backtrace over all non-catalog orders and report recovered dollars.

    python -m src.generate_data   # first time
    python main.py                # backward scan
    python main.py --preflight    # check prompt-cache eligibility, make no calls
    python main.py --no-review    # scan only, leave the review queue for later

This is the "backward scan" mode from the case study: sweep historical
non-catalog spend, reverse-map each line, total the recoverable dollars.

The scan runs in three phases, and the split is the point:

  1. Adjudicate   All orders, concurrently. Independent lines, so this is the
                  difference between N x 2s and N/8 x 2s. No human blocks here.
  2. Review       The uncertain lines, sequentially, with a person. A scan of ten
                  thousand orders cannot stop on order three waiting for someone
                  to type 'y' — so the queue is drained after the scan, not
                  during it.
  3. Report       Dollars, decisions, spend, latency, cache hit rate. The cost of
                  a scan is part of its result: recovering $4,450 is a different
                  claim depending on whether it cost $0.01 or $400.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src import config, llm
from src.agent import candidates_for, review_interactively, run_one
from src.corpus import build_corpus, corpus_version, warm_retrieval
from src.recovery import record_recovery, total_recovered
from src.schema import MatchDecision, OrderLine, ReverseMapResult

ORDERS = Path(__file__).resolve().parent / "data" / "orders" / "non_catalog_orders.json"

DECISION_ORDER = [MatchDecision.MATCH, MatchDecision.UNCERTAIN, MatchDecision.NO_MATCH]


def load_orders(limit: int | None = None) -> list[OrderLine]:
    orders = [OrderLine(**o) for o in json.loads(ORDERS.read_text())]
    return orders[:limit] if limit else orders


# --- Phase 1: adjudicate --------------------------------------------------

def scan(orders: list[OrderLine], workers: int) -> list[ReverseMapResult]:
    """Adjudicate every order. Concurrent, non-blocking, order-preserving output."""
    results: dict[str, ReverseMapResult] = {}

    if workers <= 1:
        for order in orders:
            results[order.order_id] = run_one(order, interactive=False)
            _print_line(results[order.order_id], order)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_one, order, False): order for order in orders
            }
            for future in as_completed(futures):
                order = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - one bad line must not kill the scan
                    result = ReverseMapResult(
                        order_id=order.order_id,
                        decision=MatchDecision.UNCERTAIN,
                        rationale=f"Scan error: {type(exc).__name__}: {exc}",
                        guardrail_flags=["scan_error"],
                        corpus_version=corpus_version(),
                    )
                results[order.order_id] = result
                _print_line(result, order)

    # Report in input order regardless of completion order.
    return [results[o.order_id] for o in orders]


def _print_line(result: ReverseMapResult, order: OrderLine) -> None:
    line = (f"[{result.decision.value:9}] {order.order_id}  "
            f"{order.raw_description[:38]:38}")
    if result.recoverable:
        line += f"  -> ${result.recoverable:,.2f} ({result.matched_sku})"
    elif result.decision == MatchDecision.UNCERTAIN:
        line += "  -> queued for review"
    if result.blocked_by_guardrail:
        line += f"  ! {','.join(result.guardrail_flags)}"
    print(line)


# --- Phase 2: review ------------------------------------------------------

def review_queue(results: list[ReverseMapResult],
                 orders: list[OrderLine]) -> list[ReverseMapResult]:
    """Walk the uncertain lines with a person, then record what they confirm."""
    by_id = {o.order_id: o for o in orders}
    pending = [r for r in results if r.needs_human_review]
    if not pending:
        return results

    # Failures are separated from genuine ambiguity. A wedged API call and a
    # near-duplicate glove contract both land in UNCERTAIN, but only one of them
    # is a question a human can usefully answer.
    errored = [r for r in pending if "llm_error" in r.guardrail_flags]
    errored_ids = {r.order_id for r in errored}
    reviewable = [r for r in pending if r.order_id not in errored_ids]

    if errored:
        print(f"\n{len(errored)} line(s) failed to adjudicate and were NOT reviewed "
              "(retry the scan):")
        for r in errored:
            print(f"  {r.order_id}: {r.rationale}")

    if not reviewable:
        return results

    print(f"\n--- {len(reviewable)} line(s) need human review ---")
    for result in reviewable:
        review_interactively(result, candidates_for(result), by_id[result.order_id])
        if result.human_confirmed:
            record_recovery(result)
    return results


# --- Phase 3: report ------------------------------------------------------

def report(results: list[ReverseMapResult]) -> None:
    counts = {d: 0 for d in DECISION_ORDER}
    for r in results:
        counts[r.decision] = counts.get(r.decision, 0) + 1

    print("\n" + "=" * 62)
    print("SCAN RESULT")
    print("=" * 62)
    for decision in DECISION_ORDER:
        print(f"  {decision.value:<10} {counts.get(decision, 0)}")

    still_open = sum(1 for r in results if r.needs_human_review)
    if still_open:
        print(f"  {'(unreviewed)':<10} {still_open}")

    # Two different numbers, and conflating them hides re-runs. The scan total is
    # what this pass found; the ledger total is every claim ever filed. On a
    # re-run the scan total is unchanged and the ledger total does not move,
    # because claims are idempotent on order_id.
    scan_total = sum(r.recoverable for r in results)
    print(f"\n  RECOVERED THIS SCAN: ${scan_total:,.2f}")
    print(f"  LEDGER TOTAL:        ${total_recovered():,.2f}  (all claims to date)")

    usage = llm.ACCOUNT.summary()
    if not usage.get("calls"):
        print("\n  No model calls made.")
        return

    print("\n  --- spend ---")
    print(f"  model calls      {usage['calls']}")
    if usage.get("failed"):
        print(f"  failed calls     {usage['failed']}")
    print(f"  cost             ${usage['cost_usd']:.4f}")
    if results:
        print(f"  cost per order   ${usage['cost_usd'] / len(results):.5f}")
    for tier, stats in sorted(usage["by_tier"].items()):
        print(f"    {tier:<8} {stats['calls']:>3} calls  "
              f"${stats['cost_usd']:.4f}  ({stats['model']})")

    print("\n  --- tokens ---")
    print(f"  input            {usage['input_tokens']:,}")
    print(f"  output           {usage['output_tokens']:,}")
    print(f"  cache read       {usage['cache_read_tokens']:,}")
    print(f"  cache write      {usage['cache_write_tokens']:,}")
    print(f"  cache hit rate   {usage['cache_hit_rate']:.1%}")
    if usage["cache_hit_rate"] == 0 and usage["calls"] > 1:
        print("  ! nothing served from cache across multiple calls - run "
              "`python main.py --preflight`")

    lat = usage["latency_ms"]
    print("\n  --- latency (per model call) ---")
    print(f"  mean {lat['mean']:.0f} ms   p50 {lat['p50']:.0f} ms   "
          f"p95 {lat['p95']:.0f} ms   max {lat['max']:.0f} ms")
    print(f"\n  call log: {config.CALL_LOG}")


# --- Entry point ----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtrace backward recovery scan")
    parser.add_argument("--preflight", action="store_true",
                        help="report prompt-cache eligibility and exit")
    parser.add_argument("--no-review", action="store_true",
                        help="skip the human review phase; leave the queue open")
    parser.add_argument("--sequential", action="store_true",
                        help="disable concurrency (easier to read while debugging)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only scan the first N orders")
    parser.add_argument("--workers", type=int, default=config.MAX_CONCURRENCY)
    args = parser.parse_args()

    if not config.has_api_key():
        print("ANTHROPIC_API_KEY is not set. Adjudication needs a credential.\n"
              "  PowerShell:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
              "  bash:        export ANTHROPIC_API_KEY=sk-ant-...\n"
              "Deterministic layers still run without one: pytest -q")
        sys.exit(1)

    if args.preflight:
        for tier in (config.FAST, config.PRECISE):
            info = llm.preflight(tier)
            print(f"\n{info['model']}  ({info['prompt_version']})")
            print(f"  cached prefix    {info['cached_prefix_tokens']} tokens")
            print(f"  cache minimum    {info['cache_minimum_tokens']} tokens")
            print(f"  will engage      {info['caching_will_engage']}")
            print(f"  {info['note']}")
        return

    orders = load_orders(args.limit)
    corpus = build_corpus()
    print(f"Corpus: {len(corpus)} contract lines (version {corpus_version(corpus)})")
    print(f"Scanning {len(orders)} non-catalog orders "
          f"({'sequential' if args.sequential else f'{args.workers} workers'})...\n")

    # Load the embedding model before the clock starts so the first order does
    # not absorb a multi-second one-off cost and look pathologically slow.
    warm_retrieval()

    results = scan(orders, workers=1 if args.sequential else args.workers)

    if not args.no_review:
        review_queue(results, orders)

    report(results)


if __name__ == "__main__":
    main()
