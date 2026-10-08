"""Step 3: the full pipeline on realistic orders — thresholds, calibration, escalation.

    python -m evals.real_eval --limit 20                  # pilot: measure cost per order
    python -m evals.real_eval --budget 5                   # adjudicate every order, capped
    python -m evals.real_eval --replay                     # re-score saved verdicts, no calls

Uses the API. Every other step so far was free; this is the one that is not, so it
is built to never pay twice and never overspend:

  - verdicts are appended to data/derived/<snapshot>/verdicts-seed<N>.jsonl as they
    arrive, and a rerun skips orders already there — a crash loses one call;
  - --budget is a hard cap on THIS run's spend plus everything already on file;
  - --replay re-scores the saved verdicts under any thresholds with zero calls,
    because agent.decide() is pure. That is what the whole sweep is built on.

What is scored, and why it is not accuracy
-------------------------------------------
An AUTO-CLAIM needs three things: decide() says MATCH, the model's product is the
right one, and the settled holder rule (select_contract_price) says CLAIMABLE.

    false claim    auto-claimed, but the product is wrong or the order had no
                   contract. The one number that must be zero.
    correct claim  auto-claimed, right product, seller- or maker-held contract.
    escalated      sent to a human: UNCERTAIN, or a MATCH the price rule could not
                   settle (unknown units, undated order...).
    savings        right product, but held by someone else: a tip, not a claim.

Bars are CHOSEN on the tune split — the lowest high_bar with zero false claims —
and only then reported on the test split, whose products and two degradation
operators the choice never saw.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path

from src import config, llm
from src.agent import decide
from src.config import RETRIEVAL_K, THRESHOLDS, Thresholds, pick_tier
from src.corpus import product_key, retrieve_hybrid, variant_vocabulary, warm_retrieval
from src.guardrails import Verdict
from src.ingest import load_all, snapshot_dir
from src.recovery import CLAIMABLE, SAVINGS_OPPORTUNITY, select_contract_price
from src.schema import MatchDecision, OrderLine

DERIVED = config.DATA / "derived"
POSITIVE = ("match_seller", "match_maker")
CHOICE_FLAGS = ("order_number_absent_from_match", "rival_candidate_not_excluded",
                "match_has_variant_word_order_lacks")
HIGH_BARS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99)
LOW_BARS = (0.30, 0.50)
BUCKETS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01))


# --- Phase 1: adjudicate (costs money) ------------------------------------------

def adjudicate_order(o: dict, corpus, exclude: frozenset, as_of: date) -> dict:
    """Retrieve + adjudicate one order and record everything a replay needs."""
    order = OrderLine(**{k: v for k, v in o.items() if k != "label"})
    want = tuple(o["label"]["product_key"])
    cands = retrieve_hybrid(order.raw_description, corpus, k=RETRIEVAL_K, exclude=exclude)
    rec = {
        "order_id": o["order_id"], "kind": o["label"]["kind"], "split": o["label"]["split"],
        "ops": o["label"]["ops"], "want": list(want),
        "in_shortlist": any(product_key(c.contract) == want for c in cands),
        "top_similarity": cands[0].similarity if cands else 0.0,
        "short_circuited": False,
    }
    if not cands or cands[0].similarity < THRESHOLDS.no_match_bar:
        rec["short_circuited"] = True
        return rec

    tier = pick_tier(cands[0].similarity, cands[1].similarity if len(cands) > 1 else 0.0)
    # Live flags follow config.VARIANT_CHECK; --replay --checks [--variant-check]
    # can score either setup against the same paid verdicts afterwards.
    verdict, call = llm.adjudicate(o["order_id"], order.raw_description, cands, tier,
                                   variant_vocabulary(corpus) if config.VARIANT_CHECK
                                   else frozenset())
    chosen = cands[verdict.chosen_index - 1] if verdict.chosen_index else None
    rec.update({
        "chosen_index": verdict.chosen_index, "chosen_sku": verdict.chosen_sku,
        "confidence": verdict.confidence, "is_ambiguous": verdict.is_ambiguous,
        "reason": verdict.reason, "flags": verdict.flags,
        "chosen_key": list(product_key(chosen.contract)) if chosen else None,
        "tier": call.tier, "cost_usd": call.cost_usd, "latency_ms": call.latency_ms,
        "ok": call.ok, "cache_read_tokens": call.cache_read_tokens,
    })
    if chosen is not None:
        sel = select_contract_price(order, chosen.rows, as_of)
        rec["price_status"] = sel.status
    return rec


def run_calls(orders: list[dict], corpus, exclude, out: Path, budget: float,
              workers: int, as_of: date) -> None:
    done = {json.loads(line)["order_id"] for line in out.read_text().splitlines()} \
        if out.exists() else set()
    prior_cost = sum(json.loads(line).get("cost_usd") or 0.0
                     for line in out.read_text().splitlines()) if out.exists() else 0.0
    todo = [o for o in orders if o["order_id"] not in done]
    print(f"{len(done)} already on file (${prior_cost:.4f}); {len(todo)} to adjudicate; "
          f"budget ${budget:.2f}")
    if not todo:
        return
    warm_retrieval()
    llm.warm_cache((config.FAST, config.PRECISE), batch_size=len(todo))

    def spent() -> float:
        return prior_cost + llm.ACCOUNT.summary().get("cost_usd", 0.0)

    # Waves of `workers`: the budget is checked against money actually spent, never
    # against calls merely queued, so a run overshoots by at most one wave.
    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(todo), workers):
            if spent() >= budget:
                print(f"Budget reached at ${spent():.4f}; stopping. Rerun to continue.")
                break
            wave = todo[start:start + workers]
            for rec in pool.map(lambda o: adjudicate_order(o, corpus, exclude, as_of), wave):
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  {min(start + workers, len(todo))}/{len(todo)}  spent ${spent():.4f}",
                  flush=True)
    print(f"Run spend ${llm.ACCOUNT.summary().get('cost_usd', 0.0):.4f}, total ${spent():.4f}")


# --- Phase 2: replay (free) -------------------------------------------------------

def outcome(rec: dict, t: Thresholds) -> str:
    """What the pipeline would do with this order at these bars."""
    if rec["short_circuited"]:
        return "no_match"
    v = Verdict(rec["chosen_index"], rec["chosen_sku"], rec["confidence"],
                rec["is_ambiguous"], rec.get("reason", ""), rec.get("flags", []))
    decision = decide(v, rec["chosen_index"] > 0, t)
    if decision == MatchDecision.NO_MATCH:
        return "no_match"
    if decision == MatchDecision.UNCERTAIN:
        return "escalated"
    status = rec.get("price_status")
    if status == CLAIMABLE:
        right = rec["chosen_key"] == rec["want"] and rec["kind"] != "no_contract"
        return "correct_claim" if right else "false_claim"
    if status == SAVINGS_OPPORTUNITY:
        return "savings"
    return "escalated"      # MATCH the price rule could not settle


def score(records: list[dict], t: Thresholds) -> dict:
    out = [outcome(r, t) for r in records]
    n = len(records)
    pos = [r for r in records if r["kind"] in POSITIVE]
    claims = out.count("correct_claim") + out.count("false_claim")
    return {
        "n": n, "false_claims": out.count("false_claim"),
        "auto_claim_precision": (out.count("correct_claim") / claims) if claims else None,
        "auto_claim_recall": (sum(1 for r, o in zip(records, out)
                                  if r in pos and o == "correct_claim") / len(pos)) if pos else None,
        "escalation_rate": out.count("escalated") / n if n else 0.0,
        "no_contract_rejected": _rate(records, out, "no_contract", ("no_match",)),
        "savings_found": _rate(records, out, "savings", ("savings",)),
        "outcomes": {k: out.count(k) for k in sorted(set(out))},
    }


def _rate(records, out, kind, good) -> float | None:
    sub = [o for r, o in zip(records, out) if r["kind"] == kind]
    return (sum(o in good for o in sub) / len(sub)) if sub else None


def choose_bars(tune: list[dict]) -> tuple[Thresholds, list[dict]]:
    """Lowest high_bar with zero false claims on TUNE; ties broken by fewer escalations."""
    grid = []
    for high in HIGH_BARS:
        for low in LOW_BARS:
            t = replace(THRESHOLDS, low_bar=low, high_bar=high)
            grid.append({"high": high, "low": low, "t": t, **score(tune, t)})
    safe = [g for g in grid if g["false_claims"] == 0]
    best = min(safe, key=lambda g: (g["high"], g["escalation_rate"])) if safe else None
    return (best["t"] if best else None), grid


def calibration(records: list[dict]) -> list[tuple[str, int, float | None]]:
    """When the model picks a candidate at confidence c, how often is it right?"""
    picked = [r for r in records if not r["short_circuited"] and r.get("chosen_index")]
    rows = []
    for lo, hi in BUCKETS:
        b = [r for r in picked if lo <= r["confidence"] < hi]
        right = sum(1 for r in b if r["chosen_key"] == r["want"] and r["kind"] != "no_contract")
        rows.append((f"{lo:.2f}-{min(hi, 1.0):.2f}", len(b), right / len(b) if b else None))
    return rows


def _pct(x) -> str:
    return "  n/a" if x is None else f"{x:6.1%}"


def with_choice_checks(records: list[dict], dataset: dict, corpus,
                       variant_check: bool = False) -> list[dict]:
    """Re-apply guardrails.choice_flags to saved verdicts. No API calls.

    Retrieval is deterministic, so each order's shortlist is rebuilt exactly; a
    rebuilt pick whose product differs from the saved one is reported, not trusted.
    Lets a new deterministic check be measured against verdicts already paid for.
    """
    from src.guardrails import choice_flags
    exclude = frozenset(tuple(k) for k in dataset["held_out_product_keys"])
    orders = {o["order_id"]: o for o in dataset["orders"]}
    vocab = variant_vocabulary(corpus) if variant_check else frozenset()
    out, drift = [], 0
    for r in records:
        r = dict(r)
        if not r["short_circuited"] and r.get("chosen_index"):
            text = orders[r["order_id"]]["raw_description"]
            cands = retrieve_hybrid(text, corpus, k=RETRIEVAL_K, exclude=exclude)
            idx = r["chosen_index"]
            if idx > len(cands) or list(product_key(cands[idx - 1].contract)) != r["chosen_key"]:
                drift += 1
            else:
                # Choice flags are recomputed from scratch, so a replay can switch a
                # check OFF as well as on; every other flag is kept as recorded.
                kept = [f for f in (r.get("flags") or []) if f not in CHOICE_FLAGS]
                r["flags"] = list(dict.fromkeys(kept + choice_flags(text, cands, idx, vocab)))
        out.append(r)
    if drift:
        print(f"WARNING: {drift} rebuilt shortlists disagree with the saved pick; left unchecked")
    return out


def report(records: list[dict]) -> int:
    tune = [r for r in records if r["split"] == "tune"]
    test = [r for r in records if r["split"] == "test"]
    cost = sum(r.get("cost_usd") or 0.0 for r in records)
    calls = sum(1 for r in records if not r["short_circuited"])
    print(f"\n{len(records)} orders on file ({len(tune)} tune / {len(test)} test), "
          f"{calls} model calls, ${cost:.4f} (${cost / max(calls, 1):.5f}/call)")
    print(f"Right product in shortlist: {_pct(sum(r['in_shortlist'] for r in records if r['kind'] != 'no_contract') / max(1, sum(r['kind'] != 'no_contract' for r in records)))}"
          f"   short-circuited: {sum(r['short_circuited'] for r in records)}")

    current = score(test, THRESHOLDS)
    print(f"\nCURRENT bars (high {THRESHOLDS.high_bar}, low {THRESHOLDS.low_bar}) on TEST:")
    _print_score(current)

    chosen, grid = choose_bars(tune) if tune else (None, [])
    if tune:
        print("\nSWEEP on TUNE (replayed, no calls)")
        print("  high  low   false  precision  recall  escalation")
        for g in grid:
            print(f"  {g['high']:.2f}  {g['low']:.2f}   {g['false_claims']:>4}   "
                  f"{_pct(g['auto_claim_precision'])}   {_pct(g['auto_claim_recall'])}  "
                  f"{_pct(g['escalation_rate'])}")
    if chosen:
        print(f"\nCHOSEN on tune: high_bar {chosen.high_bar}, low_bar {chosen.low_bar}. On TEST:")
        _print_score(score(test, chosen))
    elif tune:
        print("\nNo bar on the grid reaches zero false claims on tune.")

    print("\nCALIBRATION (all orders): when the model picks at confidence c, is it right?")
    for label, n, p in calibration(records):
        print(f"  {label}   n={n:<4} realised precision {_pct(p)}")
    test_false = score(test, chosen or THRESHOLDS)["false_claims"]
    return 1 if test_false else 0


def _print_score(s: dict) -> None:
    print(f"  false claims {s['false_claims']}   precision {_pct(s['auto_claim_precision'])}   "
          f"recall {_pct(s['auto_claim_recall'])}   escalation {_pct(s['escalation_rate'])}")
    print(f"  no-contract rejected {_pct(s['no_contract_rejected'])}   "
          f"savings found {_pct(s['savings_found'])}   outcomes {s['outcomes']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="adjudicate at most N new orders")
    ap.add_argument("--split", choices=("tune", "test", "all"), default="all",
                    help="adjudicate only this split (a confirmation run uses test)")
    ap.add_argument("--budget", type=float, default=5.0, help="hard USD cap, including prior runs")
    ap.add_argument("--workers", type=int, default=config.MAX_CONCURRENCY)
    ap.add_argument("--replay", action="store_true", help="score saved verdicts only; no calls")
    ap.add_argument("--checks", action="store_true",
                    help="re-apply the deterministic choice checks to saved verdicts (no calls)")
    ap.add_argument("--variant-check", action="store_true",
                    help="with --checks: include the third (variant-word) check")
    args = ap.parse_args(argv)

    snap = args.snapshot or snapshot_dir()
    base = DERIVED / snap.name
    out = base / f"verdicts-seed{args.seed}.jsonl"
    if not args.replay:
        if not config.has_api_key():
            print("ANTHROPIC_API_KEY is not set; use --replay to score saved verdicts.")
            return 1
        from src.sources.va_nac import VaNacSource
        corpus, _ = load_all([VaNacSource(snap)])
        dataset = json.loads((base / f"orders-seed{args.seed}.json").read_text())
        exclude = frozenset(tuple(k) for k in dataset["held_out_product_keys"])
        orders = [o for o in dataset["orders"]
                  if args.split == "all" or o["label"]["split"] == args.split]
        if args.limit is not None:
            done = {json.loads(l)["order_id"] for l in out.read_text().splitlines()} \
                if out.exists() else set()
            orders = [o for o in orders if o["order_id"] not in done][:args.limit]
        run_calls(orders, corpus, exclude, out, args.budget, args.workers, date.today())
    if not out.exists():
        print(f"No verdicts at {out}.")
        return 1
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    if args.checks:
        from src.sources.va_nac import VaNacSource
        corpus, _ = load_all([VaNacSource(snap)])
        dataset = json.loads((base / f"orders-seed{args.seed}.json").read_text())
        warm_retrieval()
        records = with_choice_checks(records, dataset, corpus, args.variant_check)
        fired = sum(1 for r in records for f in (r.get("flags") or []) if f in CHOICE_FLAGS)
        print(f"Choice checks re-applied: {fired} flags raised across {len(records)} orders")
    return report(records)


if __name__ == "__main__":
    sys.exit(main())
