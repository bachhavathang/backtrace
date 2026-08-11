"""Evaluation harness for the reverse-map agent.

    python -m evals.run_eval              # full run (needs ANTHROPIC_API_KEY)
    python -m evals.run_eval --offline    # guardrail + retrieval only, no key
    python -m evals.run_eval --sweep      # add the threshold curve
    python -m evals.run_eval --json out.json

Why accuracy is not the headline number
---------------------------------------
The errors here are not symmetric, so a single accuracy figure would average
together two outcomes with wildly different costs:

    A FALSE CLAIM is auto-matching an order to the wrong contract line. It puts a
    demand for money in front of a vendor that the hospital is not owed. It costs
    credibility with the customer and the vendor at once, and it is the thing this
    agent exists to avoid.

    A MISS is failing to spot a real overpayment. It costs one line of savings and
    nothing else. The status quo — a human never finding it — is exactly this.

So the harness reports three families separately and refuses to blend them:

  SAFETY        false_claims (must be 0), auto-match precision, and whether every
                injection attempt was caught. This is the gate.
  EFFECTIVENESS auto-match recall and recovered dollars — the money actually found
                without a human.
  EFFICIENCY    escalation rate — the human cost of the safety above. A system
                that escalates everything is perfectly safe and worthless, which
                is why this number is reported next to the other two.

Retrieval recall@k is reported on its own because it is a hard ceiling: if the
right contract line is not in the shortlist, no adjudicator can recover it, and
the fix is retrieval, not the prompt.

The threshold sweep replays the recorded verdicts through agent.decide() at
different bars. It costs nothing and makes no extra calls, which is the whole
reason decide() is a pure function taking a Thresholds object.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, guardrails  # noqa: E402
from src.agent import decide  # noqa: E402
from src.config import RETRIEVAL_K, THRESHOLDS, Thresholds, pick_tier  # noqa: E402
from src.corpus import build_corpus, retrieve_semantic, warm_retrieval  # noqa: E402
from src.llm import ACCOUNT, adjudicate  # noqa: E402
from src.schema import MatchDecision, OrderLine  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset.json"


# --- One evaluated case ---------------------------------------------------

@dataclass
class EvalRecord:
    case: dict
    retrieved_skus: list[str] = field(default_factory=list)
    top_similarity: float = 0.0
    short_circuited: bool = False
    verdict: guardrails.Verdict | None = None
    matched_sku: str | None = None
    tier: str | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    def decision(self, thresholds: Thresholds | None = None) -> MatchDecision:
        """Replay the policy at any thresholds. No model call."""
        if self.short_circuited or self.verdict is None:
            return MatchDecision.NO_MATCH
        return decide(self.verdict, self.matched_sku is not None, thresholds)

    @property
    def expected(self) -> str:
        return self.case["expected_decision"]

    @property
    def expected_sku(self) -> str | None:
        return self.case.get("expected_sku")

    @property
    def flagged(self) -> bool:
        return bool(self.verdict and self.verdict.flags)


def evaluate_case(case: dict) -> EvalRecord:
    """Retrieve + adjudicate one labelled case. Never touches the ledger."""
    order = OrderLine(
        order_id=case["id"],
        raw_description=case["raw_description"],
        quantity=case.get("quantity", 1),
        list_unit_price=case.get("list_unit_price", 0.0),
        sku_hint=case.get("sku_hint"),
    )
    corpus = build_corpus()
    query = order.raw_description + (f" {order.sku_hint}" if order.sku_hint else "")
    candidates = retrieve_semantic(query, corpus, k=RETRIEVAL_K)

    record = EvalRecord(
        case=case,
        retrieved_skus=[c.contract.sku for c in candidates],
        top_similarity=candidates[0].similarity if candidates else 0.0,
    )

    if not candidates or candidates[0].similarity < THRESHOLDS.no_match_bar:
        record.short_circuited = True
        return record

    tier = pick_tier(
        candidates[0].similarity,
        candidates[1].similarity if len(candidates) > 1 else 0.0,
    )
    verdict, call = adjudicate(case["id"], order.raw_description, candidates, tier)
    record.verdict = verdict
    record.matched_sku = verdict.chosen_sku
    record.tier = call.tier
    record.cost_usd = call.cost_usd
    record.latency_ms = call.latency_ms
    return record


# --- Metrics --------------------------------------------------------------

def score(records: list[EvalRecord], thresholds: Thresholds | None = None) -> dict:
    """Compute the three metric families at a given set of thresholds."""
    false_claims: list[dict] = []
    misses: list[dict] = []
    over_escalations: list[dict] = []
    auto_matches = correct_auto = 0
    escalated = 0
    should_match = sum(1 for r in records if r.expected == "match")

    for r in records:
        decision = r.decision(thresholds)

        if decision == MatchDecision.UNCERTAIN:
            escalated += 1

        if decision == MatchDecision.MATCH:
            auto_matches += 1
            correct = r.expected == "match" and r.matched_sku == r.expected_sku
            if correct:
                correct_auto += 1
            else:
                # The failure this project exists to prevent.
                false_claims.append({
                    "id": r.case["id"],
                    "category": r.case["category"],
                    "description": r.case["raw_description"][:60],
                    "claimed_sku": r.matched_sku,
                    "expected": r.expected_sku or r.expected,
                    "confidence": round(r.verdict.confidence, 3) if r.verdict else None,
                    "reason": r.verdict.reason if r.verdict else "",
                })
        elif r.expected == "match" and decision == MatchDecision.NO_MATCH:
            misses.append({
                "id": r.case["id"],
                "description": r.case["raw_description"][:60],
                "expected_sku": r.expected_sku,
                "retrieved": r.retrieved_skus,
                "confidence": round(r.verdict.confidence, 3) if r.verdict else None,
            })
        elif r.expected in ("no_match",) and decision == MatchDecision.UNCERTAIN:
            over_escalations.append(r.case["id"])

    total = len(records) or 1
    return {
        "safety": {
            "false_claims": len(false_claims),
            "false_claim_detail": false_claims,
            "auto_match_precision": round(correct_auto / auto_matches, 4) if auto_matches else None,
            "injection_caught": _injection_coverage(records, thresholds),
        },
        "effectiveness": {
            "auto_matched": auto_matches,
            "correct_auto_matched": correct_auto,
            "auto_match_recall": round(correct_auto / should_match, 4) if should_match else None,
            "missed": len(misses),
            "miss_detail": misses,
        },
        "efficiency": {
            "escalation_rate": round(escalated / total, 4),
            "escalated": escalated,
            "over_escalated_absent": len(over_escalations),
            "over_escalated_ids": over_escalations,
        },
    }


def _injection_coverage(records: list[EvalRecord],
                        thresholds: Thresholds | None = None) -> dict:
    """Injection cases must be flagged AND must not auto-claim. Both, not either."""
    cases = [r for r in records if r.case["category"] == "injection"]
    if not cases:
        return {"total": 0}
    flagged = sum(1 for r in cases
                  if r.verdict and guardrails.FLAG_INJECTION in r.verdict.flags)
    auto_claimed = sum(1 for r in cases
                       if r.decision(thresholds) == MatchDecision.MATCH)
    return {
        "total": len(cases),
        "flagged": flagged,
        "auto_claimed": auto_claimed,
        "leaked_ids": [r.case["id"] for r in cases
                       if r.decision(thresholds) == MatchDecision.MATCH],
    }


def retrieval_recall(records: list[EvalRecord]) -> dict:
    """Was the correct contract line even in the shortlist? A ceiling on everything."""
    scored = [r for r in records if r.expected_sku]
    if not scored:
        return {"n": 0}
    hits = [r for r in scored if r.expected_sku in r.retrieved_skus]
    top1 = [r for r in scored if r.retrieved_skus[:1] == [r.expected_sku]]
    return {
        "n": len(scored),
        f"recall_at_{RETRIEVAL_K}": round(len(hits) / len(scored), 4),
        "recall_at_1": round(len(top1) / len(scored), 4),
        "missed_by_retrieval": [
            {"id": r.case["id"], "expected": r.expected_sku, "got": r.retrieved_skus}
            for r in scored if r.expected_sku not in r.retrieved_skus
        ],
    }


def by_category(records: list[EvalRecord],
                thresholds: Thresholds | None = None) -> dict:
    """Per-category pass rate. 'Pass' means the decision was acceptable for that case."""
    acceptable = {
        "clear": {MatchDecision.MATCH},
        "ambiguous": {MatchDecision.UNCERTAIN},
        # Escalating an absent item wastes a reviewer's time but claims nothing,
        # so it is counted separately in efficiency rather than failed here.
        "absent": {MatchDecision.NO_MATCH, MatchDecision.UNCERTAIN},
        "trap": {MatchDecision.NO_MATCH, MatchDecision.UNCERTAIN},
        "injection": {MatchDecision.UNCERTAIN, MatchDecision.NO_MATCH},
    }
    out: dict[str, dict] = {}
    for r in records:
        cat = r.case["category"]
        bucket = out.setdefault(cat, {"n": 0, "pass": 0, "fail_ids": []})
        bucket["n"] += 1
        decision = r.decision(thresholds)
        ok = decision in acceptable.get(cat, set())
        if cat == "clear":
            ok = ok and r.matched_sku == r.expected_sku
        if ok:
            bucket["pass"] += 1
        else:
            bucket["fail_ids"].append(f"{r.case['id']}({decision.value})")
    for bucket in out.values():
        bucket["pass_rate"] = round(bucket["pass"] / bucket["n"], 4)
    return out


def sweep(records: list[EvalRecord]) -> list[dict]:
    """Replay the policy across a grid of confidence bars. Zero extra API calls."""
    rows = []
    for high in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        for low in (0.30, 0.50):
            t = Thresholds(no_match_bar=THRESHOLDS.no_match_bar,
                           low_bar=low, high_bar=high)
            s = score(records, t)
            rows.append({
                "high_bar": high,
                "low_bar": low,
                "false_claims": s["safety"]["false_claims"],
                "precision": s["safety"]["auto_match_precision"],
                "recall": s["effectiveness"]["auto_match_recall"],
                "escalation_rate": s["efficiency"]["escalation_rate"],
            })
    return rows


# --- Offline mode ---------------------------------------------------------

def run_offline(cases: list[dict]) -> int:
    """Exercise the layers that need no credential: sanitiser and retrieval.

    Worth having as its own mode. The injection guardrail is deterministic code,
    so it can be verified in CI on every commit rather than only when someone
    runs a paid eval.
    """
    print("OFFLINE MODE - guardrails + retrieval only, no model calls.\n")

    injections = [c for c in cases if c["category"] == "injection"]
    caught = 0
    print(f"Injection detection ({len(injections)} cases)")
    for case in injections:
        clean = guardrails.sanitize_order_text(case["raw_description"])
        ok = clean.suspicious
        caught += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {case['id']}  flags={clean.flags}")
        if "order_text" in clean.text.lower() and "<" in clean.text:
            print(f"        ! fence tag survived sanitising: {clean.text[:80]}")

    benign = [c for c in cases if c["category"] != "injection"]
    false_positives = [c for c in benign
                       if guardrails.sanitize_order_text(c["raw_description"]).suspicious]
    print(f"\n  caught {caught}/{len(injections)}")
    print(f"  false positives on {len(false_positives)}/{len(benign)} benign lines"
          f"{': ' + ', '.join(c['id'] for c in false_positives) if false_positives else ''}")

    warm_retrieval()
    records = []
    for case in benign:
        order_text = case["raw_description"]
        query = order_text + (f" {case['sku_hint']}" if case.get("sku_hint") else "")
        candidates = retrieve_semantic(query, build_corpus(), k=RETRIEVAL_K)
        records.append(EvalRecord(
            case=case,
            retrieved_skus=[c.contract.sku for c in candidates],
            top_similarity=candidates[0].similarity if candidates else 0.0,
        ))

    recall = retrieval_recall(records)
    print(f"\nRetrieval (k={RETRIEVAL_K}, {recall['n']} cases with a known answer)")
    print(f"  recall@{RETRIEVAL_K}  {recall[f'recall_at_{RETRIEVAL_K}']:.1%}")
    print(f"  recall@1  {recall['recall_at_1']:.1%}"
          "   <- the gap between these two is what the adjudicator has to close")
    for miss in recall["missed_by_retrieval"]:
        print(f"  MISS {miss['id']}: wanted {miss['expected']}, got {miss['got']}")

    failed = caught < len(injections) or recall["missed_by_retrieval"]
    return 1 if failed else 0


# --- Reporting ------------------------------------------------------------

def print_report(records: list[EvalRecord], show_sweep: bool) -> int:
    s = score(records)
    recall = retrieval_recall(records)
    cats = by_category(records)
    usage = ACCOUNT.summary()

    print("\n" + "=" * 66)
    print(f"EVAL - {len(records)} cases, thresholds "
          f"low={THRESHOLDS.low_bar} high={THRESHOLDS.high_bar} "
          f"floor={THRESHOLDS.no_match_bar}")
    print("=" * 66)

    print("\nSAFETY  (the gate - false_claims must be 0)")
    print(f"  false claims          {s['safety']['false_claims']}")
    precision = s["safety"]["auto_match_precision"]
    print(f"  auto-match precision  {precision:.1%}" if precision is not None
          else "  auto-match precision  n/a (nothing auto-matched)")
    inj = s["safety"]["injection_caught"]
    if inj.get("total"):
        print(f"  injections flagged    {inj['flagged']}/{inj['total']}")
        print(f"  injections auto-claimed {inj['auto_claimed']}"
              f"{'  <- ' + ', '.join(inj['leaked_ids']) if inj['leaked_ids'] else ''}")
    for fc in s["safety"]["false_claim_detail"]:
        print(f"    x {fc['id']} [{fc['category']}] claimed {fc['claimed_sku']} "
              f"(expected {fc['expected']}) conf={fc['confidence']}")
        print(f"       {fc['description']!r}")
        print(f"       model said: {fc['reason']}")

    print("\nEFFECTIVENESS")
    e = s["effectiveness"]
    print(f"  auto-matched          {e['correct_auto_matched']}/{e['auto_matched']} correct")
    if e["auto_match_recall"] is not None:
        print(f"  auto-match recall     {e['auto_match_recall']:.1%}")
    print(f"  missed                {e['missed']}")
    for m in e["miss_detail"]:
        print(f"    - {m['id']} wanted {m['expected_sku']}, retrieved {m['retrieved']}, "
              f"conf={m['confidence']}")

    print("\nEFFICIENCY")
    print(f"  escalation rate       {s['efficiency']['escalation_rate']:.1%} "
          f"({s['efficiency']['escalated']} lines need a human)")
    print(f"  absent items escalated {s['efficiency']['over_escalated_absent']}")

    print(f"\nRETRIEVAL CEILING (k={RETRIEVAL_K})")
    print(f"  recall@{RETRIEVAL_K}              {recall[f'recall_at_{RETRIEVAL_K}']:.1%}")
    print(f"  recall@1              {recall['recall_at_1']:.1%}")
    for miss in recall["missed_by_retrieval"]:
        print(f"    ! {miss['id']}: {miss['expected']} not in {miss['got']} "
              "— unfixable by prompting")

    print("\nBY CATEGORY")
    for cat, b in sorted(cats.items()):
        line = f"  {cat:<10} {b['pass']:>2}/{b['n']:<2} {b['pass_rate']:>6.1%}"
        if b["fail_ids"]:
            line += f"   fails: {', '.join(b['fail_ids'])}"
        print(line)

    if usage.get("calls"):
        print("\nCOST")
        print(f"  calls {usage['calls']}   ${usage['cost_usd']:.4f}   "
              f"cache hit {usage['cache_hit_rate']:.1%}   "
              f"p95 {usage['latency_ms']['p95']:.0f} ms")
        for tier, st in sorted(usage["by_tier"].items()):
            print(f"    {tier:<8} {st['calls']:>3} calls  ${st['cost_usd']:.4f}")

    if show_sweep:
        print("\nTHRESHOLD SWEEP  (replayed offline - no extra API calls)")
        print(f"  {'high':>5} {'low':>5} {'claims':>7} {'prec':>7} {'recall':>7} {'escal':>7}")
        for row in sweep(records):
            prec = f"{row['precision']:.1%}" if row["precision"] is not None else "  n/a"
            rec = f"{row['recall']:.1%}" if row["recall"] is not None else "  n/a"
            marker = " *" if row["false_claims"] == 0 else "  "
            print(f"  {row['high_bar']:>5.2f} {row['low_bar']:>5.2f} "
                  f"{row['false_claims']:>7} {prec:>7} {rec:>7} "
                  f"{row['escalation_rate']:>6.1%}{marker}")
        print("  * = zero false claims at these bars. Pick the row with the "
              "highest recall among those.")

    # A false claim fails the run. Everything else is a tuning conversation.
    return 1 if s["safety"]["false_claims"] else 0


# --- Entry point ----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtrace eval harness")
    parser.add_argument("--offline", action="store_true",
                        help="guardrail + retrieval checks only; no credential needed")
    parser.add_argument("--sweep", action="store_true",
                        help="print the threshold curve")
    parser.add_argument("--category", help="only run one category")
    parser.add_argument("--workers", type=int, default=config.MAX_CONCURRENCY)
    parser.add_argument("--json", help="write the full result to this path")
    args = parser.parse_args()

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]

    if args.offline:
        sys.exit(run_offline(cases))

    if not config.has_api_key():
        print("ANTHROPIC_API_KEY is not set.\n"
              "Run `python -m evals.run_eval --offline` for the credential-free "
              "guardrail and retrieval checks.")
        sys.exit(1)

    print(f"Evaluating {len(cases)} cases with {args.workers} workers...")
    warm_retrieval()
    ACCOUNT.reset()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(evaluate_case, cases))

    exit_code = print_report(records, show_sweep=args.sweep)

    if args.json:
        payload = {
            "thresholds": {
                "no_match_bar": THRESHOLDS.no_match_bar,
                "low_bar": THRESHOLDS.low_bar,
                "high_bar": THRESHOLDS.high_bar,
            },
            "score": score(records),
            "retrieval": retrieval_recall(records),
            "by_category": by_category(records),
            "usage": ACCOUNT.summary(),
            "sweep": sweep(records),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
