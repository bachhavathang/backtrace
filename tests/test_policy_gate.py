"""The CI gate: zero false claims on every recorded real-data verdict. No network.

evals/fixtures/ holds 915 model verdicts on realistic orders over 24,883 real
contract lines (evals/real_eval.py), trimmed to what agent.decide() and the
outcome scoring read, with the deterministic choice checks already applied.
decide() is pure, so the money policy can be replayed against them on every commit.

This fails the build if any change — a threshold, the decision order, the outcome
rules, the holder rule's status — would turn one of those recorded verdicts into
a claim against a vendor for money it does not owe. It cannot catch a change in
what the MODEL would now say; that takes a paid run of evals/real_eval.py.
"""
import json
from pathlib import Path

import pytest

from evals.real_eval import outcome, score
from src.config import THRESHOLDS

FIXTURES = sorted((Path(__file__).parent.parent / "evals" / "fixtures").glob("verdicts-*.jsonl"))


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_fixtures_are_present():
    assert len(FIXTURES) >= 2 and sum(len(_load(p)) for p in FIXTURES) >= 900


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_no_false_claims_at_shipped_bars(path):
    records = _load(path)
    false = [r["order_id"] for r in records if outcome(r, THRESHOLDS) == "false_claim"]
    assert false == [], f"{len(false)} false claims at high_bar={THRESHOLDS.high_bar}: {false}"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_shipped_bars_still_claim_something(path):
    # A policy that never claims is perfectly safe and worthless; guard that too.
    s = score(_load(path), THRESHOLDS)
    assert s["auto_claim_recall"] and s["auto_claim_recall"] > 0.25


def test_gate_would_catch_a_regression():
    # The gate is only worth trusting if it fires: at the old bar, the fresh set
    # recorded one false claim (a serrated forceps the order never asked for).
    from dataclasses import replace
    fresh = next(p for p in FIXTURES if "seed11" in p.name)
    assert score(_load(fresh), replace(THRESHOLDS, high_bar=0.80))["false_claims"] >= 1
