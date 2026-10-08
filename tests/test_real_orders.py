"""Tests for the realistic-order generator and its claim-rule audit. No network.

Runs against the committed fixture snapshot (five real VA rows), so it checks the
generator's guarantees, not the statistics of a full harvest.
"""
import random
from datetime import date
from pathlib import Path

import pytest

from evals import real_orders as ro
from src.ingest import load_all
from src.recovery import CLAIMABLE
from src.schema import ContractPrice, OrderLine
from src.sources.va_nac import VaNacSource

SNAP = Path(__file__).parent / "fixtures" / "snapshot-2026-10-05"


@pytest.fixture(scope="module")
def corpus():
    rows, _ = load_all([VaNacSource(SNAP)])
    return rows


# --- Operators ------------------------------------------------------------------

def test_drop_words_never_drops_numbers():
    rng = random.Random(0)
    for _ in range(50):
        out = ro.op_drop_words("Foley Catheter 16Fr 2-way latex 10 per box", rng)
        assert "16Fr" in out and "2-way" in out and "10" in out


def test_abbreviate_uses_purchasing_shorthand():
    out = ro.op_abbreviate("Nitrile Exam Gloves Powder-Free Large", random.Random(1))
    assert {"glv", "pf", "lg"} & set(out.split())


def test_rephrase_pack_keeps_the_count():
    out = ro.op_rephrase_pack("Foley Catheter 10/BX", random.Random(2))
    assert "10" in out and "/" not in out


def test_degrade_is_deterministic_for_a_seed():
    ops = ["lowercase", "abbreviate", "shuffle", "typo"]
    a = ro.degrade("Nitrile Exam Gloves Powder-Free Large box/100", ops, random.Random(5))
    b = ro.degrade("Nitrile Exam Gloves Powder-Free Large box/100", ops, random.Random(5))
    assert a == b


# --- Dataset guarantees ---------------------------------------------------------

def test_generation_is_reproducible(corpus):
    assert ro.generate(corpus, 10, seed=3) == ro.generate(corpus, 10, seed=3)


def test_no_product_appears_in_both_splits(corpus):
    ds = ro.generate(corpus, 10, seed=3)
    split_of: dict[tuple, str] = {}
    for o in ds["orders"]:
        key = tuple(o["label"]["product_key"])
        assert split_of.setdefault(key, o["label"]["split"]) == o["label"]["split"]


def test_held_out_operators_only_in_test_split(corpus):
    for seed in range(20):
        for o in ro.generate(corpus, 10, seed=seed)["orders"]:
            if set(o["label"]["ops"]) & set(ro.HELD_OUT_OPS):
                assert o["label"]["split"] == "test"


def test_orders_load_as_order_lines_and_hide_the_label(corpus):
    for o in ro.generate(corpus, 10, seed=4)["orders"]:
        line = OrderLine(**{k: v for k, v in o.items() if k != "label"})
        assert line.raw_description and line.list_unit_price > 0
        assert "label" not in line.model_dump()


def test_seller_case_names_the_holder_as_supplier(corpus):
    for o in ro.generate(corpus, 10, seed=5)["orders"]:
        if o["label"]["kind"] == "match_seller":
            holder = o["label"]["source_row"].split("|")[0]
            assert o["supplier"].casefold() == holder


def test_list_price_is_marked_synthetic(corpus):
    assert all(o["label"]["list_price_synthetic"] for o in ro.generate(corpus, 10, seed=6)["orders"])


# --- The audit: no savings or no-contract order may ever be claimable ------------

def test_audit_finds_no_false_claims(corpus):
    for seed in range(10):
        report = ro.audit(corpus, ro.generate(corpus, 10, seed=seed), date(2026, 10, 8))
        assert report["false_claims"] == []


def test_maker_held_product_is_never_a_savings_case():
    # Regression: the first full generation built a "savings" order from a
    # reseller's row of a product whose MAKER also held a contract. Under the
    # settled rule that is a genuine claim (case B) — the label was wrong.
    def row(holder, contract):
        return ContractPrice(sku="27710812", description="Surgical gown large",
                             vendor=holder, holder=holder, manufacturer="STANDARD TEXTILE",
                             contracted_unit_price=5.0, source=contract, contract_id=contract)
    corpus = [row("Standard Textile", "MAKER"), row("Tru-Care", "RESELLER")]
    for seed in range(30):
        ds = ro.generate(corpus, 4, seed=seed, plan=ro.Plan(0, 0, 1.0, 0))
        assert ds["orders"] == []          # no eligible product, so no savings case
        assert ro.audit(corpus, ds, date(2026, 10, 8))["false_claims"] == []


def test_audit_would_catch_a_false_claim(corpus):
    # The audit is only worth trusting if it fires. Relabel a seller case as
    # savings: the rule still (correctly) finds it claimable, so the audit must flag it.
    ds = ro.generate(corpus, 10, seed=8)
    seller = next(o for o in ds["orders"] if o["label"]["kind"] == "match_seller")
    seller["label"]["kind"] = "savings"
    report = ro.audit(corpus, ds, date(2026, 10, 8))
    assert seller["order_id"] in report["false_claims"]
