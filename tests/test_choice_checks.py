"""Tests for the deterministic choice checks (guardrails.choice_flags). No network.

Every case here is a real order/contract pair from the Step 3 eval on 24,883 VA
contract lines, where the model picked the wrong variant at 0.85-0.92 confidence.
"""
import pytest

from src import guardrails as g
from src.schema import CandidateMatch, ContractPrice


def _c(desc, sku="X"):
    return CandidateMatch(contract=ContractPrice(sku=sku, description=desc, vendor="V",
                                                 contracted_unit_price=1.0, source="s"),
                          similarity=0.8)


PLAIN = _c("Symmetry Forceps; Dressing; 10 in", "30-4138")
SERRATED = _c("Symmetry Forceps; Dressing; Serrated; 10 in", "06-0025")
SIX_INCH = _c("Symmetry Forceps; Dressing; 6 in", "30-4135")


# --- Number check -----------------------------------------------------------------

def test_stated_count_absent_from_match_conflicts():
    three = _c("SUTURE BOOTS; WHITE FOAM; 3 PAIRS YELLOW; BULK NON-STERILE (500 TRAYS/BOX)")
    assert g.number_conflict("suture boots white foam 5 pairs yellow bulk nonster 500", three)


def test_all_numbers_present_is_no_conflict():
    assert not g.number_conflict("dressing symmetry 10 in", PLAIN)


@pytest.mark.parametrize("order,desc", [
    ("needle 10mm", "NEEDLE 10.0MM"),          # trailing-zero decimals are the same number
    ("suture 6/0 18in", "SUTURE 6/0 18\""),
    ("box of 100 gloves", "GLOVES 100/BX"),     # rephrased pack count
])
def test_same_numbers_in_other_notation(order, desc):
    assert not g.number_conflict(order, _c(desc))


def test_part_number_in_sku_counts():
    assert not g.number_conflict("forceps 30-4138", PLAIN)


# --- Sibling check ----------------------------------------------------------------

def test_order_silent_on_the_distinguishing_word_flags():
    # The false claim that survived the first version of this check: SKU tokens
    # made these twins look only 44% alike. Similarity is on descriptions now.
    assert g.choice_flags("dressing symmetry 10 in", [PLAIN, SERRATED, SIX_INCH], 1) \
        == [g.FLAG_SIBLING]


def test_order_naming_the_distinguishing_word_is_supported():
    assert g.choice_flags("symmetry dressing serrated 10 in", [SERRATED, PLAIN], 1) == []


def test_order_favouring_the_rival_flags():
    # "serrated" is in the order; the model picked the plain one anyway.
    assert g.FLAG_SIBLING in g.choice_flags("symmetry serrated dressing 10", [PLAIN, SERRATED], 1)


def test_rival_ruled_out_by_the_orders_numbers_is_ignored():
    # 6 in vs 10 in: the order's "10" already rejects the 6-inch rival.
    assert g.choice_flags("symmetry forceps dressing serrated 10 in", [SERRATED, SIX_INCH], 1) == []


def test_unrelated_rival_is_not_a_sibling():
    other = _c("Foley catheter 16Fr two-way latex")
    assert g.choice_flags("symmetry forceps dressing 10 in", [PLAIN, other], 1) == []


def test_order_citing_the_part_number_decides_it():
    assert g.choice_flags("forceps dressing 10 in 30-4138", [PLAIN, SERRATED], 1) == []


def test_abstention_owes_no_choice_flags():
    assert g.choice_flags("dressing symmetry 10 in", [PLAIN, SERRATED], 0) == []


def test_choice_flags_escalate():
    v = g.Verdict(1, "30-4138", 0.99, False, "r", [g.FLAG_SIBLING])
    assert v.must_escalate
