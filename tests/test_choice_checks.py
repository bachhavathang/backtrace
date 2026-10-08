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


# --- Rule 3: unconfirmed variant word (off by default; see config.VARIANT_CHECK) ----

VOCAB = frozenset({"serrated", "curved", "stylet", "sterile", "large"})


def test_variant_word_the_order_never_mentions_flags():
    # The case that beat the first two checks on a fresh run: the plain Wullstein
    # forceps was not in the catalog, so only the serrated one could be shortlisted.
    serrated = _c("FORCEPS,WULLSTEIN DRESSING,SERRATED,", "214000F")
    assert g.unconfirmed_variant("'wullstein' drsg. forecps", serrated, VOCAB)


def test_variant_word_in_the_order_is_confirmed():
    assert not g.unconfirmed_variant("wullstein serrated forceps", _c("WULLSTEIN SERRATED"), VOCAB)


@pytest.mark.parametrize("order", ["forceps serated", "forceps serr", "glove lg"])
def test_typos_prefixes_and_glossary_confirm(order):
    chosen = _c("Forceps serrated" if "ser" in order else "Glove large")
    assert not g.unconfirmed_variant(order, chosen, VOCAB)


def test_no_vocabulary_means_no_flag():
    assert not g.unconfirmed_variant("forceps", _c("FORCEPS SERRATED"), frozenset())


def test_rule_three_is_off_unless_given_a_vocabulary():
    cands = [_c("FORCEPS WULLSTEIN SERRATED")]
    assert g.choice_flags("wullstein forceps", cands, 1) == []
    assert g.choice_flags("wullstein forceps", cands, 1, VOCAB) == [g.FLAG_UNCONFIRMED_VARIANT]


@pytest.mark.parametrize("a,b,ok", [
    ("forecps", "forceps", True), ("neelde", "needle", True), ("serated", "serrated", True),
    ("cat", "cut", False), ("large", "lodge", False)])
def test_one_edit_apart(a, b, ok):
    assert g._one_edit_apart(a, b) is ok


def test_vocabulary_learns_add_ons_not_swaps():
    from src import corpus as C
    def row(sku, desc):
        return ContractPrice(sku=sku, description=desc, vendor="V", holder="V",
                             contracted_unit_price=1.0, source="s", contract_id=sku)
    # Distinct alphabetic names: words are letters-only, so "model0".."model3" would
    # all read as "model" and collapse into one sibling slot (counted once, by design).
    names = ("adson", "kelly", "mayo", "crile")
    rows = []
    for n in names:      # four products that exist with and without "serrated"
        rows += [row(f"P{n}", f"forceps dressing {n}"), row(f"S{n}", f"forceps dressing {n} serrated")]
    for n in names:      # four "dressing" vs "tissue" SWAPS — different products
        rows += [row(f"D{n}", f"clamp dressing {n}"), row(f"T{n}", f"clamp tissue {n}")]
    vocab = C.variant_vocabulary(rows)
    assert "serrated" in vocab
    assert "tissue" not in vocab and "dressing" not in vocab


def test_vocabulary_ignores_grammar_and_packaging():
    from src import corpus as C
    def row(sku, desc):
        return ContractPrice(sku=sku, description=desc, vendor="V", holder="V",
                             contracted_unit_price=1.0, source="s", contract_id=sku)
    rows = []
    for n in ("alpha", "bravo", "charlie", "delta"):
        rows += [row(f"P{n}", f"gauze sponge {n}"), row(f"B{n}", f"gauze sponge {n} box")]
    assert "box" not in C.variant_vocabulary(rows)
