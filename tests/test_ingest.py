"""Tests for the ingest seam and the contract-price policy (no LLM key needed).

Two things are pinned here. First, that moving the synthetic files behind
src/ingest.py changed nothing: same rows, same order, same prices, same
corpus_version. Second, Decision 1 from docs/WORKING_STATE.md §5.2 — which
contract price a claim uses — including every way it must refuse to guess.
"""
from datetime import date

import pytest

from src.corpus import build_corpus, corpus_version
from src.ingest import merge_amendments, sources_for
from src.recovery import (CLAIMABLE, EXPIRED, NEEDS_REVIEW, NO_VALID_CONTRACT,
                          select_contract_price)
from src.schema import ContractPrice, OrderLine

AS_OF = date(2026, 10, 5)


def _row(sku="GLV-1", price=10.0, holder="Medline", contract="C1", **kw):
    return ContractPrice(sku=sku, description="Nitrile gloves large", vendor=holder,
                         contracted_unit_price=price, source=f"src {contract}",
                         contract_id=contract, holder=holder, **kw)


def _order(**kw):
    base = dict(order_id="PO-1", raw_description="nitrile gloves lg", quantity=10,
                list_unit_price=15.0, manufacturer="Medline")
    return OrderLine(**{**base, **kw})


# --- The seam changed nothing ---------------------------------------------

def test_synthetic_corpus_is_unchanged_by_the_adapter():
    # The exact rows, in the exact order, that build_corpus produced before the
    # ingest seam existed. Order matters: it breaks retrieval ties and keys the
    # embedding cache.
    corpus = build_corpus(refresh=True)
    assert [(c.sku, c.contracted_unit_price, c.source) for c in corpus] == [
        ("GLV-N100", 9.10, "GPO overlay 2025"),
        ("GAU-4404", 2.85, "GPO overlay 2025"),
        ("SYR-L10", 0.39, "GPO overlay 2025"),
        ("CTH-F16", 3.60, "Email addendum 4/12"),
        ("DRP-LG-2", 6.75, "Local agreement - St. Mark's"),
        ("ESU-PEN", 11.40, "Local agreement - St. Mark's"),
        ("ACM-GLV-L", 8.40, "Local agreement - St. Mark's"),
    ]


def test_corpus_version_is_unchanged_by_the_adapter():
    # Measured at c1a7071, before the seam. Every claim in an existing ledger
    # carries this hash; changing it without changing a price would orphan them.
    assert corpus_version(build_corpus(refresh=True)) == "ae5a7592a43d"


def test_every_synthetic_row_carries_contract_identity():
    for row in build_corpus(refresh=True):
        assert row.contract_id and row.holder and row.source_text


def test_unknown_corpus_source_fails_loudly():
    with pytest.raises(ValueError, match="Unknown corpus source"):
        sources_for("real-but-not-built-yet")


# --- Merge: amendments collapse, distinct contracts do not ----------------

def test_amendment_of_same_contract_replaces_in_place():
    merged = merge_amendments([_row("A", 5.0), _row("B", 1.0), _row("A", 4.0)])
    assert [(r.sku, r.contracted_unit_price) for r in merged] == [("A", 4.0), ("B", 1.0)]


def test_same_sku_on_two_contracts_keeps_both():
    merged = merge_amendments([_row("A", 5.0, contract="C1"),
                               _row("A", 4.0, contract="C2")])
    assert sorted(r.contracted_unit_price for r in merged) == [4.0, 5.0]


def test_two_vendors_reusing_a_part_number_keeps_both():
    merged = merge_amendments([_row("X-1", holder="Medline"),
                               _row("X-1", holder="Cardinal")])
    assert len(merged) == 2


# --- Units ----------------------------------------------------------------

def test_box_price_without_pack_size_has_no_per_each_price():
    assert _row(price=9.10, uom="box").per_each_price is None


def test_per_each_price_divides_by_pack_size():
    row = _row(price=9.10, uom="box", units_per_pack=100)
    assert row.per_each_price == pytest.approx(0.091)


# --- Decision 1: which price a claim uses ---------------------------------

def test_single_valid_contract_is_claimable():
    sel = select_contract_price(_order(), [_row(price=9.0)], AS_OF)
    assert sel.status == CLAIMABLE and sel.chosen.contracted_unit_price == 9.0
    assert not sel.has_additional


def test_several_valid_prices_claim_the_highest_and_report_the_gap():
    rows = [_row(price=9.0, contract="C1"), _row(price=8.0, contract="C2")]
    sel = select_contract_price(_order(), rows, AS_OF)
    assert sel.status == CLAIMABLE
    assert sel.chosen.contracted_unit_price == 9.0   # smallest, undisputable claim
    assert sel.lowest.contracted_unit_price == 8.0
    assert sel.has_additional and sel.gap == pytest.approx(1.0)


def test_contract_binds_manufacturer_not_the_distributor():
    # Bought from Owens & Minor, made by Medline: Medline's contract applies.
    order = _order(supplier="Owens & Minor", manufacturer="Medline")
    assert select_contract_price(order, [_row(holder="Medline")], AS_OF).status == CLAIMABLE


def test_other_makers_contract_is_never_claimed_and_never_dropped():
    # Could be a genuine other-maker price (savings, not recovery) or a name
    # mismatch ("Medline Industries"). Either way: a human, not NO_VALID_CONTRACT.
    sel = select_contract_price(_order(manufacturer="Medline Industries"),
                                [_row(holder="Medline")], AS_OF)
    assert sel.status == NEEDS_REVIEW and sel.chosen is None


def test_unknown_manufacturer_needs_review():
    sel = select_contract_price(_order(manufacturer=None), [_row()], AS_OF)
    assert sel.status == NEEDS_REVIEW


def test_only_contracts_in_force_on_the_effective_date_count():
    rows = [_row(price=7.0, contract="OLD", effective_end=date(2025, 12, 31)),
            _row(price=9.0, contract="NEW", effective_start=date(2026, 1, 1))]
    sel = select_contract_price(_order(effective_date=date(2026, 3, 1)), rows, AS_OF)
    assert sel.chosen.contract_id == "NEW" and not sel.has_additional


def test_no_contract_in_force_is_no_valid_contract():
    rows = [_row(effective_end=date(2025, 12, 31))]
    sel = select_contract_price(_order(effective_date=date(2026, 3, 1)), rows, AS_OF)
    assert sel.status == NO_VALID_CONTRACT


def test_undated_order_against_dated_contract_needs_review():
    sel = select_contract_price(_order(), [_row(effective_start=date(2026, 1, 1))], AS_OF)
    assert sel.status == NEEDS_REVIEW


def test_prices_compare_per_each_across_pack_sizes():
    # $9.10 per box of 100 ($0.091 each) is LOWER than $0.12 each — comparing the
    # raw numbers would get this backwards.
    rows = [_row(price=9.10, contract="C1", uom="box", units_per_pack=100),
            _row(price=0.12, contract="C2", uom="each")]
    sel = select_contract_price(_order(), rows, AS_OF)
    assert sel.chosen.contract_id == "C2" and sel.lowest.contract_id == "C1"
    assert sel.gap == pytest.approx(0.029)


def test_incomparable_units_need_review():
    rows = [_row(price=9.10, contract="C1", uom="box"),
            _row(price=0.12, contract="C2", uom="each")]
    assert select_contract_price(_order(), rows, AS_OF).status == NEEDS_REVIEW


def test_closed_claim_window_is_expired_not_recoverable():
    rows = [_row(claim_window_days=90)]
    sel = select_contract_price(_order(effective_date=date(2026, 1, 1)), rows, AS_OF)
    assert sel.status == EXPIRED and sel.chosen is not None


def test_open_claim_window_is_claimable():
    rows = [_row(claim_window_days=365)]
    sel = select_contract_price(_order(effective_date=date(2026, 1, 1)), rows, AS_OF)
    assert sel.status == CLAIMABLE
