"""Tests for product grouping, the embedding cache, and the claim-holder rule.

No network and no real embedding model: the model is replaced by a tiny fake, so
these exercise the retrieval plumbing — grouping, caching, locking — not the
quality of the vectors.
"""
import threading
from datetime import date

import numpy as np
import pytest
import torch

from src import corpus as corpus_mod
from src.recovery import (CLAIMABLE, NEEDS_REVIEW, SAVINGS_OPPORTUNITY,
                          select_contract_price)
from src.schema import ContractPrice, OrderLine

from fakes import FakeModel


def _row(sku, desc="Nitrile exam glove large", holder="Medline", contract="C1", price=9.0, **kw):
    return ContractPrice(sku=sku, description=desc, vendor=holder, holder=holder,
                         contracted_unit_price=price, source=f"src {contract}",
                         contract_id=contract, **kw)


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    monkeypatch.setattr(corpus_mod, "_model", FakeModel())
    monkeypatch.setattr(corpus_mod, "_corpus_cache", None)
    monkeypatch.setattr(corpus_mod, "EMBED_CACHE", tmp_path)
    return tmp_path


# --- Grouping -----------------------------------------------------------------

def test_same_product_on_two_contracts_is_one_candidate(fake_model):
    rows = [_row("GLV-1", contract="FSS"), _row("GLV-1", contract="BPA", price=8.0),
            _row("GAU-4", desc="Gauze sponge 4x4"), _row("SYR-10", desc="Syringe 10 mL")]
    cands = corpus_mod.retrieve_keyword("nitrile glove large", rows, k=3)
    assert [c.contract.sku for c in cands].count("GLV-1") == 1     # not "the same glove x2"
    glove = next(c for c in cands if c.contract.sku == "GLV-1")
    assert {r.contract_id for r in glove.rows} == {"FSS", "BPA"}
    assert len(cands) == 3


def test_different_makers_sharing_a_part_number_stay_apart():
    a = _row("150", manufacturer="MAKER A")
    b = _row("150", manufacturer="MAKER B")
    assert corpus_mod.product_key(a) != corpus_mod.product_key(b)


def test_unjoined_rows_group_only_on_identical_description():
    assert corpus_mod.product_key(_row("X-1", desc="Glove large")) == \
        corpus_mod.product_key(_row("X-1", desc="glove   LARGE", holder="Other"))
    assert corpus_mod.product_key(_row("X-1", desc="Glove large")) != \
        corpus_mod.product_key(_row("X-1", desc="Glove medium"))


def test_ungrouped_candidate_rows_is_just_the_contract():
    rows = [_row("A"), _row("B", desc="Gauze")]
    cand = corpus_mod.retrieve_keyword("glove", rows, k=1)[0]
    assert cand.group == [] and cand.rows == [cand.contract]


# --- Semantic retrieval: locking and the disk cache ---------------------------

def test_first_semantic_query_does_not_deadlock(monkeypatch, tmp_path):
    # Regression: retrieve_semantic held the embedding lock while the first model
    # load tried to take it again. Every first query of a run hung forever.
    monkeypatch.setattr(corpus_mod, "_model", None)
    monkeypatch.setattr(corpus_mod, "_corpus_cache", None)
    monkeypatch.setattr(corpus_mod, "EMBED_CACHE", tmp_path)

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda name: FakeModel())

    result = []
    t = threading.Thread(target=lambda: result.append(
        corpus_mod.retrieve_semantic("glove", [_row("A"), _row("B", desc="Gauze")], k=1)))
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "retrieve_semantic deadlocked on first use"
    assert result and result[0][0].contract.sku == "A"


def test_embeddings_are_reused_from_disk(fake_model, monkeypatch):
    rows = [_row("A"), _row("B", desc="Gauze sponge")]
    corpus_mod.retrieve_semantic("glove", rows, k=1)
    files = list(fake_model.glob("*.npy"))
    assert len(files) == 1

    class Exploding(FakeModel):
        def encode(self, texts, **kw):
            if not isinstance(texts, str):
                raise AssertionError("corpus was re-embedded despite a cache hit")
            return super().encode(texts, **kw)
    monkeypatch.setattr(corpus_mod, "_model", Exploding())
    monkeypatch.setattr(corpus_mod, "_corpus_cache", None)
    assert corpus_mod.retrieve_semantic("glove", list(rows), k=1)[0].contract.sku == "A"


def test_cache_key_changes_when_a_description_does():
    a = corpus_mod.embedding_cache_path(["Glove large GLV-1"])
    b = corpus_mod.embedding_cache_path(["Glove medium GLV-1"])
    assert a != b


# --- Who must hold the contract (settled 2026-10-07: A and B) ------------------

def _order(**kw):
    base = dict(order_id="PO-1", raw_description="nitrile glove lg", quantity=10,
                list_unit_price=15.0)
    return OrderLine(**{**base, **kw})


AS_OF = date(2026, 10, 7)


def test_case_a_seller_holds_the_contract():
    sel = select_contract_price(_order(supplier="Brightstar"), [_row("G", holder="Brightstar")], AS_OF)
    assert sel.status == CLAIMABLE


def test_case_b_maker_holds_the_contract():
    order = _order(supplier="Owens & Minor", manufacturer="Medline")
    assert select_contract_price(order, [_row("G", holder="Medline")], AS_OF).status == CLAIMABLE


def test_case_c_someone_else_holds_it_is_savings_not_a_claim():
    order = _order(supplier="Owens & Minor", manufacturer="Medline")
    sel = select_contract_price(order, [_row("G", holder="Shop R")], AS_OF)
    assert sel.status == SAVINGS_OPPORTUNITY and sel.chosen is None
    assert sel.lowest.holder == "Shop R"         # the tip: who to buy from next time


def test_registry_maker_on_a_reseller_row_does_not_make_it_claimable():
    # The rejected "holder OR registry manufacturer" rule would claim this.
    order = _order(supplier="Owens & Minor", manufacturer="Medline")
    row = _row("G", holder="Shop R", manufacturer="Medline")
    assert select_contract_price(order, [row], AS_OF).status == SAVINGS_OPPORTUNITY


def test_order_naming_neither_seller_nor_maker_needs_review():
    sel = select_contract_price(_order(), [_row("G")], AS_OF)
    assert sel.status == NEEDS_REVIEW


def test_seller_and_maker_contracts_both_count_highest_is_claimed():
    order = _order(supplier="Brightstar", manufacturer="Medline")
    rows = [_row("G", holder="Brightstar", contract="S", price=9.0),
            _row("G", holder="Medline", contract="M", price=8.0),
            _row("G", holder="Shop R", contract="R", price=5.0)]
    sel = select_contract_price(order, rows, AS_OF)
    assert sel.status == CLAIMABLE and sel.chosen.contract_id == "S"
    assert {r.contract_id for r in sel.valid} == {"S", "M"}       # R never counts


# --- The shortlist carries the whole group ----------------------------------

def _grouped_candidate():
    from src.schema import CandidateMatch
    fss = _row("GLV-1", holder="Medline", contract="FSS")
    bpa = _row("GLV-1", holder="Owens", contract="BPA")
    return CandidateMatch(contract=fss, similarity=0.9, group=[fss, bpa]), bpa


def test_flag_on_any_row_of_a_grouped_product_escalates():
    from src import guardrails
    cand, bpa = _grouped_candidate()
    cand.group[1] = bpa.model_copy(update={"needs_verification": True})
    assert guardrails.shortlist_flags([cand]) == [guardrails.FLAG_UNVERIFIED_CONTRACT]


def test_prompt_lists_every_vendor_of_a_product():
    from src import prompts
    cand, _ = _grouped_candidate()
    line = next(l for l in prompts.build_user_message("gloves", [cand]).splitlines()
                if l.startswith("1. sku="))
    assert 'vendor="Medline; Owens"' in line


def test_review_rebuilds_the_shortlist_by_identity_not_sku(monkeypatch):
    # Two rows share SKU GLV-1. Looking up by SKU picks whichever came last; the
    # result must come back with the exact row the agent was shown.
    from src import agent
    from src.corpus import identity_string
    from src.schema import ReverseMapResult
    fss = _row("GLV-1", holder="Medline", contract="FSS")
    other = _row("GLV-1", holder="Acme", contract="LOCAL", desc="Different glove")
    monkeypatch.setattr(agent, "build_corpus", lambda: [fss, other])
    result = ReverseMapResult(order_id="PO-1", decision="uncertain",
                              candidates_considered=["GLV-1"],
                              candidate_keys=[identity_string(fss)])
    rebuilt = agent.candidates_for(result)
    assert [c.contract.contract_id for c in rebuilt] == ["FSS"]
