"""Tests for the deterministic layers (no LLM key needed).

Run: pytest -q

Covers corpus parsing (the three messy formats), keyword retrieval sanity, the
recovery dollar math, and ledger idempotency. We do NOT assert on the LLM
adjudication here — that's eval territory, not unit-test territory. Good interview
line: "I test the deterministic core hard; I eval the probabilistic parts."
"""
from src import recovery
from src.corpus import build_corpus, retrieve_keyword
from src.schema import ReverseMapResult


def test_corpus_ingests_all_three_sources():
    corpus = build_corpus()
    sources = {c.source for c in corpus}
    assert any("GPO" in s for s in sources)
    assert any("Local agreement" in s for s in sources)
    assert any("Email addendum" in s for s in sources)


def test_email_addendum_overrides_gpo_price():
    # CTH-F16 is $4.20 in the GPO overlay but $3.60 in the later email addendum.
    corpus = {c.sku: c for c in build_corpus()}
    assert corpus["CTH-F16"].contracted_unit_price == 3.60
    assert "Email" in corpus["CTH-F16"].source


def test_keyword_retrieval_returns_candidates():
    corpus = build_corpus()
    cands = retrieve_keyword("nitrile gloves large box 100", corpus, k=3)
    assert len(cands) == 3
    assert cands[0].similarity >= cands[-1].similarity  # sorted desc


def test_recoverable_math():
    r = ReverseMapResult(order_id="X", decision="match", matched_sku="GLV-N100",
                         list_unit_price=13.50, contracted_unit_price=9.10,
                         quantity=300)
    assert r.recoverable == round((13.50 - 9.10) * 300, 2)  # 1320.00


def test_no_negative_recovery():
    # If list < contract (we somehow paid LESS), recoverable floors at 0.
    r = ReverseMapResult(order_id="Y", decision="match", matched_sku="Z",
                         list_unit_price=1.00, contracted_unit_price=2.00,
                         quantity=10)
    assert r.recoverable == 0.0


def test_recovery_ledger_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "LEDGER", tmp_path / "ledger.json")
    r = ReverseMapResult(order_id="IDEMP-1", decision="match", matched_sku="GLV-N100",
                         list_unit_price=13.50, contracted_unit_price=9.10,
                         quantity=300, human_confirmed=True)
    first = recovery.record_recovery(r)
    second = recovery.record_recovery(r)
    assert first["status"] == "claimed"
    assert second["status"] == "noop_already_claimed"
    assert recovery.total_recovered() == 1320.00
