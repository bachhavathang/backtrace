"""Tests for the audit trail.

A recovery claim is a financial assertion against a vendor. When it is disputed,
"the model said so" is not an answer — you have to name which model, which
prompt, and which contract lines were on the table. These tests hold that
guarantee in place, plus the idempotency it depends on.
"""
import json
from concurrent.futures import ThreadPoolExecutor

from src import recovery
from src.corpus import build_corpus, corpus_version
from src.schema import MatchDecision, ReverseMapResult


def _claim(order_id="PO-X", **overrides) -> ReverseMapResult:
    fields = dict(
        order_id=order_id,
        decision=MatchDecision.MATCH,
        matched_sku="GLV-N100",
        matched_source="GPO overlay 2025",
        confidence=0.93,
        rationale="order specifies powder-free",
        list_unit_price=13.50,
        contracted_unit_price=9.10,
        quantity=300,
        model="claude-sonnet-5",
        tier="precise",
        prompt_version="reverse-map/v2",
        corpus_version="abc123def456",
        candidates_considered=["GLV-N100", "ACM-GLV-L", "GAU-4404"],
    )
    fields.update(overrides)
    return ReverseMapResult(**fields)


# --- Corpus versioning ----------------------------------------------------

def test_corpus_version_is_stable_across_calls():
    assert corpus_version() == corpus_version()


def test_corpus_version_tracks_a_price_change():
    """The email addendum moves CTH-F16 from $4.20 to $3.60. Claims must be able
    to say which price index they were computed against."""
    corpus = build_corpus()
    before = corpus_version(corpus)
    bumped = [c.model_copy(update={"contracted_unit_price": 99.0})
              if c.sku == "CTH-F16" else c for c in corpus]
    assert corpus_version(bumped) != before


def test_build_corpus_is_cached_but_refreshable():
    assert build_corpus() is build_corpus()
    assert build_corpus(refresh=True) is not None


# --- Ledger provenance ----------------------------------------------------

def test_claim_records_what_decided_it(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "LEDGER", tmp_path / "ledger.json")
    recovery.record_recovery(_claim())
    entry = json.loads((tmp_path / "ledger.json").read_text())[0]

    # Without these, a disputed claim cannot be reconstructed.
    assert entry["model"] == "claude-sonnet-5"
    assert entry["prompt_version"] == "reverse-map/v2"
    assert entry["corpus_version"] == "abc123def456"
    assert entry["candidates_considered"] == ["GLV-N100", "ACM-GLV-L", "GAU-4404"]
    assert entry["rationale"] == "order specifies powder-free"
    assert entry["matched_source"] == "GPO overlay 2025"
    assert entry["recoverable"] == 1320.0


def test_agent_and_human_decisions_are_distinguishable(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "LEDGER", tmp_path / "ledger.json")
    recovery.record_recovery(_claim("PO-AUTO"))
    recovery.record_recovery(_claim("PO-HUMAN", human_confirmed=True))
    entries = {e["order_id"]: e for e in json.loads((tmp_path / "ledger.json").read_text())}
    assert entries["PO-AUTO"]["decided_by"] == "agent"
    assert entries["PO-HUMAN"]["decided_by"] == "human"


def test_guardrail_flags_reach_the_ledger(tmp_path, monkeypatch):
    """If a guardrail fired on a line a human then approved, the claim says so."""
    monkeypatch.setattr(recovery, "LEDGER", tmp_path / "ledger.json")
    recovery.record_recovery(
        _claim(human_confirmed=True, guardrail_flags=["injection_suspected"]))
    entry = json.loads((tmp_path / "ledger.json").read_text())[0]
    assert entry["guardrail_flags"] == ["injection_suspected"]


def test_concurrent_claims_stay_idempotent(tmp_path, monkeypatch):
    """Idempotency is the one property this module exists for, and concurrency is
    exactly what breaks a naive read-check-append."""
    monkeypatch.setattr(recovery, "LEDGER", tmp_path / "ledger.json")
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: recovery.record_recovery(_claim("PO-DUP")),
                                 range(16)))
    claimed = [o for o in outcomes if o["status"] == "claimed"]
    assert len(claimed) == 1
    assert len(json.loads((tmp_path / "ledger.json").read_text())) == 1
    assert recovery.total_recovered() == 1320.0


# --- Result classification ------------------------------------------------

def test_uncertain_lines_land_in_the_review_queue():
    r = _claim(decision=MatchDecision.UNCERTAIN, human_confirmed=None)
    assert r.needs_human_review


def test_reviewed_lines_leave_the_queue():
    for verdict in (True, False):
        r = _claim(decision=MatchDecision.UNCERTAIN, human_confirmed=verdict)
        assert not r.needs_human_review


def test_genuine_ambiguity_is_separable_from_failure():
    """A wedged API call and a near-duplicate glove contract both land in
    UNCERTAIN, but only one is a question a human can usefully answer."""
    ambiguous = _claim(decision=MatchDecision.UNCERTAIN, human_confirmed=None)
    failed = _claim(decision=MatchDecision.UNCERTAIN, human_confirmed=None,
                    guardrail_flags=["llm_error"])
    assert not ambiguous.blocked_by_guardrail
    assert failed.blocked_by_guardrail
