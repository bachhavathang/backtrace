"""Tests for the deterministic layers (no LLM key needed).

Run: pytest -q

Covers corpus parsing (the three messy formats), keyword retrieval sanity, the
recovery dollar math, and ledger idempotency. We do NOT assert on the LLM
adjudication here — that's eval territory, not unit-test territory. Good interview
line: "I test the deterministic core hard; I eval the probabilistic parts."
"""
import os

from src import config, recovery
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


# --- .env loading ---------------------------------------------------------
# The credential path is worth testing because every failure here is a confusing
# one: a stray quote or a BOM turns into a 401 that looks like a bad key.

def _write_env(tmp_path, body):
    p = tmp_path / ".env"
    p.write_text(body, encoding="utf-8")
    return p


def test_dotenv_sets_missing_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("BT_TEST_KEY", raising=False)
    assert config.load_dotenv(_write_env(tmp_path, "BT_TEST_KEY=abc123\n")) == 1
    assert os.environ["BT_TEST_KEY"] == "abc123"


def test_real_env_var_beats_the_file(tmp_path, monkeypatch):
    # Precedence is the whole design: the file is a fallback, never the authority.
    monkeypatch.setenv("BT_TEST_KEY", "from-shell")
    config.load_dotenv(_write_env(tmp_path, "BT_TEST_KEY=from-file\n"))
    assert os.environ["BT_TEST_KEY"] == "from-shell"


def test_dotenv_ignores_placeholder_and_comments(tmp_path, monkeypatch):
    # An unfilled `KEY=` must read as absent, so main.py prints its "no
    # credential" help instead of sending an empty key and getting a 401.
    monkeypatch.delenv("BT_TEST_KEY", raising=False)
    monkeypatch.delenv("BT_TEST_OTHER", raising=False)
    body = "# a comment\n\nBT_TEST_KEY=\nBT_TEST_OTHER=fine\nnot-a-pair\n"
    assert config.load_dotenv(_write_env(tmp_path, body)) == 1
    assert "BT_TEST_KEY" not in os.environ
    assert os.environ["BT_TEST_OTHER"] == "fine"


def test_dotenv_strips_quotes_export_and_bom(tmp_path, monkeypatch):
    monkeypatch.delenv("BT_TEST_KEY", raising=False)
    monkeypatch.delenv("BT_TEST_EXPORTED", raising=False)
    p = tmp_path / ".env"
    p.write_text('BT_TEST_KEY="quoted"\nexport BT_TEST_EXPORTED=yes\n',
                 encoding="utf-8-sig")  # utf-8-sig writes the BOM Windows adds
    assert config.load_dotenv(p) == 2
    assert os.environ["BT_TEST_KEY"] == "quoted"
    assert os.environ["BT_TEST_EXPORTED"] == "yes"


def test_dotenv_missing_file_is_not_an_error(tmp_path):
    # Offline runs, CI and the test suite all have no .env. That is normal.
    assert config.load_dotenv(tmp_path / "nope.env") == 0


def test_dotenv_accepts_a_str_path(tmp_path, monkeypatch):
    # The docstring promises this never raises; a str path used to hit
    # AttributeError on .read_text before the Path() coercion.
    monkeypatch.delenv("BT_TEST_KEY", raising=False)
    assert config.load_dotenv(str(_write_env(tmp_path, "BT_TEST_KEY=v\n"))) == 1
    assert config.load_dotenv("no/such/path/.env") == 0
