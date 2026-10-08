"""Tests for BM25, rank fusion, hybrid retrieval and the recall metric. No network.

The embedding model is the deterministic fake from test_products, so these check
the plumbing — fusion, fill, exclusion — not vector quality.
"""
import pytest

from evals.retrieval_recall import recall
from src import corpus as corpus_mod
from src.schema import ContractPrice
from fakes import FakeModel


def _row(sku, desc, holder="V"):
    return ContractPrice(sku=sku, description=desc, vendor=holder, holder=holder,
                         contracted_unit_price=1.0, source="s", contract_id="c-" + sku)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(corpus_mod, "_model", FakeModel())
    monkeypatch.setattr(corpus_mod, "_corpus_cache", None)
    monkeypatch.setattr(corpus_mod, "_bm25_for", None)
    monkeypatch.setattr(corpus_mod, "EMBED_CACHE", tmp_path)


GLOVES = [_row(f"G{i}", f"nitrile exam glove large box {i}") for i in range(20)] + \
         [_row("AQ-1", "aquacel extra dressing 3/4x18"), _row("TB-9", "extension tubing 18 inch")]


# --- BM25 -------------------------------------------------------------------------

def test_bm25_weights_rare_words_over_common_ones():
    bm = corpus_mod.BM25(GLOVES)
    scores = bm.scores("aquacel glove")
    best = max(scores, key=scores.get)
    assert GLOVES[best].sku == "AQ-1"      # "aquacel" is rare; "glove" is on 20 lines


def test_bm25_ignores_unknown_words():
    assert corpus_mod.BM25(GLOVES).scores("zzzz qqqq") == {}


# --- Fusion -----------------------------------------------------------------------

def test_fuse_is_rank_based_and_weighted():
    a, b = [1, 2, 3], [3, 2, 1]
    assert corpus_mod.fuse([(1.0, a), (0.1, b)])[0] == 1    # the heavier ranking leads
    assert corpus_mod.fuse([(0.1, a), (1.0, b)])[0] == 3


def test_fuse_skips_zero_weight():
    assert corpus_mod.fuse([(1.0, [5, 6]), (0.0, [7, 8])]) == [5, 6]


# --- Hybrid retrieval ----------------------------------------------------------------

def test_hybrid_finds_the_distinctive_product(fake):
    top = corpus_mod.retrieve_hybrid("aquacel drsg extra", GLOVES, k=3)
    assert top[0].contract.sku == "AQ-1"


def test_hybrid_returns_k_products_even_when_bm25_matches_few(fake):
    # One word matches nothing lexically; the semantic order fills the shortlist.
    rows = [_row("A", "alpha"), _row("B", "beta"), _row("C", "gamma")]
    assert len(corpus_mod.retrieve_hybrid("unrelated", rows, k=3)) == 3


def test_hybrid_similarity_is_still_cosine(fake):
    cands = corpus_mod.retrieve_hybrid("aquacel", GLOVES, k=2)
    assert all(-1.0 <= c.similarity <= 1.0 for c in cands)


def test_exclude_hides_a_whole_product(fake):
    key = corpus_mod.product_key(GLOVES[-2])
    top = corpus_mod.retrieve_hybrid("aquacel", GLOVES, k=5, exclude=frozenset({key}))
    assert all(c.contract.sku != "AQ-1" for c in top)


# --- The metric ------------------------------------------------------------------------

def test_recall_counts_positives_only():
    recs = [{"kind": "match_seller", "rank": 1}, {"kind": "savings", "rank": 4},
            {"kind": "match_maker", "rank": None}, {"kind": "no_contract", "rank": None}]
    r = recall(recs)
    assert r[1] == pytest.approx(1 / 3) and r[5] == pytest.approx(2 / 3)
