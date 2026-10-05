"""Stage 1 — Corpus: messy contract sources -> queryable price index + retrieval.

This is the "Ingest" + "make it queryable" phase from Resolvd's case study. The
hard truth it embodies: contracted prices live in wildly different formats (a
pipe-delimited GPO table, a prose local-agreement letter, a chatty email that
changes one price). All of it must become one searchable index.

What lives here:
  - build_corpus(): the configured sources (src/ingest.py) -> one unified
    list[ContractPrice], with amendments of a contract collapsed
  - two retrievers over that index:
      retrieve_keyword   token overlap. No model, no key — the pipeline and the
                         whole test suite run without either.
      retrieve_semantic  sentence embeddings (all-MiniLM-L6-v2). What the scan
                         actually uses; the model and corpus vectors are built
                         once and shared across concurrent workers.

Keeping both is deliberate: the keyword retriever is the control that shows why
the semantic one earns its cost. It misses "foley cath" ~ "Foley Catheter", and
on PO-5001 it scores the two rival glove contracts *identically* — a dead tie no
threshold can break. Embeddings separate them, but by a 0.056 margin and with the
wrong glove on top, because "pf" (powder-free) is domain shorthand the embedding
model never saw. Neither retriever is trustworthy alone on a money decision.

Which is the answer to "why retrieve at all, if an LLM adjudicates anyway?" —
retrieval is a recall tool and the LLM is a precision tool. Retrieval narrows
10k contract lines to 3 (grounding: the model can only choose among real
contracts, so it cannot invent a price; plus the cost and latency of not
stuffing the corpus into every prompt). The LLM then makes the fine distinction
retrieval got wrong. Composing them is what's robust; a tuned similarity cutoff
on either one alone is not.
"""
from __future__ import annotations

import hashlib
import re
import threading

from .config import CORPUS_SOURCE
from .ingest import IngestReport, load_all, sources_for
from .schema import CandidateMatch, ContractPrice

_corpus: list[ContractPrice] | None = None
_report: IngestReport | None = None
_corpus_lock = threading.Lock()


def build_corpus(refresh: bool = False) -> list[ContractPrice]:
    """Ingest all messy sources into one unified price index. Cached after first call.

    Which files or feeds are read is config.CORPUS_SOURCE's business (see
    src/ingest.py). Within one contract the newest amendment wins — the email
    addendum's $3.60 replaces Cardinal's GPO $4.20 for CTH-F16 because it amends
    that same contract. Across contracts nothing is overridden: two contracts
    pricing one SKU are two rows, and recovery.select_contract_price() decides.

    The cache matters more than it looks. This used to be called once per order
    line from inside the retrieve node, so a scan of N orders re-read and
    re-regex-parsed three files N times before doing any useful work. Contracts
    do not change during a scan; pass refresh=True if they did.
    """
    global _corpus, _report
    if _corpus is not None and not refresh:
        return _corpus

    with _corpus_lock:
        if _corpus is not None and not refresh:
            return _corpus
        _corpus, _report = load_all(sources_for(CORPUS_SOURCE))
    return _corpus


def ingest_report() -> IngestReport:
    """What the last build_corpus() read, kept, dropped and flagged."""
    build_corpus()
    return _report


def corpus_version(corpus: list[ContractPrice] | None = None) -> str:
    """A short content hash of the price index.

    Stamped onto every recovery claim. Contract prices change — the email
    addendum in this very corpus moves CTH-F16 from $4.20 to $3.60 — so a claim
    has to record which version of the index it was computed against, or you
    cannot explain a year later why two claims for the same SKU differ.

    Units, dates and the verification flag change what a claim is worth, so they
    are hashed too — but only when set, which keeps the hash of a corpus that
    has none of them (the synthetic one) identical to what older claims carry.
    """
    corpus = corpus if corpus is not None else build_corpus()
    payload = "|".join(
        f"{c.sku}:{c.contracted_unit_price}:{c.source}" + _hash_extras(c)
        for c in sorted(corpus, key=lambda c: (c.sku, c.source))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _hash_extras(c: ContractPrice) -> str:
    extras = [f"{name}={value}" for name, value in (
        ("uom", c.uom), ("pack", c.units_per_pack), ("from", c.effective_start),
        ("to", c.effective_end), ("unverified", c.needs_verification or None),
    ) if value is not None]
    return (":" + ",".join(extras)) if extras else ""


# --- Retrieval -----------------------------------------------------------

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def retrieve_keyword(query: str, corpus: list[ContractPrice], k: int = 3
                     ) -> list[CandidateMatch]:
    """Token-overlap (Jaccard-ish) retrieval. Works with no API key.

    Good enough to run the pipeline and tests; deliberately weak on synonyms and
    abbreviations so you can SEE why embeddings matter.
    """
    q = _tokens(query)
    scored: list[CandidateMatch] = []
    for cp in corpus:
        c = _tokens(cp.description + " " + cp.sku)
        overlap = len(q & c) / len(q | c) if (q | c) else 0.0
        scored.append(CandidateMatch(contract=cp, similarity=round(overlap, 3)))
    scored.sort(key=lambda m: m.similarity, reverse=True)
    return scored[:k]


# Module-level cache so we load the model + embed the corpus only ONCE,
# not on every order. (Loading the model is slow; doing it per-call would crawl.)
# The lock matters once a scan runs orders concurrently: without it, eight
# workers starting together would each load their own copy of the model.
_model = None
_corpus_cache = None
_corpus_embeddings = None
_embed_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _embed_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def warm_retrieval() -> None:
    """Load the embedding model and embed the corpus before the scan starts.

    Pure latency management: the first retrieval otherwise pays a one-off model
    load (seconds) that would land on whichever order happened to go first and
    look like a slow order rather than a slow startup.
    """
    retrieve_semantic("warmup", build_corpus(), k=1)


def retrieve_semantic(query: str, corpus: list[ContractPrice], k: int = 3
                      ) -> list[CandidateMatch]:
    """Embedding-based retrieval. Ranks by cosine similarity of MEANING."""
    global _corpus_cache, _corpus_embeddings
    from sentence_transformers import util

    model = _get_model()

    # Embed the corpus once and reuse it. We key the cache on the SKUs present,
    # so if the corpus changes we rebuild.
    corpus_key = tuple(c.sku for c in corpus)
    if _corpus_cache != corpus_key:
        with _embed_lock:
            if _corpus_cache != corpus_key:
                texts = [c.description + " " + c.sku for c in corpus]
                _corpus_embeddings = model.encode(texts, convert_to_tensor=True,
                                                  normalize_embeddings=True)
                _corpus_cache = corpus_key

    q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, _corpus_embeddings)[0]  # one score per contract

    scored = [
        CandidateMatch(contract=corpus[i], similarity=round(float(scores[i]), 3))
        for i in range(len(corpus))
    ]
    scored.sort(key=lambda m: m.similarity, reverse=True)
    return scored[:k]
