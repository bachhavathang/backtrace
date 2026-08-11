"""Stage 1 — Corpus: messy contract sources -> queryable price index + retrieval.

This is the "Ingest" + "make it queryable" phase from Resolvd's case study. The
hard truth it embodies: contracted prices live in wildly different formats (a
pipe-delimited GPO table, a prose local-agreement letter, a chatty email that
changes one price). All of it must become one searchable index.

What lives here:
  - parsers for each messy source -> a unified list[ContractPrice], newest
    source winning when two of them price the same SKU
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
from pathlib import Path

from .schema import CandidateMatch, ContractPrice

CONTRACTS = Path(__file__).resolve().parent.parent / "data" / "contracts"


# --- Parsers: each messy source -> ContractPrice rows --------------------

def _parse_gpo_overlay(text: str) -> list[ContractPrice]:
    rows: list[ContractPrice] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 4 and parts[0].lower() != "vendor" and not parts[0].startswith("GPO"):
            vendor, sku, desc, price = parts
            try:
                rows.append(ContractPrice(sku=sku, description=desc, vendor=vendor,
                                          contracted_unit_price=float(price),
                                          source="GPO overlay 2025"))
            except ValueError:
                continue
    return rows


def _parse_local_agreement(text: str) -> list[ContractPrice]:
    """Prose letter: pull '(Vendor #SKU): $price' style lines."""
    rows: list[ContractPrice] = []
    pattern = re.compile(r"-\s*(.+?)\s*\((\w[\w\s]*?)\s*#([\w\-]+)\):\s*\$([\d.]+)")
    for m in pattern.finditer(text):
        desc, vendor, sku, price = m.groups()
        rows.append(ContractPrice(sku=sku.strip(), description=desc.strip(),
                                  vendor=vendor.strip(),
                                  contracted_unit_price=float(price),
                                  source="Local agreement - St. Mark's"))
    return rows


def _parse_email_addendum(text: str) -> list[ContractPrice]:
    """Email that changes a price: find a SKU code + a $price near it."""
    rows: list[ContractPrice] = []
    sku_m = re.search(r"\b([A-Z]{3}-[A-Z0-9]+)\b", text)
    price_m = re.search(r"\$([\d.]+)", text)
    if sku_m and price_m:
        rows.append(ContractPrice(
            sku=sku_m.group(1),
            description="Foley Catheter 16Fr 2-way (email price update)",
            vendor="Cardinal", contracted_unit_price=float(price_m.group(1)),
            source="Email addendum 4/12"))
    return rows


_corpus: list[ContractPrice] | None = None
_corpus_lock = threading.Lock()


def build_corpus(refresh: bool = False) -> list[ContractPrice]:
    """Ingest all messy sources into one unified price index. Cached after first call.

    Note: later sources OVERRIDE earlier ones for the same SKU (the email
    addendum's $3.60 beats the GPO's $4.20 for CTH-F16). That ordering is a
    deliberate policy — newest price wins. Be ready to defend it.

    The cache matters more than it looks. This used to be called once per order
    line from inside the retrieve node, so a scan of N orders re-read and
    re-regex-parsed three files N times before doing any useful work. Contracts
    do not change during a scan; pass refresh=True if they did.
    """
    global _corpus
    if _corpus is not None and not refresh:
        return _corpus

    with _corpus_lock:
        if _corpus is not None and not refresh:
            return _corpus
        corpus: dict[str, ContractPrice] = {}
        ordered_sources = [
            _parse_gpo_overlay((CONTRACTS / "gpo_overlay.txt").read_text()),
            _parse_local_agreement((CONTRACTS / "local_agreement.txt").read_text()),
            _parse_email_addendum((CONTRACTS / "email_addendum.txt").read_text()),
        ]
        for rows in ordered_sources:
            for cp in rows:
                corpus[cp.sku] = cp  # later wins
        _corpus = list(corpus.values())
    return _corpus


def corpus_version(corpus: list[ContractPrice] | None = None) -> str:
    """A short content hash of the price index.

    Stamped onto every recovery claim. Contract prices change — the email
    addendum in this very corpus moves CTH-F16 from $4.20 to $3.60 — so a claim
    has to record which version of the index it was computed against, or you
    cannot explain a year later why two claims for the same SKU differ.
    """
    corpus = corpus if corpus is not None else build_corpus()
    payload = "|".join(
        f"{c.sku}:{c.contracted_unit_price}:{c.source}"
        for c in sorted(corpus, key=lambda c: c.sku)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


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
