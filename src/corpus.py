"""Stage 1 — Corpus: messy contract sources -> queryable price index + retrieval.

This is the "Ingest" + "make it queryable" phase from Resolvd's case study. The
hard truth it embodies: contracted prices live in wildly different formats (a
pipe-delimited GPO table, a prose local-agreement letter, a chatty email that
changes one price). All of it must become one searchable index.

What's built for you:
  - parsers for each messy source -> a unified list[ContractPrice]
  - a keyword/token similarity retriever so the whole pipeline RUNS WITHOUT AN
    API KEY (great for tests + fast iteration)

What's YOURS (# TODO):
  - the embedding-based retriever (semantic match). The keyword one will miss
    "foley cath" ~ "Foley Catheter" style gaps; embeddings fix that. Build it,
    then compare the two — that comparison IS an interview talking point.

Interview tradeoff: "Why retrieve candidates before calling the LLM to adjudicate?"
-> grounding (the LLM only chooses among real contract lines, can't invent a
price), cost (you don't stuff 10k contract lines into every prompt), and latency.
Retrieval narrows; the LLM judges. Same pattern as good RAG.
"""
from __future__ import annotations

import re
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


def build_corpus() -> list[ContractPrice]:
    """Ingest all messy sources into one unified price index.

    Note: later sources OVERRIDE earlier ones for the same SKU (the email
    addendum's $3.60 beats the GPO's $4.20 for CTH-F16). That ordering is a
    deliberate policy — newest price wins. Be ready to defend it.
    """
    corpus: dict[str, ContractPrice] = {}
    ordered_sources = [
        _parse_gpo_overlay((CONTRACTS / "gpo_overlay.txt").read_text()),
        _parse_local_agreement((CONTRACTS / "local_agreement.txt").read_text()),
        _parse_email_addendum((CONTRACTS / "email_addendum.txt").read_text()),
    ]
    for rows in ordered_sources:
        for cp in rows:
            corpus[cp.sku] = cp  # later wins
    return list(corpus.values())


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


def retrieve_semantic(query: str, corpus: list[ContractPrice], k: int = 3
                      ) -> list[CandidateMatch]:
    """Embedding-based retrieval.

    # TODO(you): implement. Embed each contract's (description + sku) once, embed
    # the query, rank by cosine similarity. Use sentence-transformers locally or
    # an embeddings API. Then compare results vs retrieve_keyword on PO-5003
    # ("foley cath 16fr two way" vs "Foley Catheter 16Fr 2-way") — keyword will
    # underrate it, embeddings won't. That delta is your talking point.
    """
   # Module-level cache so we load the model + embed the corpus only ONCE,
# not on every order. (Loading the model is slow; doing it per-call would crawl.)
_model = None
_corpus_cache = None
_corpus_embeddings = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


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
