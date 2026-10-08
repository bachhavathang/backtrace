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

from .config import CORPUS_SOURCE, EMBED_CACHE
from .ingest import IngestReport, load_all, sources_for
from .schema import CandidateMatch, ContractPrice
from .sources.openfda import normalise_part

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


# --- Products --------------------------------------------------------------
#
# One physical product can sit on many contract rows: an FSS and a BPA price for
# the same item, or several resellers carrying one maker's part. In the 24,883-line
# VA corpus, 878 SKUs appear on more than one row. Retrieved row by row, a shortlist
# of three can be "the same glove x3", and the adjudicator — correctly told to
# abstain when two candidates fit equally — abstains on every common item (G6).
#
# So the shortlist is three PRODUCTS. product_key() errs towards keeping rows
# apart: merging two different products would put one product's price on the
# other (a false claim), while failing to merge two copies only costs an
# abstention (a human look). Rows join one product only when they share a
# normalised part number AND either the registry maker or the exact description.

def identity_string(row: ContractPrice) -> str:
    """A row's identity_key as one string, for provenance and lookup."""
    return "|".join(row.identity_key)


def product_key(row: ContractPrice) -> tuple[str, str]:
    part = normalise_part(row.sku) or row.sku
    maker = (row.manufacturer or "").strip().upper()
    if maker:
        return part, "maker:" + maker
    return part, "desc:" + " ".join(row.description.lower().split())


_groups_for: tuple[int, int] | None = None
_groups: dict[tuple[str, str], list[ContractPrice]] = {}
_groups_lock = threading.Lock()


def product_groups(corpus: list[ContractPrice]) -> dict[tuple[str, str], list[ContractPrice]]:
    """Every product in the corpus, with all of its contract rows. Built once per corpus."""
    global _groups_for, _groups
    ident = (id(corpus), len(corpus))
    if _groups_for != ident:
        with _groups_lock:
            if _groups_for != ident:
                groups: dict[tuple[str, str], list[ContractPrice]] = {}
                for row in corpus:
                    groups.setdefault(product_key(row), []).append(row)
                _groups, _groups_for = groups, ident
    return _groups


def _top_products(ranked: list[int], scores: list[float], corpus: list[ContractPrice],
                  k: int) -> list[CandidateMatch]:
    """Walk rows best-first, keeping the best row of each product until k products."""
    groups = product_groups(corpus)
    seen: set[tuple[str, str]] = set()
    out: list[CandidateMatch] = []
    for i in ranked:
        row = corpus[i]
        key = product_key(row)
        if key in seen:
            continue
        seen.add(key)
        group = groups[key]
        out.append(CandidateMatch(contract=row, similarity=round(float(scores[i]), 3),
                                  group=group if len(group) > 1 else []))
        if len(out) == k:
            break
    return out


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
    scores = []
    for cp in corpus:
        c = _tokens(cp.description + " " + cp.sku)
        scores.append(round(len(q & c) / len(q | c), 3) if (q | c) else 0.0)
    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    return _top_products(ranked, scores, corpus, k)


# Module-level cache so we load the model + embed the corpus only ONCE,
# not on every order. (Loading the model is slow; doing it per-call would crawl.)
# The lock matters once a scan runs orders concurrently: without it, eight
# workers starting together would each load their own copy of the model.
EMBED_MODEL = "all-MiniLM-L6-v2"
_model = None
_corpus_cache: tuple[int, int] | None = None
_corpus_embeddings = None
# Re-entrant: retrieve_semantic holds it while embedding the corpus, and embedding
# needs the model, whose first load takes the same lock. A plain Lock deadlocked
# there on the first query of every run.
_embed_lock = threading.RLock()


def _get_model():
    global _model
    if _model is None:
        with _embed_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(EMBED_MODEL)
    return _model


def warm_retrieval() -> None:
    """Load the embedding model and embed the corpus before the scan starts.

    Pure latency management: the first retrieval otherwise pays a one-off model
    load (seconds) that would land on whichever order happened to go first and
    look like a slow order rather than a slow startup.
    """
    retrieve_semantic("warmup", build_corpus(), k=1)


def _embedding_texts(corpus: list[ContractPrice]) -> list[str]:
    return [c.description + " " + c.sku for c in corpus]


def embedding_cache_path(texts: list[str]):
    """Where the vectors for exactly these texts, from exactly this model, live.

    Keyed on a hash of the texts themselves. The old in-memory key was the tuple
    of SKUs, which cannot tell two rows sharing a SKU apart and does not change
    when a description does.
    """
    payload = chr(31).join([EMBED_MODEL, *texts]).encode("utf-8")
    return EMBED_CACHE / f"{EMBED_MODEL}-{hashlib.sha256(payload).hexdigest()[:16]}.npy"


def _load_or_embed(corpus: list[ContractPrice]):
    """Corpus vectors, from disk when this exact corpus was embedded before (G12).

    Embedding 25k lines on a CPU takes minutes; doing it on every run and every
    eval made the real corpus impractical. The cache is only ever an optimisation:
    any read or write failure falls through to embedding in memory.
    """
    import numpy as np
    import torch

    texts = _embedding_texts(corpus)
    path = embedding_cache_path(texts)
    try:
        arr = np.load(path)
        if arr.shape[0] == len(texts):
            return torch.from_numpy(arr)
    except (OSError, ValueError):
        pass
    emb = _get_model().encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                              batch_size=128, show_progress_bar=len(texts) > 2000)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb.cpu().numpy())
    except OSError:
        pass
    return emb


def retrieve_semantic(query: str, corpus: list[ContractPrice], k: int = 3
                      ) -> list[CandidateMatch]:
    """Embedding-based retrieval. Ranks by cosine similarity of MEANING."""
    global _corpus_cache, _corpus_embeddings
    import torch
    from sentence_transformers import util

    ident = (id(corpus), len(corpus))
    if _corpus_cache != ident:
        with _embed_lock:
            if _corpus_cache != ident:
                _corpus_embeddings = _load_or_embed(corpus)
                _corpus_cache = ident

    q_emb = _get_model().encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, _corpus_embeddings.to(q_emb.device))[0]  # one per row
    ranked = torch.argsort(scores, descending=True).tolist()
    return _top_products(ranked, scores.tolist(), corpus, k)
