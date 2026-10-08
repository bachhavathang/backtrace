"""Step 2: how often does retrieval put the right product in the shortlist?

    python -m evals.retrieval_recall                    # newest snapshot, seed-7 orders
    python -m evals.retrieval_recall --keyword          # add the keyword control
    python -m evals.retrieval_recall --split all

No API calls. This is the ceiling on everything downstream: if the right product
is not among the k candidates the model is shown, no prompt, model tier or
threshold can recover it. Recall@3 on the 7-line synthetic corpus was 100% by
arithmetic; this measures it against 24,717 real products.

Setup, matching how the orders were built (evals/real_orders.py):
  - the corpus searched is the snapshot MINUS the held-out no_contract products;
  - a hit is the right PRODUCT anywhere in the top k (an equivalence set — the
    same product on another contract counts);
  - the headline is the TEST split, whose products were never used for tuning and
    whose orders include degradation operators never seen in the tune split.

Also reported, because they set the other two retrieval knobs:
  - top-1 similarity for orders whose product IS in the corpus vs orders whose
    product is NOT — the evidence for no_match_bar, which short-circuits low-
    similarity orders to NO_MATCH without a model call. Every true match below
    that bar is money silently skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

from src.config import DATA, RETRIEVAL_K, THRESHOLDS
from src.config import HYBRID_SEMANTIC_WEIGHT
from src.corpus import (_get_model, _load_or_embed, _top_products, bm25_index, fuse,
                        product_key, retrieve_keyword)
from src.ingest import load_all, snapshot_dir

KS = (1, 3, 5, 10, 20)
METHODS = ("semantic", "bm25", "hybrid")


def evaluate(corpus, dataset: dict, split: str, keyword: bool = False) -> dict:
    """Ranks for every order in `split`; pure aside from the embedding model."""
    from sentence_transformers import util

    exclude = frozenset(tuple(k) for k in dataset["held_out_product_keys"])
    orders = [o for o in dataset["orders"]
              if split == "all" or o["label"]["split"] == split]

    emb = _load_or_embed(corpus)
    model = _get_model()
    queries = [o["raw_description"] for o in orders]
    q_emb = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True,
                         batch_size=64)
    sims = util.cos_sim(q_emb, emb.to(q_emb.device))

    bm25 = bm25_index(corpus)
    records = []
    for i, o in enumerate(orders):
        want = tuple(o["label"]["product_key"])
        sem_scores = sims[i].tolist()
        sem_ranked = sims[i].argsort(descending=True).tolist()
        bm = bm25.scores(o["raw_description"])
        bm_ranked = sorted(bm, key=bm.get, reverse=True)
        rankings = {"semantic": sem_ranked, "bm25": bm_ranked,
                    "hybrid": fuse([(1.0, bm_ranked), (HYBRID_SEMANTIC_WEIGHT, sem_ranked)])}
        rec = {
            "order_id": o["order_id"], "kind": o["label"]["kind"],
            "split": o["label"]["split"], "ops": o["label"]["ops"],
            "query": o["raw_description"],
        }
        for method, ranked in rankings.items():
            cands = _top_products(ranked, sem_scores, corpus, max(KS), exclude)
            keys = [product_key(c.contract) for c in cands]
            rec[f"rank_{method}"] = keys.index(want) + 1 if want in keys else None
            if method == "semantic":
                rec["top1_similarity"] = cands[0].similarity if cands else 0.0
        rec["rank"] = rec["rank_semantic"]
        if keyword and o["label"]["kind"] != "no_contract":
            kc = retrieve_keyword(o["raw_description"], corpus, k=max(KS), exclude=exclude)
            kk = [product_key(c.contract) for c in kc]
            rec["keyword_rank"] = kk.index(want) + 1 if want in kk else None
        records.append(rec)
    return {"records": records, "n_corpus_rows": len(corpus), "excluded_products": len(exclude)}


def recall(records: list[dict], field: str = "rank") -> dict[int, float]:
    pos = [r for r in records if r["kind"] != "no_contract"]
    if not pos:
        return {k: 0.0 for k in KS}
    return {k: sum(1 for r in pos if r.get(field) and r[field] <= k) / len(pos) for k in KS}


def _row(label: str, rec: dict[int, float], n: int) -> str:
    return f"  {label:28} n={n:<4} " + "  ".join(f"@{k}={rec[k]:6.1%}" for k in KS)


def report(result: dict, keyword: bool) -> dict:
    records = result["records"]
    pos = [r for r in records if r["kind"] != "no_contract"]
    neg = [r for r in records if r["kind"] == "no_contract"]
    print(f"Corpus {result['n_corpus_rows']} rows, {result['excluded_products']} products held out\n")
    print("Recall — is the right product in the top k?")
    for method in METHODS:
        print(_row(f"{method} (all positives)", recall(records, f"rank_{method}"), len(pos)))
    if keyword:
        print(_row("jaccard keyword (today's control)", recall(records, "keyword_rank"), len(pos)))
    overall = recall(records, "rank_hybrid")
    print("\nBreakdown, hybrid:")
    for kind in ("match_seller", "match_maker", "savings"):
        sub = [r for r in records if r["kind"] == kind]
        if sub:
            print(_row(f"  kind={kind}", recall(sub, "rank_hybrid"), len(sub)))
    held = [r for r in pos if set(r["ops"]) & {"typo", "rephrase_pack"}]
    seen = [r for r in pos if not set(r["ops"]) & {"typo", "rephrase_pack"}]
    if held:
        print(_row("  with held-out operators", recall(held, "rank_hybrid"), len(held)))
        print(_row("  without", recall(seen, "rank_hybrid"), len(seen)))
    for op in ("truncate", "drop_words", "abbreviate", "shuffle", "typo", "rephrase_pack"):
        sub = [r for r in pos if op in r["ops"]]
        if sub:
            print(_row(f"  op={op}", recall(sub, "rank_hybrid"), len(sub)))

    bar = THRESHOLDS.no_match_bar
    p1 = [r["top1_similarity"] for r in pos]
    n1 = [r["top1_similarity"] for r in neg]
    print(f"\nTop-1 similarity (no_match_bar = {bar})")
    if p1:
        print(f"  in corpus      median {median(p1):.3f}   below bar: "
              f"{sum(s < bar for s in p1)}/{len(p1)}  <- skipped without a model call")
    if n1:
        print(f"  not in corpus  median {median(n1):.3f}   below bar: "
              f"{sum(s < bar for s in n1)}/{len(n1)}  <- correctly short-circuited")

    misses = [r for r in pos if not r["rank_hybrid"] or r["rank_hybrid"] > RETRIEVAL_K]
    print(f"\nHybrid misses at current RETRIEVAL_K={RETRIEVAL_K}: {len(misses)}/{len(pos)}. Examples:")
    for r in misses[:8]:
        print(f"  {r['order_id']} rank={r['rank_hybrid']}  ops={','.join(r['ops'][1:])}"
              f"  | {r['query'][:60]}")
    return {"recall": {m: recall(records, f"rank_{m}") for m in METHODS},
            "n_positive": len(pos), "n_negative": len(neg)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--orders", type=Path, default=None,
                    help="defaults to data/derived/<snapshot>/orders-seed7.json")
    ap.add_argument("--split", choices=("test", "tune", "all"), default="test")
    ap.add_argument("--keyword", action="store_true", help="also run the keyword control")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    snap = args.snapshot or snapshot_dir()
    from src.sources.va_nac import VaNacSource
    corpus, _ = load_all([VaNacSource(snap)])
    orders_path = args.orders or DATA / "derived" / snap.name / "orders-seed7.json"
    if not orders_path.exists():
        print(f"No orders at {orders_path}. Run: python -m evals.real_orders")
        return 1
    dataset = json.loads(orders_path.read_text())
    print(f"Snapshot {snap.name}, orders {orders_path.name}, split={args.split}")
    result = evaluate(corpus, dataset, args.split, args.keyword)
    summary = report(result, args.keyword)
    if args.json:
        args.json.write_text(json.dumps({**summary, "records": result["records"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
