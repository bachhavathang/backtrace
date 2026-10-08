"""Realistic non-catalog order lines, generated from a real contract snapshot.

    python -m evals.real_orders                         # newest snapshot, 600 orders
    python -m evals.real_orders --n 1000 --seed 11

Real hospital purchase orders are not public (docs/WORKING_STATE.md §3C), so this
builds them the defensible way: start from a REAL contract description and degrade
it the way purchasing staff type — abbreviate, drop words, shuffle, truncate. The
contract side is real; only the degradation is modelled, and every order records
exactly which operators made it.

Each order carries a hidden answer key ("label") that the pipeline never reads:

    kind            match_seller   the seller holds the contract       -> claimable (A)
                    match_maker    the maker holds it, a distributor   -> claimable (B)
                                   delivered
                    savings        someone else holds it               -> savings, never a claim (C)
                    no_contract    the product was held out of the     -> no match
                                   corpus the pipeline searches
    product_key     the right answer as an EQUIVALENCE set — the product group,
                    so finding the same product on another contract counts (G11)
    split           tune | test. Split by product, so no product is in both, and
                    two operators (typo, pack rephrasing) appear only in test —
                    a score tuned against one style of degradation cannot pass
                    for a score on unseen styles.

Two things are synthetic and labelled as such: the list price (a markup over the
contract price) and the supplier/maker names on savings and maker cases, which
are fictional distributors. No dollar total computed from these orders is a
real recovery figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from src.config import DATA
from src.corpus import identity_string, product_groups, product_key
from src.ingest import load_all, snapshot_dir
from src.recovery import CLAIMABLE, _same_party, select_contract_price
from src.schema import ContractPrice, OrderLine

DERIVED = DATA / "derived"

# Fictional distributors and makers, so no real company is attached to a
# fabricated purchase. "Owens & Minor" appears in the docs as an example only.
DISTRIBUTORS = ("Northwind Medical Supply", "Harbor Clinical Distribution",
                "Summit Health Logistics", "Bluepeak Surgical Supply")
OTHER_MAKERS = ("Corvane Medical", "Halden Health Products", "Ostrand Surgical")

# Purchasing shorthand: the inverse of the system prompt's glossary.
ABBREVIATIONS = {
    "gloves": "glv", "glove": "glv", "powder-free": "pf", "powder free": "pf",
    "powderfree": "pf", "large": "lg", "medium": "md", "small": "sm",
    "extra large": "xl", "x-large": "xl", "catheter": "cath", "catheters": "cath",
    "syringe": "syr", "syringes": "syr", "sterile": "ster", "non-sterile": "nonster",
    "examination": "exam", "box": "bx", "case": "cs", "each": "ea", "package": "pk",
    "pack": "pk", "french": "fr", "dressing": "drsg", "sponge": "spg",
    "sponges": "spg", "surgical": "surg", "disposable": "disp", "needle": "ndl",
    "tubing": "tbg", "with": "w/", "without": "w/o", "latex": "ltx",
}

TUNE_OPS = ("lowercase", "abbreviate", "drop_words", "shuffle", "truncate")
HELD_OUT_OPS = ("typo", "rephrase_pack")


# --- Degradation operators -----------------------------------------------------

def op_lowercase(text: str, rng: random.Random) -> str:
    return re.sub(r"[,;:()]", " ", text.lower())


def op_abbreviate(text: str, rng: random.Random) -> str:
    for full, short in sorted(ABBREVIATIONS.items(), key=lambda kv: -len(kv[0])):
        if rng.random() < 0.8:
            text = re.sub(rf"\b{re.escape(full)}\b", short, text, flags=re.I)
    return text


def _words(text: str) -> list[str]:
    return text.split()


def op_drop_words(text: str, rng: random.Random) -> str:
    """Drop up to a third of the words, never numbers (sizes and counts identify)."""
    words = _words(text)
    if len(words) <= 3:
        return text
    droppable = [i for i, w in enumerate(words) if not re.search(r"\d", w)]
    drop = set(rng.sample(droppable, min(len(droppable), max(1, len(words) // 3))))
    kept = [w for i, w in enumerate(words) if i not in drop]
    return " ".join(kept if len(kept) >= 3 else words[:3])


def op_shuffle(text: str, rng: random.Random) -> str:
    words = _words(text)
    if len(words) < 3:
        return text
    i = rng.randrange(0, len(words) - 2)
    window = words[i:i + 3]
    rng.shuffle(window)
    return " ".join(words[:i] + window + words[i + 3:])


def op_truncate(text: str, rng: random.Random) -> str:
    words = _words(text)
    return " ".join(words[:max(3, min(len(words), rng.randint(4, 7)))])


def op_typo(text: str, rng: random.Random) -> str:
    words = _words(text)
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def op_rephrase_pack(text: str, rng: random.Random) -> str:
    """'100/BX' -> 'bx of 100' or 'box 100' — the same fact, unseen wording."""
    def swap(m: re.Match) -> str:
        n, unit = m.group(1), m.group(2).lower()
        return rng.choice([f"{unit} of {n}", f"{unit} {n}", f"{n}{unit}"])
    return re.sub(r"\b(\d{1,5})\s*/\s*([A-Za-z]{2,5})\b", swap, text)


OPS = {"lowercase": op_lowercase, "abbreviate": op_abbreviate, "drop_words": op_drop_words,
       "shuffle": op_shuffle, "truncate": op_truncate, "typo": op_typo,
       "rephrase_pack": op_rephrase_pack}


def degrade(description: str, ops: list[str], rng: random.Random) -> str:
    text = description
    for name in ops:
        text = OPS[name](text, rng)
    text = " ".join(text.split())
    return text or description.lower()


# --- Order construction ----------------------------------------------------------

def split_for(key: tuple[str, str]) -> str:
    """Deterministic by product, so a product never lands in both halves."""
    return "test" if hashlib.sha256("|".join(key).encode()).digest()[0] % 2 else "tune"


def _pick_ops(split: str, rng: random.Random) -> list[str]:
    ops = ["lowercase"] + rng.sample(TUNE_OPS[1:], rng.randint(1, 3))
    if split == "test" and rng.random() < 0.6:
        ops += rng.sample(HELD_OUT_OPS, rng.randint(1, 2))
    return ops


def _effective_date(row: ContractPrice, rng: random.Random):
    if row.effective_start and row.effective_end and row.effective_end > row.effective_start:
        span = (row.effective_end - row.effective_start).days
        return row.effective_start + timedelta(days=rng.randrange(0, span))
    return row.effective_start


def _maker_holds(row: ContractPrice) -> bool:
    return bool(row.manufacturer) and _same_party(row.holder or row.vendor, row.manufacturer)


@dataclass
class Plan:
    match_seller: float = 0.55
    match_maker: float = 0.10
    savings: float = 0.20
    no_contract: float = 0.15


def make_order(n: int, kind: str, row: ContractPrice, rng: random.Random) -> dict:
    key = product_key(row)
    split = split_for(key)
    ops = _pick_ops(split, rng)
    holder = row.holder or row.vendor
    if kind == "match_seller":
        supplier, maker = holder, row.manufacturer
    elif kind == "match_maker":
        supplier, maker = rng.choice(DISTRIBUTORS), row.manufacturer
    elif kind == "savings":
        supplier = rng.choice(DISTRIBUTORS)
        maker = (row.manufacturer if row.manufacturer and not _maker_holds(row)
                 else rng.choice(OTHER_MAKERS))
    else:  # no_contract
        # The REAL holder, not a fictional distributor. The dangerous real case is
        # buying an off-contract variant from the very vendor that holds contracts
        # on its siblings ("with stylet" when only "without" is contracted): the
        # holder rule passes, so only the match decision stands between that order
        # and a false claim. A fictional seller let the holder rule hide it — the
        # Step 3 pilot found the model matching such siblings at 0.85-0.90.
        supplier, maker = holder, row.manufacturer
    markup = rng.uniform(1.10, 1.80)
    return {
        "order_id": f"RO-{n:05d}",
        "raw_description": degrade(row.description, ops, rng),
        "quantity": rng.randint(1, 50),
        "list_unit_price": round(row.contracted_unit_price * markup, 2),
        "supplier": supplier,
        "manufacturer": maker,
        "effective_date": (d.isoformat() if (d := _effective_date(row, rng)) else None),
        "label": {
            "kind": kind, "product_key": list(key), "source_row": identity_string(row),
            "split": split, "ops": ops, "list_price_synthetic": True,
        },
    }


def generate(corpus: list[ContractPrice], n: int, seed: int, plan: Plan = Plan()) -> dict:
    """Orders plus the product keys held out of the corpus for no_contract cases."""
    rng = random.Random(seed)
    groups = product_groups(corpus)
    keys = sorted(groups)
    rng.shuffle(keys)

    counts = {k: round(n * getattr(plan, k)) for k in ("match_seller", "match_maker",
                                                       "savings", "no_contract")}
    held_out = keys[:counts["no_contract"]]
    pool = keys[counts["no_contract"]:]
    maker_pool = [k for k in pool if any(_maker_holds(r) for r in groups[k])]
    # A savings case must have NO row its maker holds, anywhere in the product's
    # group. The first generated set built one from a reseller's row while the maker
    # held its own contract for the same product — a real case B, labelled C. The
    # claim-rule audit caught it as a "false claim"; the label was what was wrong.
    savings_pool = [k for k in pool if k not in set(maker_pool)]

    orders: list[dict] = []
    for k in held_out:
        orders.append(make_order(len(orders), "no_contract", groups[k][0], rng))
    for kind, candidates in (("match_seller", pool), ("savings", savings_pool)):
        for k in rng.sample(candidates, min(len(candidates), counts[kind])):
            orders.append(make_order(len(orders), kind, rng.choice(groups[k]), rng))
    for k in rng.sample(maker_pool, min(len(maker_pool), counts["match_maker"])):
        row = next(r for r in groups[k] if _maker_holds(r))
        orders.append(make_order(len(orders), "match_maker", row, rng))

    # Interleave kinds, so a budget-capped partial run is a fair sample and not
    # "the no-contract cases first" (which is what the first Step 3 pilot got).
    rng.shuffle(orders)
    for i, o in enumerate(orders):
        o["order_id"] = f"RO-{i:05d}"
    return {"seed": seed, "n_requested": n, "held_out_product_keys": [list(k) for k in held_out],
            "orders": orders}


# --- The claim-rule audit -----------------------------------------------------------

def audit(corpus: list[ContractPrice], dataset: dict, as_of: date) -> dict:
    """Run the settled claim rule on every order against its true product's rows.

    Retrieval and the model are skipped on purpose: this checks the PRICE rule in
    isolation. The one hard requirement is that no savings or no-contract order
    ever comes out CLAIMABLE — that would be a false claim against a vendor.
    """
    groups = product_groups(corpus)
    table: dict[str, dict[str, int]] = {}
    false_claims = []
    for o in dataset["orders"]:
        label = o["label"]
        # A no_contract product is held out of the corpus the pipeline searches,
        # so it has no rows to price against — by construction, not by lookup.
        rows = ([] if label["kind"] == "no_contract"
                else groups.get(tuple(label["product_key"]), []))
        order = OrderLine(**{k: v for k, v in o.items() if k != "label"})
        status = select_contract_price(order, rows, as_of).status if rows else "no_rows"
        table.setdefault(label["kind"], {}).setdefault(status, 0)
        table[label["kind"]][status] += 1
        if status == CLAIMABLE and label["kind"] in ("savings", "no_contract"):
            false_claims.append(o["order_id"])
    return {"by_kind": table, "false_claims": false_claims}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--snapshot", type=Path, default=None)
    args = ap.parse_args(argv)

    snap = args.snapshot or snapshot_dir()
    from src.sources.va_nac import VaNacSource
    corpus, _ = load_all([VaNacSource(snap)])
    dataset = generate(corpus, args.n, args.seed)
    dataset["snapshot"] = snap.name
    out = DERIVED / snap.name / f"orders-seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, indent=1))

    kinds: dict[str, int] = {}
    splits: dict[str, int] = {}
    for o in dataset["orders"]:
        kinds[o["label"]["kind"]] = kinds.get(o["label"]["kind"], 0) + 1
        splits[o["label"]["split"]] = splits.get(o["label"]["split"], 0) + 1
    print(f"{len(dataset['orders'])} orders from {snap.name} -> {out}")
    print(f"  kinds  {kinds}\n  splits {splits}")
    report = audit(corpus, dataset, date.today())
    print(f"  claim-rule audit {report['by_kind']}")
    print(f"  false claims {len(report['false_claims'])}")
    return 1 if report["false_claims"] else 0


if __name__ == "__main__":
    sys.exit(main())
