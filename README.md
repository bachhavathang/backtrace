# Backtrace — a non-catalog spend recovery agent

Backtrace is a working slice of the flagship workflow Resolvd describes in its
case study: **reverse-mapping non-catalog spend**. When a hospital orders an item
"non-catalog," it pays **list price** — even when that exact item is already under
contract somewhere (a GPO overlay, a local agreement, an email addendum). Catching
it by hand means digging through hundreds of contract documents. Backtrace ingests
all of them into a queryable corpus, reverse-maps each non-catalog order to its
contracted price, and computes the recoverable dollars — then watches new orders
going forward.

The output it measures is **dollars**, not hours saved.

## The pipeline

```
order line (vague text, no clean SKU)
      │
      ▼
  RETRIEVE candidates from the contract corpus
      │
      ▼
  REVERSE-MAP  — adjudicate the true match + a calibrated confidence
      │
      ├─ confident ─▶ RECOVER  (list − contract) × qty  ─▶ audited ledger
      ├─ uncertain ─▶ HUMAN GATE ─ confirm ─▶ recover ─▶ ledger
      └─ no match ──▶ logged, stays manual
```

## Design decisions (and the tradeoffs behind them)

**1. One queryable corpus out of many messy formats.**
Contracted prices live in a pipe-delimited GPO table, a prose local-agreement
letter, and a chatty email that changes a single price. Each gets its own parser
into one unified index. When two sources price the same SKU, the **newest source
wins** — the email addendum's $3.60 overrides the GPO's $4.20. Provenance is kept
on every price, because a recovery claim has to point at the document it came from.

**2. Retrieve to narrow, then let the LLM adjudicate — never trust one alone.**
Retrieval (semantic embeddings via `all-MiniLM-L6-v2`) produces a shortlist of
candidate contract lines; an LLM then picks the true match from that shortlist or
declares it ambiguous. Each tool does what it's good at: retrieval is a *recall*
tool (cast a wide net cheaply), the LLM is a *precision* tool (fine distinctions
between near-duplicates). This division is load-bearing, not decorative — see the
"Why retrieve-then-judge" note below for the concrete failure that motivated it.
The LLM only ever chooses among real candidates by SKU, so it cannot invent a
price, and it never sees a dollar figure at all.

**3. The deterministic money math never touches the LLM.**
Recovery dollars are `(list − contract) × quantity`, computed in plain code. The
LLM decides *whether two items are the same*; it never decides *how much money is
owed*. That boundary keeps every dollar figure reproducible and auditable.

**4. A false positive is worse than a miss.**
This is the core risk. Wrongly mapping an order to a contract means filing a
recovery claim for money that isn't owed — a credibility hit with both the
customer and the vendor. So the bar for an automatic claim is deliberately high,
and genuine ambiguity is escalated, never guessed. The clearest example in the
sample data: two different glove contracts ($9.10 Medline, $8.40 Acme) both match
"exam gloves nitrile lg" equally well. The agent must send that to a human, not
silently pick one.

**5. The human gate disambiguates — it doesn't just rubber-stamp.**
When the agent is merely unsure, the gate is a yes/no confirm. But when it
*abstains* on genuine ambiguity (two equally-good contracts), a yes/no is
meaningless — so the gate presents the candidate contracts with their prices and
makes the human *choose which one*. Only then are prices filled and the recovery
recorded. The audit trail keeps both the human's choice and the source document.

**6. Recovery claims are idempotent and audited.**
Claims are keyed on order ID — re-running the scan never double-counts. Every
claim records the matched SKU, the source document, both prices, the confidence,
and whether a human confirmed it.


## Why retrieve-then-judge (a concrete failure)

The two glove contracts are near-duplicates at different prices: Medline GLV-N100
"Nitrile Exam Gloves, **Powder-Free**, Large, box/100" at $9.10, and Acme ACM-GLV-L
"Nitrile exam gloves, large, box of 100" at $8.40.

- **Keyword retrieval** scored both gloves *identically* for order PO-5001
  ("nitrile gloves large pf box 100") — a dead tie, no way to choose.
- **Semantic retrieval** broke the tie, but ranked them by general language
  similarity and put the *wrong* glove on top (Acme, 0.798 vs Medline 0.742) — a
  razor-thin 0.056 margin. It under-weighted "pf", which is domain shorthand for
  powder-free that the embedding model was never trained on.
- **The LLM adjudicator** — given the shortlist plus the hint that "pf" means
  powder-free — correctly picked Medline, the powder-free contract.

The lesson: retrieval alone cannot be trusted for a money decision on
near-duplicates, and a hand-tuned similarity threshold is fragile (0.056 barely
clearing a 0.05 cutoff is luck, not safety). The robustness comes from *composing*
the tools — retrieval narrows, the LLM judges with domain context, and genuine
ambiguity escalates to a human with an auditable reason.

## Two modes, one engine

- **Backward (recovery):** sweep historical non-catalog orders, total the
  recoverable dollars. (`python main.py`)
- **Forward (monitor):** the same reverse-map engine runs on each *new* order at
  the moment the PO is cut, surfacing the contracted price to the buyer before the
  money goes out the door.

## The sample data (designed to exercise judgment)

| Order | Challenge | Outcome |
|---|---|---|
| PO-5001 | "pf" shorthand; retrieval ranks the wrong glove | match Medline GLV-N100 — LLM uses domain hint |
| PO-5002 | abbreviated, partial SKU hint | match Cardinal SYR-L10 |
| PO-5003 | price lives only in an email addendum | match at the *new* $3.60, not the old $4.20 |
| PO-5004 | two glove contracts, no distinguishing detail | uncertain → human picks the contract |
| PO-5005 | nothing in any contract | confident no-match, never escalated |
| PO-5006 | price only in the prose local agreement | match Acme DRP-LG-2 |

A clean run recovers **$4,450** across the matched orders ($1,320 + $320 + $360 +
$1,000 auto-matched, plus $1,450 from the human-resolved PO-5004).

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python -m src.generate_data
python main.py          # backward scan over all non-catalog orders
pytest -q               # deterministic layers, no key needed
```

## What I'd build next for production

- Semantic (embedding) retrieval to catch abbreviation/synonym gaps the keyword
  retriever misses, with a retrieval-quality eval set.
- Confidence calibration so the auto-claim threshold is data-driven, not guessed.
- LangGraph interrupts + a real review queue for the human gate.
- Item-master write-back so confirmed matches update the catalog (Resolvd's
  "item-master automation" expansion).
- Per-vendor match-accuracy tracking over time.
