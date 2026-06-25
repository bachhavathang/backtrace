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

**2. Retrieve first, then let the model judge.**
The agent retrieves a few candidate contract lines before any LLM call. This
grounds the model (it can only choose among real contracted prices — it can't
invent one), cuts cost (no stuffing the whole corpus into a prompt), and cuts
latency. The retrieval score is *not* the final answer; it just narrows the field.

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

**5. A human gate guards every recovery claim.**
Uncertain matches pause for human confirmation before any claim is recorded,
because the cost of a wrong claim is asymmetric. In production this becomes a
review queue; here it's a CLI confirmation.

**6. Recovery claims are idempotent and audited.**
Claims are keyed on order ID — re-running the scan never double-counts. Every
claim records the matched SKU, the source document, both prices, the confidence,
and whether a human confirmed it.

## Two modes, one engine

- **Backward (recovery):** sweep historical non-catalog orders, total the
  recoverable dollars. (`python main.py`)
- **Forward (monitor):** the same reverse-map engine runs on each *new* order at
  the moment the PO is cut, surfacing the contracted price to the buyer before the
  money goes out the door.

## The sample data (designed to exercise judgment)

| Order | Challenge | Intended outcome |
|---|---|---|
| PO-5001 | synonym-y glove description | match Medline GLV-N100 |
| PO-5002 | abbreviated, partial SKU hint | match Cardinal SYR-L10 |
| PO-5003 | price lives only in an email addendum | match at the *new* $3.60 |
| PO-5004 | two valid glove contracts, different prices | **uncertain → escalate** |
| PO-5005 | nothing in any contract | no match, stays manual |
| PO-5006 | price only in the prose local agreement | match Acme DRP-LG-2 |

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
