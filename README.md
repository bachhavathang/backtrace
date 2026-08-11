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
  SANITISE      untrusted PO text; injection attempts flagged
      │
      ▼
  RETRIEVE      shortlist of 3 from the contract corpus (local embeddings)
      │
      ├─ nothing clears the floor ──▶ NO_MATCH, no model call at all
      ▼
  ADJUDICATE    via the LLM gateway; tier chosen by how close the top two are
      │
      ▼
  VALIDATE      index in range, echoed SKU cross-checked, confidence in [0,1]
      │
      ├─ confident ─▶ RECOVER  (list − contract) × qty  ─▶ audited ledger
      ├─ uncertain ─▶ REVIEW QUEUE ─ human confirms ─▶ recover ─▶ ledger
      └─ no match ──▶ logged, stays manual
```

Everything above the ledger is in `src/`; the layers that make it operable —
gateway, guardrails, evals — are described under [Production
concerns](#production-concerns-and-how-each-one-is-handled).

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

## Production concerns, and how each one is handled

### Evaluation — `evals/`

The deterministic layers are unit-tested; the adjudicator is a judgement call over
messy text, so it is **evaluated against a labelled set** rather than asserted on.
39 cases in `evals/dataset.json`, weighted toward the cases that produce false
claims rather than the easy ones:

| Category | n | What it tests |
|---|---|---|
| `clear` | 14 | One contract line is unambiguously right. Auto-match expected. |
| `ambiguous` | 4 | Two contracts fit equally well at different prices. Escalation expected. |
| `absent` | 8 | Nothing in the corpus is this product. |
| `trap` | 11 | Superficially near-identical but conflicting on one attribute — 14Fr vs 16Fr, 5 mL vs 10 mL, non-sterile vs sterile, box/50 vs box/100. Retrieval scores these high, which is exactly why they are the risk. |
| `injection` | 4 | Order text carrying instruction-like content. |

Accuracy is deliberately **not** the headline number, because the two error types
are not symmetric. A **false claim** demands money from a vendor that isn't owed
and costs credibility with the customer and the vendor at once. A **miss** costs
one line of savings — which is exactly the status quo of a human never finding it.
So the harness reports three families and refuses to blend them:

- **Safety** — false claims (must be 0), auto-match precision, injection coverage.
- **Effectiveness** — auto-match recall, the money found without a human.
- **Efficiency** — escalation rate. A system that escalates everything is
  perfectly safe and worthless, which is why this sits next to the other two.

Retrieval recall@k is reported separately because it is a hard ceiling: if the
right contract line isn't in the shortlist, no amount of prompting recovers it.

The **threshold sweep** (`--sweep`) is what turns `high_bar = 0.85` from a magic
number into a defended one. `agent.decide()` is a pure function of
`(verdict, thresholds)`, so the harness adjudicates once and then replays the
recorded verdicts across a grid of bars — the whole precision-vs-escalation curve
for zero extra API calls.

```bash
python -m evals.run_eval --offline   # guardrails + retrieval, no key needed
python -m evals.run_eval --sweep     # full run + the threshold curve
```

### Security and guardrails — `src/guardrails.py`

The threat model is specific: a purchase-order description is free text from
outside the hospital's control, and adjudicating it produces a **financial claim**.
An attacker who can influence order text has a motive — induce a confident match
to the wrong contract and cause a bogus claim. Five layers, weakest to strongest:

1. **Sanitise** — NFKC-normalise, strip control characters, collapse newlines, cap
   length, neutralise the `<order_text>` fence so a payload can't close it early.
2. **Delimit** — order text is fenced and the system prompt declares everything
   inside it to be data. Standard, and bypassable alone, which is why it isn't last.
3. **Constrain** — *the layer that actually bounds the damage.* The model answers
   with an **index into a shortlist it did not choose**, and **no price is ever in
   the prompt**. The best possible injection can only move the answer between three
   real contract lines retrieval already selected. It cannot invent a SKU, cannot
   invent a price, and cannot reach the ledger.
4. **Verify** — every field re-derived in code. Index in range; echoed SKU
   cross-checked against the one actually shown at that index; confidence a real
   number in [0,1]. A hallucinated SKU is caught here even at confidence 1.0.
5. **Escalate** — any violation, and any suspected injection, downgrades to "needs
   a human" regardless of confidence. Nothing that tripped a guardrail is ever
   auto-claimed.

Layers 1–2 raise the cost of an attack; layer 3 caps the payoff; 4–5 make failure
safe rather than silent. Current state: **4/4 injections caught, 0 false positives
across 37 benign order lines.**

### Token optimisation

| Lever | Effect |
|---|---|
| Retrieve before prompting | The corpus could hold 10k contract lines; the prompt sees 3. ~75x fewer input tokens than stuffing the corpus. |
| Retrieval floor short-circuit | Lines below the similarity floor return NO_MATCH with **no model call at all**. On real non-catalog spend most lines genuinely aren't under contract, so this is the largest single saving. |
| Prompt caching | The system prompt is a stable ~1,500-token prefix with the cache breakpoint on it; only the order + shortlist vary. Cached tokens bill at ~10%. |
| Model tiering | A runaway top candidate goes to Haiku 4.5; a photo-finish between two contracts at different prices buys Sonnet 5 — the close calls are precisely where false claims come from. |
| Structured outputs | A fixed JSON schema, so no retry-on-parse loop and a bounded 300-token ceiling. |

The caching decision is worth spelling out, because "make the prompt shorter" is
the wrong instinct here. Caching does **not** engage below a model-specific minimum
prefix (1024 tokens), and the API doesn't error when you're under it — it accepts
the `cache_control` marker, ignores it, and bills every call at full price. A terse
200-token prompt is *uncacheable*, so shortening it would have saved a little per
call and forfeited the ~90% discount on the shared prefix. The prompt is
deliberately substantial instead — the domain glossary in it is also what lets the
adjudicator beat retrieval on "pf" — and `main.py --preflight` **measures** the
prefix rather than assuming:

```bash
python main.py --preflight   # reports the real token count vs the cache minimum
```

`cache_hit_rate` is in every scan report for the same reason: a silent cache miss
has no other symptom.

### Latency management

- **Concurrent scan.** Order lines are independent; the scan runs them through a
  thread pool. Sequential is still available (`--sequential`) for debugging.
- **Deferred human gate.** A scan of ten thousand orders cannot stop on order three
  waiting for someone to type `y`. Uncertain lines are queued and reviewed *after*
  the scan, in a separate phase.
- **Build-once caching.** The LangGraph is compiled once, the corpus is parsed
  once, the embedding model is loaded once. All three used to happen per order.
- **Warm start.** The embedding model is loaded before the clock starts, so the
  first order doesn't absorb a multi-second one-off cost and look pathologically slow.
- **Explicit timeout** (30s) and bounded retries, and mean/p50/p95/max per-call
  latency in the report — a p95 is the number that tells you whether a scan will
  finish, and a mean hides it.

### LLM gateway — `src/llm.py`

Every model call goes through `adjudicate()`. Nothing else constructs a client,
names a model, or parses a response. That chokepoint buys:

- **Provenance.** Every call appended to `data/logs/llm_calls.jsonl` with model,
  prompt version, request id, latency, token usage and cost. Every *claim* records
  the model, prompt version, corpus version, the candidate SKUs that were on the
  table, and whether a guardrail fired. "The model said so" is not an answer to a
  disputed claim; this is.
- **Model pinning.** Model ids live in `config.py`. An upgrade is a deliberate
  change that re-runs the eval suite, not a silent behaviour shift.
- **Reliability.** One client, explicit timeout, bounded retries — and, more
  importantly, a decision about what a *failed* adjudication means. It means
  abstain and escalate. Never "no match", which would silently drop money.
- **Cost and cache visibility.** Usage accumulated per tier, so a scan reports what
  it spent. Recovering $4,450 is a different claim depending on whether it cost
  $0.01 or $400.
- **Safety.** Raw model output never leaves the module unvalidated.

`adjudicate()` never raises. Transport failure, rate limit, safety refusal,
unparseable output, hallucinated SKU — all five return an abstention carrying the
flag that explains it, so the caller's correct move is identical in every case.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # PowerShell: $env:ANTHROPIC_API_KEY = '...'
python -m src.generate_data

python main.py                      # backward scan (concurrent, then review)
python main.py --preflight          # prompt-cache eligibility; makes no calls
python main.py --no-review          # scan only, leave the queue open
python main.py --sequential         # no concurrency, easier to follow

python -m evals.run_eval --offline  # guardrail + retrieval checks, no key
python -m evals.run_eval --sweep    # full eval + threshold curve
pytest -q                           # 81 tests, no key needed
```

## Layout

```
src/config.py      model tiers, thresholds, budgets — all tunables in one place
src/prompts.py     system prompt (cached prefix) + per-order suffix + schema
src/guardrails.py  input sanitising, output validation, escalation policy
src/llm.py         the gateway: the only place a model is called
src/corpus.py      messy sources -> one price index + semantic retrieval
src/agent.py       the reverse-map graph and the confidence policy
src/recovery.py    dollar math + idempotent, audited ledger
evals/             labelled set + metrics + threshold sweep
```

## What I'd build next

- **Calibration curve.** Bucket predictions by claimed confidence and check whether
  0.9-confidence matches are right ~90% of the time. Models are usually
  overconfident, which is the assumption the current bars rest on.
- **Batch API for the backward scan.** A historical sweep has nobody waiting on it,
  so it should take the 50% batch discount; per-call latency only matters in
  forward-monitor mode.
- **Item-master write-back** so confirmed matches update the catalog (the
  "item-master automation" expansion).
- **Per-vendor match-accuracy tracking** over time, fed from the call log.
- **A real review UI** on top of the queue, replacing the stdin prompt.
