# CLAUDE.md — Backtrace build guide

## What this is
A toy of Resolvd's actual flagship workflow: **non-catalog spend recovery via
reverse-mapping**. When a hospital orders a "non-catalog" item, it's paid at LIST
price — even when that exact item is already under contract somewhere (a GPO
overlay, a local agreement, an email addendum). Backtrace ingests every messy
contract source into a queryable corpus, then for each non-catalog order asks:
*is this item under contract somewhere, and at what price?* — and computes the
recoverable dollars.

This mirrors Resolvd's case study: $49M scanned -> $12M found under contract but
paid at list -> $1.5-2M recoverable. The output we measure is DOLLARS.

## The one rule
The value is the **reverse-map agent** (fuzzy-match a vague order line to a
contracted SKU, with a confidence-gated decision) and your ability to NARRATE the
tradeoffs. Not pretty data, not a pretty UI. If tempted to polish those, stop and
go deepen the matching/decision logic instead.

## Pipeline
```
order line (vague, no clean SKU)
      │
      ▼
  RETRIEVE candidates from contract corpus   (src/corpus.py — built for you)
      │
      ▼
  REVERSE-MAP: adjudicate best match + confidence   (src/agent.py — YOUR CORE)
      │
      ├──(confident match)────▶ RECOVER: list vs contract = $ recovered ──▶ ledger
      ├──(uncertain)──────────▶ HUMAN GATE ──(confirm)──▶ recover ──▶ ledger
      └──(no match)───────────▶ log as "no contract found", keep manual
```

## Layout
```
src/schema.py      Pydantic models. The contract between stages, plus provenance.
src/config.py      Model tiers, thresholds, budgets. ALL tunables live here.
src/prompts.py     System prompt (the cached prefix) + per-order suffix + schema.
src/guardrails.py  Input sanitising, output validation, escalation policy.
src/llm.py         The gateway. The ONLY place a model is called.
src/corpus.py      Messy sources -> one price index + semantic retrieval.
src/agent.py       The reverse-map graph and the confidence policy.
src/recovery.py    Dollar math + idempotent, audited ledger.
evals/             Labelled set + metrics + threshold sweep.
```

## Invariants — do not break these
- **No price ever enters a prompt.** The model decides whether two items are the
  same; `(list - contract) x qty` is plain Python. Any change that puts a dollar
  figure in front of the model breaks the auditability of every claim.
- **The model picks an index, not a SKU.** It chooses from a shortlist it did not
  select. `guardrails.validate_verdict` cross-checks the echoed SKU against the
  one actually shown at that index. This caps the blast radius of prompt injection.
- **Failure abstains, never no-matches.** A timeout, refusal, or malformed verdict
  becomes UNCERTAIN with a flag — never NO_MATCH, which would silently drop money.
- **A guardrail flag outranks confidence.** Anything flagged escalates at any
  confidence. See `agent.decide()`; the check order there is load-bearing.
- **All model calls go through `llm.adjudicate()`.** Do not construct an
  `Anthropic()` client anywhere else — it bypasses the call log, and a claim with
  no provenance is not defensible.
- **`agent.decide()` stays pure.** The eval harness replays it across a threshold
  grid to produce the precision-vs-escalation curve without re-calling the API.

## Tradeoffs to be able to defend
- corpus:   why retrieve before the LLM? (grounding / cost / latency) Why keep the
            deterministic price math OUT of the LLM?
- agent:    fuzzy match across messy text - how do you avoid false positives?
            (a wrong contract match = a false recovery claim against a vendor =
            expensive + trust-destroying). How is confidence derived & gated?
- prompts:  why is the system prompt LONG? (caching does not engage below 1024
            tokens; a terse prompt is uncacheable AND weaker on domain shorthand)
- evals:    why not report accuracy? (the two error types are not symmetric)
- recovery: idempotency + audit — why does a recovery claim need a paper trail?
- forward:  recovery (backward) vs monitor (forward) — same engine, two modes.

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python -m src.generate_data

python main.py                      # backward scan, then the review queue
python main.py --preflight          # cache eligibility check, makes no calls
python -m evals.run_eval --offline  # guardrail + retrieval checks, no key
python -m evals.run_eval --sweep    # full eval + threshold curve
pytest -q                           # deterministic layers, no key needed
```
