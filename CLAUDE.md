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

## Build order (do not reorder)
1. `src/schema.py`        — Pydantic models. The contract between stages. (built)
2. `src/generate_data.py` — messy contract sources + non-catalog orders. (built)
3. `src/corpus.py`        — ingest messy sources -> queryable price index + retrieval. (built, ONE todo)
4. `src/agent.py`         — reverse-map: retrieve -> adjudicate -> gate -> recover. (YOUR CORE)
5. `src/recovery.py`      — list-vs-contract dollar math + idempotent ledger. (built)
6. `app.py`              — optional Streamlit (day 4 only, skippable).

## Where YOUR learning lives (# TODO(you))
- `src/corpus.py`  -> the embedding/similarity call for candidate retrieval.
- `src/agent.py`   -> node_reverse_map (the adjudication + confidence policy),
                      route_after_map (branching), build_graph (LangGraph wiring).
Write these yourself FIRST, then ask the AI to critique. Defending these is the
entire point of the demo.

## Interview tradeoffs to be ready to defend
- corpus:   why a retrieval step before the LLM? (cost/latency/grounding) Why
            keep the deterministic price math OUT of the LLM?
- agent:    fuzzy match across messy text — how do you avoid false positives?
            (a wrong contract match = a false recovery claim against a vendor =
            expensive + trust-destroying). How is confidence derived & gated?
- recovery: idempotency + audit — why does a recovery claim need a paper trail?
- forward:  recovery (backward) vs monitor (forward) — same engine, two modes.

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python -m src.generate_data
python main.py            # runs reverse-map over all non-catalog orders
pytest -q                 # deterministic layers, no key needed
```
