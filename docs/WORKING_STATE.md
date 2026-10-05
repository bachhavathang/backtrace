# Working state — where Backtrace is, and what happens next

**Snapshot date:** 2026-09-02 · **Commit:** `c1a7071` · **Updated:** 2026-10-05 (§5 Step 1 design review) · **Companion to:** [`PRODUCTION_DESIGN.md`](PRODUCTION_DESIGN.md)

This document exists so the next session can start without re-deriving anything. It
records the **verified** state of each production layer (checked against `src/`, not
against the README), the external-dataset research, and the ordered next steps.

Where this document and the README disagree, **this document is right and the README
is stale** — the two disagreements are called out explicitly in §2.

---

## 1. Verified state as of this snapshot

All three checks were run at commit `c1a7071` and pass:

```
$ pytest -q
108 passed in 3.43s

$ python -m evals.run_eval --offline
injection detection   7/7 caught, 0 false positives on 57 benign lines
retrieval  recall@3   100.0%
           recall@1    95.0%

$ python main.py --preflight
claude-haiku-4-5   prefix 1704 tok   minimum 4096   caches: NO   (2392 short)
claude-sonnet-5    prefix 2291 tok   minimum 1024   caches: YES
```

**Read `recall@3 = 100%` with suspicion.** `build_corpus()` returns **7 contract
lines** and `RETRIEVAL_K = 3`, so a shortlist holds 43% of the entire corpus. That
number is arithmetic, not a measurement. It is the single most misleading figure in
the repo and §4 is ordered around fixing it.

The corpus, in full:

| SKU | Description | Source |
|---|---|---|
| `GLV-N100` | Nitrile Exam Gloves, Powder-Free, Large, box/100 | GPO overlay 2025 |
| `GAU-4404` | Gauze Sponge 4x4 12-ply sterile, pk/25 | GPO overlay 2025 |
| `SYR-L10` | Luer-Lok Syringe 10 mL sterile | GPO overlay 2025 |
| `CTH-F16` | Foley Catheter 16Fr 2-way | Email addendum 4/12 |
| `DRP-LG-2` | Sterile surgical drape, fenestrated, large | Local agreement |
| `ESU-PEN` | Electrosurgical pencil, hand control, disposable | Local agreement |
| `ACM-GLV-L` | Nitrile exam gloves, large, box of 100 | Local agreement |

---

## 2. Layer status, verified against the source

| Track | State | Notes |
|---|---|---|
| **Latency** | 6 / 6 | Complete *at prototype scale* |
| **Guardrails** | 5 / 6 | Strong; one real gap |
| **Token cost** | 5.5 / 7 | One measured hole |
| **Gateway** | 6.5 / 9 | Further along than the roadmap assumed |
| **Evaluation** | 5 / 8 | Weakest, and it gates the others |

### Two corrections to the README

**1. `warm_cache()` already exists** — `src/llm.py:377`. It is better than
`PRODUCTION_DESIGN.md` §14 row 7 assumed: it serialises one call per tier before the
fan-out, dedupes by *model* (not `Tier`, which is unhashable), skips tiers whose
prefix cannot clear their own model's minimum, and is gated on
`CACHE_WARM_MIN_BATCH = 12`. **Only the `max_tokens: 0` prefill is missing** — it
currently warms with a real adjudication and pays for output tokens it discards.
Row 7 is *partial*, not *pending*.

**2. Prompt caching engages on one tier of two.** Haiku's 1,704-token prefix never
reaches its 4,096 minimum. The code handles this correctly (`preflight()` measures it,
`warm_cache()` refuses to warm that tier). The README's token-optimisation table reads
as though caching is simply on. It is a **hole, not a feature** — and it is worth
roughly $18/yr, so it must not jump the queue.

### Confirmed absent

- No `--calibration` flag. The eval CLI is `--offline`, `--sweep`, `--category`,
  `--workers`, `--json`.
- No `.github/` directory at all — there is no CI gate.
- Only `FAST` and `PRECISE` in `config.py`; no third `deliberate` tier.
- **Ingest-side sanitising is missing.** `sanitize_order_text()` is applied to order
  text only. Contract descriptions are parsed straight into the corpus unchecked.
  Harmless while the three contract files are hand-written; a live injection surface
  the moment external documents are ingested. This is §14 row 13.

### Current thresholds (`src/config.py`)

```
no_match_bar             0.15      RETRIEVAL_K            3
low_bar                  0.50      REQUEST_TIMEOUT_S      30.0
high_bar                 0.85      MAX_RETRIES            3
fast_tier_min_similarity 0.55      MAX_CONCURRENCY        8
fast_tier_min_margin     0.15      CACHE_WARM_MIN_BATCH   12
```

Every one of these was chosen against the 7-line corpus above. Treat them as
provisional the moment real data lands.

---

## 3. External dataset sourcing

**This is new ground — it is not covered anywhere else in the repo.**
`PRODUCTION_DESIGN.md` §13 covers the ERP boundary only; it says nothing about
sourcing reference data. Everything today comes from `src/generate_data.py`.

Backtrace needs three different kinds of data, and their public availability is
wildly uneven. That asymmetry is the finding.

### A. Product identity — abundant and free

- **[AccessGUDID](https://accessgudid.nlm.nih.gov/download/query)** — the FDA/NLM
  front end to [GUDID](https://www.fda.gov/medical-devices/unique-device-identification-system-udi-system/global-unique-device-identification-database-gudid).
  No account needed. **Monthly Full Release** of all published Device Identifier
  records, plus daily and weekly delta files. Gives device identifier, brand name,
  company, description, packaging and unit-of-measure — exactly the corpus shape.
  The packaging fields would let the **unit-of-measure mismatch** traps be generated
  from real data rather than by hand.
- **[GMDN](https://www.gmdnagency.org/faqs/)** — free registration; free for
  healthcare providers and academic researchers. Multi-hierarchical with mutually
  exclusive terms, which matters: the `trap` category is largely product-class
  near-synonyms, and a nomenclature with genuinely exclusive terms is a labelling
  source for exactly those.
- **UNSPSC** — UNDP-owned, annually updated, licensed. Single-hierarchical, so
  weaker than GMDN for the distinctions the traps turn on.

### B. Contract prices — the hard one

Contract pricing is the confidential core of the business. GPO overlay pricing
(Vizient, Premier, HealthTrust) is contractually protected and will not be publicly
sourced. There is essentially **one** real public source:

- **[VA NAC Contract Catalog Search Tool](https://www.vendorportal.ecms.va.gov/nac/)**
  — updated daily, 1,700+ active contract vehicles, **over 1 million catalog line
  items** with contract pricing and contractor details.
  [Schedule 65 II A](https://www.va.gov/opal/nac/fss/schedule65IIA.asp) is medical
  equipment and supplies: gloves, gauze, catheters, syringes, drapes — the same
  categories as the synthetic corpus.
- **[VA pharmaceutical pricing](https://www.va.gov/opal/nac/fss/pharmprices.asp)**
  (twice monthly) and the
  [VA Drug Pricing Database](https://catalog.data.gov/dataset/va-drug-pricing-database)
  — bulk-downloadable, but drugs, not med-surg.
- Entry point worth checking first: [VA eTools](https://www.va.gov/opal/nac/fss/etoolsVA.asp).

**Caveat, verified:** CCST is publicly *searchable* per query, but an exported
line-item spreadsheet goes through an assigned contract specialist. There is no
advertised clean bulk dump for med-surg the way there is for pharma. Plan on
per-query harvesting or an actual request — not a one-line `wget`.

What CCST gives that synthetic data cannot: genuinely messy real vendor
descriptions, real unit-of-measure conventions, and multiple contract vehicles
pricing the same physical item differently — precisely the ambiguity the human gate
exists to resolve.

### C. Messy non-catalog order lines — this basically does not exist publicly

The *input* side is the real gap, and it is the side that makes the problem hard.
Options in order of honesty:

1. **Derive from B.** Take real CCST descriptions and degrade them programmatically
   — abbreviate, drop attributes, inject domain shorthand, permute word order. The
   `pf` case is this transformation applied by hand. Defensible because the contract
   side is real and the degradation is the modelled part.
2. **Hospital price transparency files — skip these.** They are chargemaster/payer
   rates for patient billing, not supply acquisition prices. Wrong side of the ledger.
3. **The dispute flywheel** (§14 row 15) is the only source of genuinely labelled
   data about *these* contracts, and it cannot start until there is a customer.

**Recommendation:** GUDID full release for identity + CCST Schedule 65 II A for
contract prices + programmatic degradation for order lines. Target **20k–50k lines,
not the full million** — a corpus that stresses retrieval without making every
iteration slow.

---

## 4. The next three steps

> **This supersedes an earlier recommendation in-session of "calibration + CI gate
> first."** That was correct for a static corpus and is wrong now: both calibrate an
> instrument against data that is about to be replaced. The dataset work legitimately
> reorders the roadmap.

```
             ingest              measure the        re-calibrate
             the data            new ceiling        the policy
   ───────────────────────▶──────────────────▶──────────────────▶
   STEP 1               STEP 2              STEP 3
   corpus + sanitise    retrieval recall@k  sweep + calibration
   7 → ~50,000 lines    the hard ceiling    thresholds are stale
   (row 13 rides along) (zero API calls)    (CI gate lands here)
```

### Step 1 — Ingest behind an adapter, with ingest-side sanitising in the same change

Load GUDID (identity) and CCST Schedule 65 II A (prices) through one ingest
interface, keeping `ContractPrice` and its provenance field intact.

**Do §14 row 13 now, not later.** The moment third-party vendor documents are
ingested, contract descriptions become an untrusted input. Retrofitting after 50k
lines are loaded costs far more than generalising `sanitize_order_text()` now.

**Done when:** the corpus builds from real sources with provenance per line; a
tripped injection pattern on a *contract* description sets `needs_verification` and
blocks auto-claim.

> **Revised 2026-10-05 — read §5 before starting.** The original criterion also said
> "`build_corpus()` keeps its signature so nothing downstream needs edits." That is
> **false**: real contracts force schema changes (unit of measure, dates, contract
> identity), a product-level shortlist, and adjustment entries in the ledger. §5 holds
> the design review, the two settled policy decisions, and Step 1 re-cut into four PRs.

### Step 2 — Re-baseline retrieval

The highest-information measurement available, and it **costs zero API calls** —
`--offline` already runs retrieval checks.

Recall@3 against 7 lines told us nothing. Against 50,000 it determines whether the
system can work at all: if the right contract line is not in the shortlist, no
prompt, no model tier and no threshold recovers it. Sweep `k` the way `high_bar` was
swept — recall@1/3/5/10 against real near-duplicates. CCST has many vendors selling
the same glove, so the confusion structure will be far denser than the two-glove case
that motivated the design.

**Done when:** recall@k is reported against the real corpus and `RETRIEVAL_K` is
chosen from a curve instead of being `3`. If recall@10 is materially better than
recall@3, a reranker becomes a real design question — that is a finding, not a
failure.

### Step 3 — Re-run the sweep, add `--calibration`, gate it in CI

Every threshold in `config.py` was chosen on 64 cases against 7 contract lines.
`high_bar = 0.85` was "the lowest bar reaching zero false claims" *in that world*.
With real near-duplicates the confusion structure changes, and the honest expectation
is that **0.85 will move**.

Calibration belongs here rather than earlier because bucketing by confidence only
means something once there are enough cases per bucket — which the small corpus could
never supply. The **CI gate lands with this step**, not before: it is worth most as
the thing protecting the *new* baseline.

**Done when:** the sweep is re-run on real data, `high_bar` is re-derived and the
change explained, `--calibration` reports realised precision per confidence bucket,
and `false_claims == 0` blocks the build.

### Deliberately not in the three

- **Third model tier (§14 row 8), Batch API (row 6), the Haiku cache hole.** Row 8
  ships *only if false claims stay 0* — a verdict from a harness about to be
  re-baselined. Batch and the cache hole are cost work worth ~$18–65/yr against a
  ~$13,500 reviewer bill.
- **Rows 1–2 (DB idempotency, `Decimal` money) are the exception.** They are
  correctness bugs, they are orthogonal to the dataset work, and a bigger corpus makes
  a duplicate claim *more* likely. If there is capacity for four, this is the fourth.
  **Update 2026-10-05:** row 1 is no longer fully orthogonal — Decision 1 (§5.2) needs
  the ledger to accept *adjustment entries*, which a DB-backed ledger should be
  designed for from the start (gap G3).

---

## 5. Step 1 design review (2026-10-05)

Two review passes over Step 1, checked against `src/corpus.py`, `src/schema.py`,
`src/prompts.py`, `src/guardrails.py`, `src/agent.py` and `src/recovery.py`, and
against how hospital purchasing actually works. Organising principle throughout, from
the rest of the design: **a false claim is worse than a missed one.**

### 5.1 Gaps found

**🔴 Blockers — the dollar figure would be wrong**

| # | Gap | Evidence | Fix |
|---|---|---|---|
| G1 | **No unit of measure.** One float price; real contracts price per box/case/each. $9.10/box vs $0.15/each makes `(list − contract) × qty` fabricate a claim. | `schema.py` `ContractPrice` | Add `uom`, `units_per_pack`; normalise to per-each. Unreconcilable UoM → UNCERTAIN. |
| G2 | **Keyed on SKU, last wins.** Same SKU on several contract vehicles, and different vendors reuse part numbers — rows silently discarded. | `corpus.py` `corpus[cp.sku] = cp` | Key on (contract holder, SKU, contract ID). Price selection by explicit rule (§5.2). |
| G3 | **Ledger can't record a follow-up.** One claim per order; a later human-approved top-up returns `noop_already_claimed` and is silently ignored. | `recovery.py` | Append-only **adjustment entries** referencing the original claim. Never mutate. |
| G4 | **No dates.** Contracts start and end; neither `ContractPrice` nor `OrderLine` has a date. | `schema.py` | Add effective dates to contracts, an effective date to orders. |
| G5 | **Distributor ≠ manufacturer.** Hospitals buy Medline gloves *from* Owens & Minor; the contract is with Medline and the distributor honours it via chargeback. A "same vendor" filter rejects most real matches. `OrderLine` has no vendor field at all. | `schema.py` `OrderLine` | Order line carries **billing supplier** and **manufacturer**. Filter contracts on manufacturer; route the claim to the billing supplier. |
| G6 | **Shortlist fills with duplicates.** One glove on 5 contracts → top 3 is "the same glove ×3" → model correctly abstains as ambiguous → every common item escalates. | `agent.py` `node_retrieve`, `prompts.py` | **Group contract rows by physical product** before retrieval. The model picks a product; Python picks the price. |
| G7 | **No join key between GUDID and CCST.** GUDID keys on device identifier; CCST lists vendor part numbers. Join via (manufacturer, catalog number) is plausible but unmeasured. | — | Measure join rate on a sample *before* building on it. GUDID's main job is pack size (G1); a poor join rate changes the plan. |

**🟠 Major — security and measurement**

| # | Gap | Fix |
|---|---|---|
| G8 | **Contract text enters the prompt as trusted.** Rendered raw inside `"…"`, outside the `<order_text>` fence; the system prompt only distrusts the ORDER block. A `"` or newline breaks the line format (`prompts.py:189-193`). | Fence + escape candidate text; update the system prompt to distrust it; cap description length. Sanitising alone is not enough. |
| G9 | **Flag has nowhere to live.** `ContractPrice` has no flag field; `decide()` reads flags on the result only. | Add the field; inject a shortlist flag into the verdict *before* `decide()` so it stays pure and the flag still outranks confidence. |
| G10 | **Sanitiser false-positive rate unmeasured at scale.** "0 FP" is 0/57. Vendor text will contain "override", "review", "mark as". 1% FP = 500 blocked contracts. | Run the patterns over the full corpus pre-ship; budget < 0.1%. |
| G11 | **Eval answers become sets.** 40 vendors sell the identical glove; single-answer recall@k undercounts. Same author writing degradation rules and tuning retrieval inflates scores. List prices on the order side are invented. | Equivalence groups (via GUDID); held-out degradation operators; label all dollar totals **synthetic**. |
| G12 | **Embeddings recomputed at startup**, keyed on a tuple of SKUs (collides once SKUs repeat, G2). 50k lines on CPU = minutes per run and per test. | Persist embeddings to disk keyed on `corpus_version`. |

**🟠 Major — overstated dollars**

| # | Gap | Fix |
|---|---|---|
| G13 | **Which date is "in force"?** PO, ship or invoice date; signed vs effective (retroactive amendments). "Email addendum 4/12" doesn't say which. | Price on **effective** date. Orders within N days of a contract boundary → human. |
| G14 | **Claim windows.** Contracts often bar price-discrepancy claims after ~90–180 days. A backward scan ignores this and overstates. | Report out-of-window money as **found but expired**, not recoverable. |
| G15 | **Already credited.** A credit memo may already exist; claiming again double-collects. | Check against credit records. No such data exists here — state it plainly. |
| G16 | **Price tiers and contract activation.** GPO price depends on the hospital's tier and on having activated the agreement. | The highest-valid-price rule (§5.2) handles tiers conservatively — say so deliberately. Eligibility is assumed in the demo and labelled. |

**🟡 Process**

| # | Gap | Fix |
|---|---|---|
| G17 | **Silent row loss.** `_parse_gpo_overlay` does `except ValueError: continue`. | Every ingest emits a report: read / parsed / rejected-by-reason / deduped / joined / flagged. Quarantine, never drop. |
| G18 | **Validation.** | Price > 0, plausible range, required fields present. |
| G19 | **Reproducibility.** | Snapshot raw files with fetch date + sha256 manifest. Never fetch live in tests/CI. Raw data gitignored; a ~200-row fixture committed. |
| G20 | **Terms of use.** GUDID is public; CCST scraping terms unchecked. | Check before harvesting; rate-limit. |
| G21 | **No rollback.** | `CORPUS_SOURCE` = `synthetic` or `real`; the 108 tests and current eval stay green, real is opt-in. |
| G22 | **Raw vs cleaned text.** | Embed and prompt on cleaned text; keep raw for audit. |
| G23 | **Flood attack.** Planting flagged text across popular contracts escalates everything. | Ingest-time review absorbs it; alert on a jump in flagged count per ingest. |

**Critical-path risk:** CCST bulk export goes through a VA contract specialist. **File
the request on day one.** Fallback: per-query harvest for the five categories already
in the synthetic corpus (gloves, gauze, syringes, catheters, drapes).

### 5.2 Settled decisions

**Decision 1 — which contract price counts.**

1. The model picks a **physical product** from a product-grouped shortlist (G6) —
   never a price, never a contract row.
2. Python filters that product's contract rows to those where the **holder is the
   product's manufacturer** (G5), the contract was **in force on the effective date**
   (G13), and the hospital is **eligible** (assumed in the demo, labelled — G16).
3. Within one contract, the newest amendment wins (today's "newest wins" policy is
   correct *only* inside one contract, never across contracts).
4. If more than one valid price remains, **auto-claim at the highest** — the smallest,
   undisputable claim. The gap to the lowest valid price goes to a human as *possible
   additional recovery*, recorded as an **adjustment entry** if approved (G3).
5. The claim is routed to the **billing supplier**. Out-of-window money is reported as
   **expired** (G14).
6. A cheaper price from a *different* manufacturer is not a recovery — it is a savings
   opportunity, reported separately (forward/monitor mode).

*Why highest, not lowest:* lowest maximises dollars, but if wrong it accuses a vendor
of overcharging by more than they did. Highest gives up some money so that every
automatic claim holds up.

**Decision 2 — flagged contract lines.**

1. If a flagged contract **was shown to the model**, the line cannot auto-claim —
   whether or not it was chosen. Injected text in candidate #2 can steer the choice of
   #1. Never converts to NO_MATCH (failure abstains). A flagged line below the
   retrieval floor, never shown to the model, does not escalate.
2. A human reviews each flagged contract **once at ingest**, not once per order. The
   clearance records reviewer + time and is bound to a **checksum of the text**; it
   resets if the text changes. Flag/clearance state is included in `corpus_version`.
3. The regex is a tripwire. The real defences remain the prompt fence (G8) and
   index-not-SKU validation.

**New fields this requires**

- `OrderLine`: billing supplier, manufacturer, effective date.
- `ContractPrice`: contract holder, contract ID, effective start/end, `uom`,
  `units_per_pack`, claim window, `needs_verification`, raw + cleaned description.

### 5.3 Step 1, re-cut into four PRs

| PR | Change | Behaviour change |
|---|---|---|
| 1 | Schema fields above + ingest adapter interface, synthetic data behind it | None |
| 2 | Contract-side sanitising, prompt fence, flag-in-shortlist escalation (G8–G10) | Security only |
| 3 | GUDID + CCST loaders, snapshots, ingest report, product grouping, persisted embeddings | Opt-in via `CORPUS_SOURCE` |
| 4 | Degraded order lines, equivalence-group labels, held-out operators | Eval only |

Decision 1's price-selection rule and adjustment entries land with PR 1 (logic) and
row 1 of the migration table (DB ledger).

**PR 1 status (2026-10-05, branch `step1-pr1-schema-adapter`):** schema fields,
`src/ingest.py` (`ContractSource`, synthetic sources, `merge_amendments` keyed on
`identity_key`), `config.CORPUS_SOURCE` (env `BACKTRACE_CORPUS_SOURCE`), and
`recovery.select_contract_price()` — pure, tested, **not yet wired** into the agent
(it needs the product-grouped shortlist from PR 3). Synthetic corpus rows, order and
`corpus_version` (`ae5a7592a43d`) pinned unchanged. 129 tests pass.

Carried forward from PR 1:
- `agent.candidates_for()` and the human gate look contracts up **by SKU**
  (`{c.sku: c for c in build_corpus()}`). Correct while SKUs are unique; wrong once
  one SKU sits on two contracts. Re-key on `identity_key` in PR 3.
- `corpus_version` still hashes `sku:price:source` only. Add dates, units and
  flag/clearance state (Decision 2) when those fields carry real data.
- Holder matching is exact after case/whitespace folding; a near-miss goes to a
  human. Vendor entity resolution is PR 3 work.
- Adjustment entries in the ledger are not built — waiting on the DB ledger (row 1).

---

## 6. Housekeeping / loose ends

- [x] **`docs/` is untracked.** *(Done 2026-10-05 — committed.)* Both this file and `PRODUCTION_DESIGN.md` (~45KB,
      ~7k words) are outside git. A prior build attempt on 2026-08-19 died
      mid-generation and left `docs/` empty, nearly losing the settled decisions.
      **Commit this first.**
- [x] `README.md` has uncommitted changes (54 insertions / 9 deletions). *(Done 2026-10-05 — committed.)*
- [x] **CV files are untracked but not ignored** — `Athang_Bachhav_CV.docx`,
      `.docx.bak`, `.pdf`. A `git add -A` would commit them into the project.
      Add to `.gitignore`. *(Done 2026-10-05 — `Athang_Bachhav_CV.*` ignored.)*
- [ ] `PRODUCTION_DESIGN.md` §2 computes its cost tables off a **1,586-token** prefix;
      `--preflight` now measures 1,704 (Haiku) / 2,291 (Sonnet). Conclusions unchanged
      — Haiku still misses its minimum, Sonnet still caches — but the per-call
      arithmetic is stale.
- [ ] README's token-optimisation section should state the Haiku cache hole plainly
      (see §2 above).

No `TODO`/`FIXME`/`XXX`/`HACK` markers anywhere in `src/`, `evals/`, `tests/` or
`main.py` — the code has no self-declared loose ends.

---

## 7. Published artifacts

Two shareable pages were published from this work. Both are private until shared.

| Page | URL |
|---|---|
| **Backtrace** — POC walkthrough: problem, pipeline, tradeoffs, measured results | https://claude.ai/code/artifact/c2c59b9e-d7d2-44e4-97bd-80470ee7e334 |
| **Where Backtrace Stands** — visual layer status, six diagrams | https://claude.ai/code/artifact/0d7aa421-90ea-4a49-82b8-9e7cd0cbf041 |

To update either from a later session, pass its URL as `url` when republishing —
publishing without it creates a separate artifact instead.
