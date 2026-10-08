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
2. Python filters that product's contract rows to those where the **holder is bound
   to the sale** (G5, revised 2026-10-07 — see below), the contract was **in force on
   the effective date** (G13), and the hospital is **eligible** (assumed in the demo,
   labelled — G16).

   **Who must hold the contract — settled 2026-10-07 by the user ("Strict: A and B").**
   A contract is a promise by its holder; money is owed back only by a party bound
   by it. A claim counts when:
   - **A — the seller holds it** (holder == order's billing supplier): the seller
     broke its own promise. Added after the VA data showed most contracts are held
     by sellers/resellers, not makers.
   - **B — the maker holds it** (holder == order's manufacturer): the distributor that
     delivered must honour the maker's price (GPO / chargeback model).
   - **C — anyone else holds it**: **not a claim.** Reported as a *savings
     opportunity* ("buy from the holder next time") — the forward/monitor mode.
     Rejected alternative "holder OR registry manufacturer" would claim case C: a
     false claim against a vendor that made no promise.
   Holder names match exactly (case/space folded) until vendor entity resolution
   exists, so a name variant can turn an A/B into a C — missed money, never a false
   claim, and visible in the savings list rather than dropped.
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

**PR 1 merged** as #1 (`fc0610b`).

**PR 2 status (2026-10-05, branch `step1-pr2-contract-trust`):** G8–G10 and G22.
`ingest.vet()` cleans every contract row's description, vendor and holder with the
same sanitiser as order text, checks the SKU against `guardrails.SKU_PATTERN`, and
sets `needs_verification` on a hit — never drops the row. The prompt fences contract
lines in `<contract_lines>` and JSON-escapes each field; the system prompt now treats
them as data (`reverse-map/v4`; Sonnet prefix 2,447 tok, still caches; Haiku 1,818,
still doesn't). `guardrails.shortlist_flags()` adds `unverified_contract_shown` to the
input flags inside `llm.adjudicate`, so it survives every exit and `decide()` stays
pure. Offline eval reports the contract flag rate and fails above
`CONTRACT_FLAG_ALERT_RATE` (0.1%). 146 tests pass; mutation-checked.

Carried forward from PR 2:
- **Clearance is not built.** A flagged contract escalates every order it is shown
  beside, forever — safe, but noisy. Decision 2's one-time review, bound to a text
  checksum and folded into `corpus_version`, lands when real data makes flags happen.
- The 0.1% false-positive budget is untested against real catalog wording — PR 3.
- The model is still called when a flagged line is in the shortlist; its pick goes to
  the human as a suggestion. Skipping the call is an option if cost ever matters.

**PR 3 split** into 3a (real sources) and 3b (product grouping, persisted embeddings,
human gate re-keyed on identity) to keep each reviewable.

**PR 3a status (2026-10-06, branch `step1-pr3a-real-sources`, stacked on PR 2):**
`BACKTRACE_CORPUS_SOURCE=real` loads VA NAC contract prices enriched from openFDA,
from a dated snapshot in `data/raw/` (gitignored). `python -m src.sources.harvest`
is the only network code; `--resume` finishes an interrupted run.
`python main.py --ingest-report` shows rows read/kept/rejected/joined/flagged/merged.

Neither source has a usable bulk API for this: openFDA is a real API; the VA has none,
but its public search page is a plain GET returning a table (400 rows/page), and a
detail page per item gives the price unit and dates. Contact details on those pages
are never stored.

Measured on snapshot `2026-10-05-wide` (12 single-word terms, 800 detail pages, 900
keyless openFDA queries):

| | |
|---|---|
| Contract lines | **24,883** (27,249 read; 2 rejected for price; 2,364 duplicates merged) |
| Contracts / contractors | 325 / 314 |
| Dates known | 100% — one detail page per contract dates every row on it |
| Unit known | **3%** — needs one detail page per item; 800 of ~25k fetched |
| Per-each price computable | 3% |
| Maker joined from FDA | 13% (3,287 joins; refusals: no exact part 19,813, shared by several makers 1,471, too generic 1,466, descriptions disagree 1,210) |
| Flagged by vetting | **0** — injection patterns: 0 false positives on 24,883 real descriptions (G10 measured) |
| Same SKU on >1 contract row | 878 · identical descriptions on >1 row: 926 (G6 is real) |

Findings the live data forced, each fixed with a regression test:
1. **False registry join** — part "150" joined a Foley catheter to another maker's
   sclerotherapy catheter. Joins now need a ≥5-char part, one maker, and two shared
   product words.
2. **SKU false positives, twice** — an allowlist rejected spaces (3/22 lines), the next
   rejected `, * " & =` (174/24,883 = 0.7%, 7× budget). Replaced by `sku_is_clean()`
   (rejects control chars, fences, injection phrasing, our own field syntax, >64
   chars) and the SKU is now printed as an escaped literal: `reverse-map/v5`.
3. **Phrase search is narrow** — 6 phrases returned 205 of 510,681 lines; single
   words return thousands. Default terms are single words.
4. **Harvest was fragile** — a DNS failure lost all in-memory detail results. Detail
   failures are now counted and skipped, progress saved every 50 pages, `--resume`.

Carried forward from PR 3a:
- **Units are the binding constraint.** At 3% known, almost every claim will route
  to a human under Decision 1. Fix is a long detail harvest (~7 h at 1 req/s; run
  with `--resume --details 25000`, can span nights) or another unit source.
- **FSS contracts are often held by resellers**, not manufacturers. Resolved
  2026-10-07: Decision 1 now accepts holder == seller (A) or holder == maker (B);
  other holders are savings opportunities (§5.2). Implementation lands in PR 3b.
- FDA join rate 13% keyless; an `OPENFDA_API_KEY` lifts the 900-query cap.
- VA "EA" sometimes prices a pack (e.g. one catheter at $1,171). This overstates the
  contract price, which shrinks a claim — the safe direction — but it is noise.

**PR 3b status (2026-10-08, branch `step1-pr3b-products`, stacked on PR 3a):**
- **Shortlist = products.** `retrieve_*` return k distinct products; each
  `CandidateMatch` carries `group` (every contract row for that product). Rows join
  only on a shared normalised part number plus the same registry maker or identical
  description — under-merge costs a human look, over-merge would cost a false claim.
- **Holder rule** (settled 2026-10-07) is in `select_contract_price`: seller holds it
  (A) or maker holds it (B) → claimable; anyone else → `SAVINGS_OPPORTUNITY`.
- **Embeddings persist** to `data/cache/` keyed on a hash of the exact texts + model.
- Flags, the prompt's vendor field (`reverse-map/v6`, lists all vendors of a product)
  and the human gate see the whole group; the gate rebuilds by identity, not SKU.
- **Deadlock found and fixed:** the first semantic query held the embedding lock
  while the model load took it again — every run hung. Pinned by a threaded test.

Measured on the 24,883-line corpus:

| | |
|---|---|
| Products | 24,717 (152 products span more than one contract row) |
| Shortlists repeating a product, row-level → product-level | 1% → **0%** (300 sampled queries) |
| Embedding the corpus, first run → cached | 155 s → **0.1 s** |
| Remaining startup cost | corpus parse 22 s, `sentence_transformers` import 11 s |
| Query latency | ~80 ms |

**G6 was smaller than feared.** "878 SKUs on >1 row" were mostly *different*
products sharing a part number (different maker or description), which the
conservative key correctly keeps apart. The hard retrieval problem in real data is
near-duplicates — same product family, different size or pack — which is exactly
what Step 2's recall@k measurement is for.

**PRs #2, #3, #4 merged 2026-10-08** (top-down; `main` at `cb7ba13`, 206 tests pass).

**PR 4 status (2026-10-08, branch `step1-pr4-real-orders`):** `python -m evals.real_orders`
turns a real snapshot into labelled, degraded order lines (§3C option 1). Written to
`data/derived/<snapshot>/orders-seed<N>.json` (gitignored; reproducible from snapshot +
seed). Each order has a hidden `label`: kind (match_seller A / match_maker B / savings C
/ no_contract), the product key as an equivalence set, split, operators, and
`list_price_synthetic: true`.

- Operators: lowercase, abbreviate (inverse glossary), drop_words (never numbers),
  shuffle, truncate; **held out for the test split only:** typo, rephrase_pack.
- Split **by product** — no product in both halves.
- no_contract products are listed in `held_out_product_keys`; Step 2's eval must
  search the corpus *minus* those.
- Supplier/maker names on B and C cases are fictional distributors/makers.

600 orders, seed 7: 330 A, 60 B, 120 C, 90 no-contract; 265 tune / 335 test.
**Claim-rule audit: 0 false claims** (seeds 7, 11, 23) — every A/B claimable, every C a
savings opportunity, every no-contract unpriceable.

The audit's first run found **1 "false claim"** — and the rule was right: a reseller's
row was labelled C while the product's *maker* held its own contract (a genuine B).
The generator now builds C cases only from products with no maker-held row. Pinned
by a regression test, plus a test that the audit fires on a mislabelled case.

Known limit: `truncate` can strip every identifying word ("the blcak knight is",
from a brand-led description). Realistic for vague orders; Step 2 will count them.

**PR 4 merged** as #5 (`31c4994`).

### Step 2 — retrieval re-baselined (2026-10-08, branch `step2-retrieval-recall`)

`python -m evals.retrieval_recall [--split test|tune|all] [--keyword]` — no API calls.
Searches the corpus minus held-out products; a hit is the right *product* in the top k.

**Test split, 272 orders, 24,717 products:**

| Retriever | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| Embeddings (what the scan used) | 48.9% | 63.2% | 69.9% | 80.9% | 86.0% |
| Jaccard keyword (the "control") | 71.7% | 84.6% | 88.2% | 92.6% | 94.1% |
| BM25 | 72.4% | 85.3% | 89.3% | 94.9% | 96.0% |
| **Hybrid (BM25 + 0.1 × embeddings)** | 72.4% | 84.6% | 89.3% | **94.9%** | 96.0% |

Findings:
1. **Embeddings alone were the wrong retriever for real catalog text.** Part numbers,
   brands and sizes are exact tokens; a sentence embedding blurs them. The 7-line
   corpus could never show this. Caveat: orders are degraded *from* the source text,
   which flatters lexical retrieval; independently typed orders share fewer exact words.
2. **Hybrid weight chosen on the tune split** (0.1 beat 0, 0.25, 0.5, 1.0 at @3: 85.7%)
   and only then scored on test, where it ties pure BM25. Kept for shorthand/synonyms.
   Synthetic corpus: 20/20 @3 for every weighting; @1 rose 95% → 100%.
3. **`RETRIEVAL_K` 3 → 10** — the knee of the curve (@10 94.9%, @20 96.0%). A missed
   product is silent lost money; a candidate costs ~40 uncached tokens.
4. **`no_match_bar` never fires on real data.** Top-1 cosine: median 0.778 when the
   product is in the corpus, 0.763 when it is not — no threshold separates them. The
   README's "largest single saving" assumed it removed over half the lines; corrected.
5. Held-out operators cost ~5 points @10 (87.4% vs 92.4% hybrid) — the size of the
   optimism a tune-only score would have carried.

**Step 3 must now:** re-sweep `high_bar`/`low_bar` at k=10 with hybrid retrieval (bars
were set at k=3); decide whether `no_match_bar` should be removed or re-founded on a
signal that separates (e.g. the model's own NO_MATCH); measure escalation rate on the
real orders. All need an API key.

**Step 2 merged** as #6.

### Step 3 — the full pipeline on real data (2026-10-08, branch `step3-calibration`)

`python -m evals.real_eval` adjudicates the realistic orders (API), appends each verdict
to `data/derived/<snapshot>/verdicts-seed<N>.jsonl`, skips work already on file, and
stops at a hard `--budget`. `--replay` re-scores saved verdicts free (decide() is pure);
`--checks` re-applies the deterministic choice checks to saved verdicts.

**Spend:** $2.82 (600 orders, seed 7) + $1.47 (315 fresh test orders, seed 11) + $0.21
(synthetic eval) = **$4.50**, under the $5 cap. $0.0047 per order.

**What the model gets wrong on real data — one pattern.** Every false claim was a
*sibling*: the order omits the one attribute two catalog variants differ on
("serrated", "with stylet", "AIR", "extended insulation"), and the model picks one at
0.85–0.92, sometimes asserting the attribute ("without stylet matches exactly") when
the order never said it. The prompt's "silence is not agreement" rule did not hold.

**Two deterministic checks** (`guardrails.choice_flags`, escalating flags, run inside
`llm.adjudicate`, so `decide()` stays pure):
- *number check* — every number in the order must be in the chosen product
  ("5 pairs" cannot be the "3 PAIRS" line);
- *sibling check* — a near-identical rival (description Jaccard ≥ 0.5) the order gives
  no reason to reject, or a reason to prefer, escalates. First version scored
  similarity with SKU tokens and missed a textbook pair (0.44); now description-only.

**Results** (false claims / auto-claim recall / escalation):

| Bar | Checks | seed-7 tune | seed-7 test | seed-11 test (fresh, locked) |
|---|---|---|---|---|
| 0.85 (old) | no | — | **3** / 41% / 39% | — |
| 0.95 | no | 0 / 25% / 52% | 0 / 20% / 59% | — |
| 0.80 | yes | 0 / 42% / 38% | 0 / 35% / 45% | **1** / 43% / 35% |
| **0.90 (shipped)** | yes | 0 / 38% / 40% | 0 / 32% / 47% | **0** / 41% / 37% |

**Honest status of 0.90: provisional.** Process: bars chosen on tune → 0.80; locked
confirmation on fresh seed-11 orders → **1 false claim** (failed: a plain Wullstein
forceps absent from the catalog, its serrated sibling claimed at 0.85 — no rival in the
shortlist, so the sibling check had nothing to compare). 0.90 is clean on all three sets
but was chosen *after* seeing seed-11, so it needs one more fresh run to count. Also
disclosed: the two checks were designed after looking at false claims that included
seed-7 *test* orders.

**Calibration** (fresh seed-11): confidence 0.90–1.00 → 100% right (124/124); 0.80–0.90
→ 83%; 0.70–0.80 → 89%. Below 0.9 the model's stated confidence is not reliable enough
to auto-claim on — which is what 0.90 encodes.

**CI gate landed:** `.github/workflows/ci.yml` runs pytest + the offline eval on every
push/PR. `tests/test_policy_gate.py` replays 915 recorded verdicts
(`evals/fixtures/`) and fails the build on any false claim at the shipped bars — plus a
test that it *does* fire at 0.80.

**CI's first runs caught two real bugs**, both now fixed: plain `pytest` (what the README
documents) could not import `src/` — only `python -m pytest` worked (`pytest.ini` now sets
`pythonpath`); and the project only ran on Windows — the synthetic contract files carry a
cp1252 em dash that the platform-default `read_text()` decoded on Windows and crashed
on Linux (`ingest.read_document` now tries UTF-8, then cp1252).

**Not changed:** `no_match_bar` (never fires on real data; harmless; replacing it is a
design question for later). Synthetic eval at k=10/hybrid: 0 false claims at every bar.

**Next:** a third check for the case that beat this one — the chosen product carries a
variant word ("serrated") the order never mentions and no rival is shown — derived from
the corpus's own sibling differences, then **one more fresh, locked run (~$1.50)**.

**Step 3 merged** as #7.

### Rule 3 — "unconfirmed variant word" (2026-10-09, branch `step3b-variant-check`)

Targets the case that beat the first two checks: the order's product is not in the
catalog, so its sibling is the only one shortlisted ("'wullstein' drsg. forecps" →
the SERRATED forceps at 0.85). No rival is shown, so the sibling check has nothing to
compare. Rule 3 escalates a pick whose product carries an **optional qualifier** the
order never mentions.

- **Vocabulary learned from the catalog**, not hand-listed (`corpus.variant_vocabulary`):
  words that appear as a one-word *add-on* between products from one holder (the same
  product exists with and without it), seen in ≥ 3 places, minus grammar/packaging.
  186 words on the real corpus (198 before removing grammar/packaging): serrated, curved, straight, sterile, lock, stylet…
  Linear time (0.5 s on 24,717 products); a first pairwise version timed out.
- **Confirmation** tolerates prefixes ("ster"), one-letter typos ("forecps"), and only
  the shorthand the *production* prompt glossary teaches — not the eval's degradation
  table, which would grade the check against its own answer key.
- Iterations, all measured free by replay: all one-word differences (633 words) →
  auto-claims halved; add-ons only → better; minus grammar/packaging → best, still
  below two rules at 0.90.

| Setup | seed-7 tune | seed-7 test | seed-11 test |
|---|---|---|---|
| 2 rules @ 0.90 (shipped) | 0 / 38% / 40% | 0 / 32% / 47% | 0 / 41% / 37% |
| 3 rules @ 0.80 | 0 / 32% / 48% | 0 / 25% / 53% | 0 / 34% / 43% |

(false claims / auto-claim recall / escalation)

**Ships OFF** (`config.VARIANT_CHECK`, env `BACKTRACE_VARIANT_CHECK=1`). It is
deterministic, so one paid fresh run can be replayed under both setups for free:
`python -m evals.real_eval --replay --checks [--variant-check]`.

> **Pre-registered decision (written before the run):** on the next fresh, locked run,
> ship the setup with **zero false claims and the higher auto-claim recall**. If both
> are zero → keep rule 3 **off** (two rules @ 0.90, simpler). If only rule 3 is zero →
> turn it **on** and re-choose its bar on that run's *tune* half only. If neither →
> no ship; analyse first.

**Also found:** a second instance of the same lock bug — `variant_vocabulary` held the
groups lock and called `product_groups`, which took it again; it only worked when
something else had grouped the corpus first. Lock is now re-entrant, and a test runs
every cached builder cold, in a thread with a timeout.

**The paid run, when credits are available** (after the overnight unit download, so
the price rule sees units):
```
python -m evals.real_orders --seed 23
python -m evals.real_eval --seed 23 --split test --budget 2
python -m evals.real_eval --seed 23 --replay --checks                    # 2 rules
python -m evals.real_eval --seed 23 --replay --checks --variant-check    # 3 rules
```
Expected ~$1.50. Note the unit download changes the snapshot, so seed 23's orders and
prices come from the fuller data; the recorded fixtures stay valid (replay only).

Carried forward from PR 3b:
- `select_contract_price` is still not wired into the scan: it needs order lines that
  name a seller and a maker, which PR 4's degraded real orders will carry.
- Corpus parse (22 s) could be cached like the embeddings; not yet worth it.

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
