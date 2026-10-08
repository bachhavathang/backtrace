"""Central configuration for Backtrace: model tiers, thresholds, budgets, flags.

Everything tunable lives here so that (a) the eval harness can sweep thresholds
without editing agent code, and (b) a claim in the ledger can name the exact
config that produced it.

Two ideas drive the layout:

  TIERS      Not every adjudication is equally hard. When retrieval returns one
             obvious winner, a small model is enough. When the top two candidates
             are neck-and-neck — exactly the case that produces false claims — we
             pay for the stronger model. See pick_tier().

  THRESHOLDS The confidence bands are the money policy. They are deliberately
             *data*, not literals buried in agent.py, because evals/run_eval.py
             sweeps them and prints the precision/escalation curve they produce.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Paths ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOGS = DATA / "logs"
CALL_LOG = LOGS / "llm_calls.jsonl"
ENV_FILE = ROOT / ".env"


# --- Credentials ---------------------------------------------------------

def load_dotenv(path: Path | str | None = None) -> int:
    """Read KEY=value lines from a .env file into os.environ. Returns the count set.

    Hand-rolled instead of taking a python-dotenv dependency: this file carries
    one credential in a two-line format, and a new package would put a
    `pip install` between a fresh clone and its first run.

    A variable already present in the environment always wins. That precedence is
    the point, not an accident — it lets CI, a shell export, or a throwaway
    `$env:ANTHROPIC_API_KEY` override the file without anyone having to remember
    to edit or delete it. The file is the fallback, never the authority.

    Never raises. A missing or unreadable .env is the normal case (offline runs,
    CI, the test suite), and failing to find one is not an error.
    """
    path = Path(path) if path is not None else ENV_FILE
    try:
        # utf-8-sig: Windows editors and PowerShell redirection both like to
        # leave a BOM, which would otherwise ride along inside the first key name.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return 0

    loaded = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        # Empty value is treated as absent, so a placeholder `ANTHROPIC_API_KEY=`
        # left unfilled produces the clean "no credential" message rather than a
        # baffling 401 from the API.
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


# Import-time so that every entry point (main, evals, tests) picks the file up
# without having to remember to call it. config is imported before any code
# reads a credential, which makes this the one place it can go.
load_dotenv()


# --- Model tiers ---------------------------------------------------------

@dataclass(frozen=True)
class Tier:
    """One model tier plus the exact request knobs it accepts.

    `request_kwargs` exists because the tiers genuinely differ at the API level:
    Haiku 4.5 rejects `output_config.effort`, and Sonnet 5 runs adaptive thinking
    by default (which we turn off — this is a schema-constrained classification
    over a 3-item shortlist, not multi-step reasoning, and thinking tokens would
    be pure latency and cost here).
    """
    name: str
    model: str
    max_tokens: int
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    request_kwargs: dict
    # Minimum cacheable prefix, which is PER MODEL and is not monotonic across
    # generations — Sonnet 5 caches from 1024 tokens, Haiku 4.5 needs 4096. A
    # single global constant here silently mis-reports one of them; see
    # CACHE_MIN_PREFIX_TOKENS below.
    cache_min_prefix_tokens: int

    def cost_usd(self, input_tokens: int, output_tokens: int,
                 cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
        """Blended cost for one call. Cache reads bill ~0.1x, 5-minute writes ~1.25x."""
        rate_in = self.input_usd_per_mtok / 1_000_000
        rate_out = self.output_usd_per_mtok / 1_000_000
        return (
            input_tokens * rate_in
            + output_tokens * rate_out
            + cache_read_tokens * rate_in * 0.10
            + cache_write_tokens * rate_in * 1.25
        )


# Prices are the standard published rates per million tokens. Sonnet 5 carries a
# lower introductory rate for a limited window; we deliberately book the standard
# rate so the cost report is an over-estimate rather than an under-estimate.
FAST = Tier(
    name="fast",
    model="claude-haiku-4-5",
    max_tokens=300,
    input_usd_per_mtok=1.00,
    output_usd_per_mtok=5.00,
    # Haiku 4.5 does not accept `effort`; sending it is a 400.
    request_kwargs={},
    # 4096, not 1024. This is the highest minimum of any current model and the
    # reason the fast tier never cached: our ~1,600-token prefix clears Sonnet's
    # bar and is nowhere near Haiku's. Measured, not assumed — every fast-tier
    # call in data/logs/llm_calls.jsonl had cache_write=0 and cache_read=0.
    cache_min_prefix_tokens=4096,
)

PRECISE = Tier(
    name="precise",
    model="claude-sonnet-5",
    max_tokens=300,
    input_usd_per_mtok=3.00,
    output_usd_per_mtok=15.00,
    request_kwargs={
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
    },
    cache_min_prefix_tokens=1024,
)

TIERS = {t.name: t for t in (FAST, PRECISE)}


# --- Retrieval + decision thresholds -------------------------------------

@dataclass(frozen=True)
class Thresholds:
    """The money policy, in one object so evals can sweep it.

    no_match_bar  Below this retrieval similarity nothing plausible was found, so
                  we short-circuit to NO_MATCH *without spending an LLM call*.
    low_bar       Below this adjudicated confidence we treat the match as absent
                  rather than uncertain — not worth a human's time.
    high_bar      At or above this we file the claim automatically. Everything
                  between low_bar and high_bar escalates to a human.
    """
    no_match_bar: float = 0.15
    low_bar: float = 0.50
    # 0.90 (provisional). On 600 realistic orders over 24,883 real contract lines, the
    # TUNE split reached zero false claims at 0.80 once the deterministic choice
    # checks (guardrails.choice_flags) run; a fresh, locked confirmation set then
    # found one false claim at 0.85. 0.90 is clean on all three sets (915 orders),
    # but was chosen after seeing that set, so it needs one more fresh run. Without
    # the checks, zero needed 0.95 and escalated ~59%. Was 0.85, set on 7 synthetic lines.
    high_bar: float = 0.90

    # Tier routing. A "clear winner" is a high-scoring top candidate that also
    # beats the runner-up by a comfortable margin. Anything closer than this is
    # a near-duplicate decision and goes to the stronger model.
    fast_tier_min_similarity: float = 0.55
    fast_tier_min_margin: float = 0.15


THRESHOLDS = Thresholds()

# How many products retrieval puts in front of the LLM. Chosen from a measured
# curve (evals/retrieval_recall.py, 272 held-out test orders against 24,717 real
# products, hybrid retrieval): recall@3 84.6%, @5 89.3%, @10 94.9%, @20 96.0%.
# The knee is at 10. A product missing from the shortlist is money lost with no
# signal at all; an extra candidate costs ~40 uncached prompt tokens. Was 3, which
# was chosen against 7 contract lines, where any k >= 3 scores 100% by arithmetic.
# The confidence bars were swept at k=3 and must be re-swept at 10 (Step 3).
RETRIEVAL_K = 10

# Hybrid retrieval: reciprocal-rank fusion of BM25 (weight 1) and embeddings.
# On real catalog text the embedding model alone found the right product in the
# top 3 only 63% of the time; BM25, 85%. Part numbers, brands and sizes are exact
# tokens, and a sentence embedding blurs them. The semantic weight was chosen on
# the TUNE split (0.1 beat 0, 0.25, 0.5, 1.0 at @3) and only then scored on test,
# where it ties pure BM25. It stays as a net for shorthand and synonyms ("cath",
# "pf") that BM25 cannot connect and that degraded-from-source orders under-test.
HYBRID_SEMANTIC_WEIGHT = 0.1
RRF_K = 60            # the standard reciprocal-rank-fusion constant
FUSION_DEPTH = 200    # rows taken from each ranking before fusing


# --- Corpus source -------------------------------------------------------

# Which contract sources build the price index (see src/ingest.py). "synthetic"
# is the three hand-written files and stays the default: every test and the
# current eval are pinned to it. Real sources are opt-in so a bad ingest can be
# rolled back by unsetting one variable.
CORPUS_SOURCE = os.environ.get("BACKTRACE_CORPUS_SOURCE", "synthetic")

# Harvested real data (gitignored). "real" reads BACKTRACE_SNAPSHOT if set, else the
# newest dated folder here. Snapshots are immutable: a scan names the files it read.
RAW_DATA = DATA / "raw"
SNAPSHOT = os.environ.get("BACKTRACE_SNAPSHOT")

# Corpus embeddings, keyed on a hash of the exact texts embedded (gitignored).
EMBED_CACHE = DATA / "cache"


# --- Gateway behaviour ---------------------------------------------------

# Wall-clock ceiling for one adjudication. Short on purpose: this is a small
# classification, and a slow call is more likely wedged than working.
REQUEST_TIMEOUT_S = 30.0

# The SDK retries connection errors, 408/409/429 and 5xx with backoff itself.
MAX_RETRIES = 3

# Parallel adjudications during a backward scan. Orders are independent, so this
# is the difference between an N x 2s scan and an N/8 x 2s scan.
MAX_CONCURRENCY = 8

# Write every request/response to data/logs/llm_calls.jsonl. A recovery claim is
# a financial assertion; you must be able to reconstruct what produced it.
LOG_CALLS = True

# Include the rendered prompt in the call log. Full provenance, larger log.
LOG_PROMPTS = True

# Prompt caching only pays off above a model-specific minimum prefix (1024
# tokens on Sonnet 5 / Haiku 4.5). Below it the API silently declines to cache.
# preflight() in llm.py checks this and warns rather than assuming.
# Kept only as the floor used when a tier is unknown. The real minimum is
# PER MODEL and lives on Tier.cache_min_prefix_tokens.
#
# This constant used to be the only one, applied to both tiers, and it was the
# bug: it said 1024 for Haiku 4.5, whose real minimum is 4096. preflight()
# compared the fast tier's 1,586-token prefix against 1024, reported
# "will engage True", and every fast-tier call then silently billed at full
# price. The check built specifically to catch silent cache failure produced a
# false positive, which is the worst possible failure for that check — the call
# log was the only thing that showed the truth (cache_write=0, forever).
#
# The minimum is not monotonic across model generations, so it can never be a
# single number: a prompt that caches on Sonnet 5 may silently not cache on
# Haiku 4.5 even though Haiku is the newer, cheaper model.
CACHE_MIN_PREFIX_TOKENS = 4096
ENABLE_PROMPT_CACHING = True

# Batch size at which warming the cache starts paying for itself. A warm-up costs
# one full-price call per tier; each later call then saves ~90% on a ~1,600-2,100
# token prefix. Below this many lines the warm-up costs more than it saves, so
# llm.warm_cache() declines to run. Set from the observed ~50% hit rate on a
# 41-case eval fanned out across MAX_CONCURRENCY workers.
CACHE_WARM_MIN_BATCH = 12


# --- Guardrail limits ----------------------------------------------------

# Order descriptions are vendor-controlled free text on a purchase order, so they
# are untrusted input to the prompt. Cap the length an order line can contribute.
MAX_ORDER_TEXT_CHARS = 400

# Contract descriptions and vendor names are third-party text too once real
# catalogs are ingested, and each one is repeated in every prompt whose shortlist
# it joins — RETRIEVAL_K of them per call. The cap bounds both the attack surface
# and the uncached half of the prompt. The verbatim text survives in source_text.
MAX_CONTRACT_TEXT_CHARS = 300
MAX_VENDOR_CHARS = 120

# Share of contract lines the ingest-side injection check may flag before the
# offline eval fails. Each flagged line escalates every order it is shown beside,
# so this is a cost budget, and a sudden jump is the signature of a planted
# payload (docs/WORKING_STATE.md §5.1, G10 and G23).
CONTRACT_FLAG_ALERT_RATE = 0.001


def pick_tier(top_similarity: float, runner_up_similarity: float) -> Tier:
    """Route an adjudication to the cheapest model that can safely make it.

    The rule mirrors the project's core risk: a *close* call between two contract
    lines at different prices is exactly how a false recovery claim gets filed,
    so close calls buy the better model. A runaway leader is cheap to confirm.
    """
    margin = top_similarity - runner_up_similarity
    clear_winner = (
        top_similarity >= THRESHOLDS.fast_tier_min_similarity
        and margin >= THRESHOLDS.fast_tier_min_margin
    )
    return FAST if clear_winner else PRECISE


def has_api_key() -> bool:
    """True when a credential is present. Lets tests and offline runs skip cleanly."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
