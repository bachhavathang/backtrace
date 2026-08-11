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
    high_bar: float = 0.85

    # Tier routing. A "clear winner" is a high-scoring top candidate that also
    # beats the runner-up by a comfortable margin. Anything closer than this is
    # a near-duplicate decision and goes to the stronger model.
    fast_tier_min_similarity: float = 0.55
    fast_tier_min_margin: float = 0.15


THRESHOLDS = Thresholds()

# How many contract lines retrieval puts in front of the LLM. This is the single
# biggest lever on prompt size: the corpus could be 10k lines; the prompt sees 3.
RETRIEVAL_K = 3


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
CACHE_MIN_PREFIX_TOKENS = 1024
ENABLE_PROMPT_CACHING = True


# --- Guardrail limits ----------------------------------------------------

# Order descriptions are vendor-controlled free text on a purchase order, so they
# are untrusted input to the prompt. Cap the length an order line can contribute.
MAX_ORDER_TEXT_CHARS = 400


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
