"""The LLM gateway: the single place this project is allowed to call a model.

Every model call in Backtrace goes through adjudicate(). Nothing else constructs
a client, names a model, or parses a response. That chokepoint buys five things
that are awkward to retrofit and impossible to enforce when `Anthropic()` is
constructed inline at the call site:

  Provenance   Every call is appended to data/logs/llm_calls.jsonl with its model,
               prompt version, request id, latency, token usage and cost. A
               recovery claim is a financial assertion against a vendor; when it
               is disputed months later you must be able to reconstruct exactly
               what produced it. The ledger records *which* call decided a claim;
               this log records what that call actually was.

  Pinning      The model id lives in config.py, not scattered through agent code.
               A model upgrade is a deliberate change that re-runs the eval suite,
               not something that happens silently on a library bump.

  Reliability  One client with an explicit timeout and retry policy. The SDK
               already retries 429/5xx with backoff; what it cannot decide for us
               is what a *failed* adjudication means. Here it means "abstain and
               escalate" — never "no match", which would silently drop money.

  Cost         Usage is accumulated per tier so a scan can report what it spent
               and whether the prompt cache is actually being hit. Cache misses
               are invisible without this: the API does not error when a prefix
               is too short to cache, it just quietly does not cache it.

  Safety       Raw model output never leaves this module unvalidated. adjudicate()
               returns a guardrails.Verdict, which by construction has an in-range
               index into the shortlist that was actually shown.
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import anthropic

from . import config, guardrails, prompts
from .config import Tier


# --- Client ---------------------------------------------------------------

_client: anthropic.Anthropic | None = None
_client_lock = threading.Lock()


def get_client() -> anthropic.Anthropic:
    """One shared, lazily-built client.

    Lazy so that importing this module never requires a credential (tests and the
    deterministic layers must run key-free). Shared so that connection pooling
    actually happens across a concurrent scan.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = anthropic.Anthropic(
                    timeout=config.REQUEST_TIMEOUT_S,
                    max_retries=config.MAX_RETRIES,
                )
    return _client


def reset_client() -> None:
    """Drop the cached client. Used by tests that swap credentials or transports."""
    global _client
    with _client_lock:
        _client = None


# --- Call records and usage accounting ------------------------------------

@dataclass
class CallRecord:
    """Everything needed to reconstruct and price one model call."""
    order_id: str
    tier: str
    model: str
    prompt_version: str
    ok: bool
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    request_id: str | None = None
    stop_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    flags: list[str] = field(default_factory=list)
    timestamp: str = ""

    @property
    def cache_hit(self) -> bool:
        return self.cache_read_tokens > 0


class UsageAccount:
    """Thread-safe running totals for a scan. Reads are cheap; a scan reports once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[CallRecord] = []

    def add(self, record: CallRecord) -> None:
        with self._lock:
            self.records.append(record)

    def reset(self) -> None:
        with self._lock:
            self.records = []

    def summary(self) -> dict:
        with self._lock:
            records = list(self.records)

        if not records:
            return {"calls": 0, "cost_usd": 0.0}

        latencies = sorted(r.latency_ms for r in records)
        ok = [r for r in records if r.ok]
        by_tier: dict[str, dict] = {}
        for r in records:
            t = by_tier.setdefault(r.tier, {"calls": 0, "cost_usd": 0.0, "model": r.model})
            t["calls"] += 1
            t["cost_usd"] += r.cost_usd

        def pct(p: float) -> float:
            if not latencies:
                return 0.0
            idx = min(int(len(latencies) * p), len(latencies) - 1)
            return round(latencies[idx], 1)

        total_in = sum(r.input_tokens for r in records)
        total_cache_read = sum(r.cache_read_tokens for r in records)
        cacheable = total_in + total_cache_read

        return {
            "calls": len(records),
            "failed": len(records) - len(ok),
            "cost_usd": round(sum(r.cost_usd for r in records), 6),
            "input_tokens": total_in,
            "output_tokens": sum(r.output_tokens for r in records),
            "cache_read_tokens": total_cache_read,
            "cache_write_tokens": sum(r.cache_write_tokens for r in records),
            # What fraction of billable prefix tokens were served from cache. Zero
            # here across a multi-order scan means caching is silently not working.
            "cache_hit_rate": round(total_cache_read / cacheable, 3) if cacheable else 0.0,
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 1),
                "p50": pct(0.50),
                "p95": pct(0.95),
                "max": round(latencies[-1], 1),
            },
            "by_tier": {k: {**v, "cost_usd": round(v["cost_usd"], 6)}
                        for k, v in by_tier.items()},
        }


ACCOUNT = UsageAccount()


# --- Call log -------------------------------------------------------------

_log_lock = threading.Lock()


def _log_call(record: CallRecord, system: str | None, user: str | None,
              response_text: str | None) -> None:
    """Append one line of provenance to the JSONL call log.

    Deliberately append-only and one-JSON-object-per-line: it survives a crash
    mid-scan, and it can be tailed or replayed without loading the whole file.
    """
    if not config.LOG_CALLS:
        return
    entry = asdict(record)
    if config.LOG_PROMPTS:
        entry["prompt"] = {
            "system_sha": _sha(system) if system else None,
            "system_chars": len(system) if system else 0,
            "user": user,
            "response": response_text,
        }
    try:
        config.LOGS.mkdir(parents=True, exist_ok=True)
        with _log_lock:
            with open(config.CALL_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        # Logging must never take down a scan. A dropped log line is recoverable;
        # a crashed run that already filed half its claims is not.
        pass


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# --- The one public call --------------------------------------------------

def adjudicate(order_id: str, order_text: str, candidates: list,
               tier: Tier) -> tuple[guardrails.Verdict, CallRecord]:
    """Ask a model which shortlisted contract line is the same product. Never raises.

    `order_text` is untrusted purchase-order free text; it is sanitised here so
    no caller can forget to. The returned Verdict is always safe to act on: on
    any failure it abstains and carries a flag explaining why, so the caller's
    correct move is the same in every case.
    """
    clean = guardrails.sanitize_order_text(order_text)
    system_blocks = _system_blocks()
    user = prompts.build_user_message(clean.text, candidates)

    started = time.perf_counter()
    record = CallRecord(
        order_id=order_id,
        tier=tier.name,
        model=tier.model,
        prompt_version=prompts.PROMPT_VERSION,
        ok=False,
        latency_ms=0.0,
        flags=list(clean.flags),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    try:
        response = get_client().messages.create(
            model=tier.model,
            max_tokens=tier.max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": user}],
            **_request_kwargs(tier),
        )
    except anthropic.NotFoundError as exc:
        return _fail(record, clean, "model_not_found",
                     f"Model {tier.model!r} unavailable: {exc}",
                     started, system_blocks, user)
    except anthropic.AuthenticationError as exc:
        return _fail(record, clean, "auth", f"Credential rejected: {exc}",
                     started, system_blocks, user)
    except anthropic.RateLimitError as exc:
        # The SDK already backed off and retried MAX_RETRIES times before this.
        return _fail(record, clean, "rate_limit", f"Rate limited after retries: {exc}",
                     started, system_blocks, user)
    except anthropic.APIStatusError as exc:
        return _fail(record, clean, f"http_{exc.status_code}", str(exc),
                     started, system_blocks, user)
    except anthropic.APIConnectionError as exc:
        return _fail(record, clean, "connection", f"Network failure: {exc}",
                     started, system_blocks, user)

    record.latency_ms = (time.perf_counter() - started) * 1000
    record.request_id = getattr(response, "_request_id", None)
    record.stop_reason = getattr(response, "stop_reason", None)
    _record_usage(record, response, tier)

    # A refusal is a successful HTTP response with no usable content. Read
    # stop_reason before touching content, or this indexes into an empty list.
    if record.stop_reason == "refusal":
        verdict = guardrails.error_verdict(
            "Model declined to answer (safety refusal).", clean.flags)
        record.error_type = "refusal"
        record.flags = verdict.flags
        _finish(record, system_blocks, user, None)
        return verdict, record

    text = next((b.text for b in response.content if b.type == "text"), "")

    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # output_config.format should make this unreachable. It is kept because
        # "unreachable" and "never happens in production" are different claims,
        # and the cost of being wrong here is a crashed scan.
        verdict = guardrails.error_verdict(
            "Model output was not valid JSON.", clean.flags)
        record.error_type = "unparseable"
        record.flags = verdict.flags
        _finish(record, system_blocks, user, text)
        return verdict, record

    verdict = guardrails.validate_verdict(raw, candidates, clean.flags)
    record.ok = True
    record.flags = verdict.flags
    _finish(record, system_blocks, user, text)
    return verdict, record


# --- Internals ------------------------------------------------------------

def _system_blocks() -> list[dict]:
    """The system prompt as a content block, with the cache breakpoint on it.

    Render order is tools -> system -> messages, so a breakpoint here caches
    everything stable and leaves the per-order user turn uncached. That is the
    whole shape of the token-cost optimisation: one large shared prefix paid at
    full price once, then at roughly a tenth for the rest of the scan.
    """
    block: dict = {"type": "text", "text": prompts.SYSTEM_PROMPT}
    if config.ENABLE_PROMPT_CACHING:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def _request_kwargs(tier: Tier) -> dict:
    """Merge the tier's knobs with the structured-output contract.

    `format` and `effort` both live inside output_config, so the tier's
    output_config (if it has one) is merged rather than overwritten.
    """
    kwargs = {k: v for k, v in tier.request_kwargs.items() if k != "output_config"}
    output_config = dict(tier.request_kwargs.get("output_config", {}))
    output_config["format"] = {
        "type": "json_schema",
        "schema": prompts.VERDICT_SCHEMA,
    }
    kwargs["output_config"] = output_config
    return kwargs


def _record_usage(record: CallRecord, response, tier: Tier) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    record.input_tokens = getattr(usage, "input_tokens", 0) or 0
    record.output_tokens = getattr(usage, "output_tokens", 0) or 0
    record.cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
    record.cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    record.cost_usd = tier.cost_usd(
        record.input_tokens, record.output_tokens,
        record.cache_read_tokens, record.cache_write_tokens,
    )


def _finish(record: CallRecord, system_blocks: list[dict], user: str,
            response_text: str | None) -> None:
    ACCOUNT.add(record)
    _log_call(record, system_blocks[0]["text"], user, response_text)


def _fail(record: CallRecord, clean: guardrails.SanitizedText, error_type: str,
          message: str, started: float, system_blocks: list[dict],
          user: str) -> tuple[guardrails.Verdict, CallRecord]:
    record.latency_ms = (time.perf_counter() - started) * 1000
    record.error_type = error_type
    record.error_message = message[:500]
    verdict = guardrails.error_verdict(message, clean.flags)
    record.flags = verdict.flags
    _finish(record, system_blocks, user, None)
    return verdict, record


# --- Preflight ------------------------------------------------------------

def preflight(tier: Tier | None = None) -> dict:
    """Measure the cached prefix and report whether caching will actually engage.

    Worth its own function because this is the failure mode you cannot see: when
    the cacheable prefix is under the model's minimum (1024 tokens on Sonnet 5 and
    Haiku 4.5), the API does not reject the cache_control marker. It accepts the
    request, ignores the marker, and bills every call at full price. The only
    signals are this count up front and cache_read_input_tokens afterwards.
    """
    tier = tier or config.PRECISE
    counted = get_client().messages.count_tokens(
        model=tier.model,
        system=_system_blocks(),
        messages=[{"role": "user", "content": "probe"}],
    )
    system_tokens = counted.input_tokens
    eligible = system_tokens >= config.CACHE_MIN_PREFIX_TOKENS
    return {
        "model": tier.model,
        "prompt_version": prompts.PROMPT_VERSION,
        "cached_prefix_tokens": system_tokens,
        "cache_minimum_tokens": config.CACHE_MIN_PREFIX_TOKENS,
        "caching_will_engage": eligible and config.ENABLE_PROMPT_CACHING,
        "note": (
            "Prefix clears the minimum; expect cache_read_input_tokens > 0 from "
            "the second call onward."
            if eligible else
            f"Prefix is {config.CACHE_MIN_PREFIX_TOKENS - system_tokens} tokens "
            "short of the cache minimum. The marker will be silently ignored and "
            "every call billed at full input price."
        ),
    }
