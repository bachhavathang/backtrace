"""Guardrails: what we do to untrusted input, and what we refuse to trust on output.

The threat model is specific and worth stating, because it shapes every choice
here. A purchase-order description is free text that reaches us from outside the
hospital's control. The output of adjudicating it becomes a **financial claim
filed against a vendor**. So an attacker who can influence order text has a
motive: induce a confident match to the wrong contract line and cause a bogus
claim, or induce a match to the cheapest line and inflate the recovery.

Defence is layered, weakest-to-strongest:

  1. Sanitise      Cap length, strip control characters, neutralise the delimiter
                   used to fence untrusted text. Cheap, catches the sloppy case.
  2. Delimit       The order text is fenced in <order_text> tags and the system
                   prompt says everything inside is data. Standard, and bypassable
                   on its own — which is why it is not the last line.
  3. Constrain     The model answers with an *index into a shortlist it did not
                   choose*, and the price is never in the prompt. This is the
                   layer that actually bounds the damage: the very best possible
                   injection can only move the answer between three real contract
                   lines that retrieval already selected. It cannot invent a SKU,
                   cannot invent a price, and cannot reach the ledger directly.
  4. Verify        Every field is re-checked in code below. The index must be in
                   range; the echoed SKU must match the one at that index; the
                   confidence must be a real number in [0, 1].
  5. Escalate      Any violation, and any injection attempt detected in step 1,
                   downgrades the outcome to "needs a human". Nothing that
                   tripped a guardrail is ever auto-claimed.

Layer 3 is the load-bearing one. Layers 1 and 2 raise the cost of an attack;
layer 3 caps the payoff. Layers 4 and 5 make failure safe rather than silent.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

from .config import MAX_CONTRACT_TEXT_CHARS, MAX_ORDER_TEXT_CHARS

# Flag constants. These travel with the result into the ledger, so an auditor can
# see not just that a line was escalated but which guardrail escalated it.
FLAG_INJECTION = "injection_suspected"
FLAG_TRUNCATED = "order_text_truncated"
FLAG_INDEX_OUT_OF_RANGE = "chosen_index_out_of_range"
FLAG_SKU_MISMATCH = "index_sku_disagreement"
FLAG_BAD_CONFIDENCE = "confidence_out_of_range"
FLAG_MALFORMED = "malformed_verdict"
FLAG_LLM_ERROR = "llm_error"
FLAG_UNVERIFIED_CONTRACT = "unverified_contract_shown"

# Flags that must never be auto-claimed, however confident the model was.
ESCALATING_FLAGS = frozenset({
    FLAG_INJECTION,
    FLAG_INDEX_OUT_OF_RANGE,
    FLAG_SKU_MISMATCH,
    FLAG_BAD_CONFIDENCE,
    FLAG_MALFORMED,
    FLAG_LLM_ERROR,
    FLAG_UNVERIFIED_CONTRACT,
})


# --- 1. Input sanitisation ------------------------------------------------

# Phrases whose only purpose in a *product description* is to address the reader.
# A purchase order line describes a physical good; it has no legitimate reason to
# say "ignore previous instructions". Matching here is intentionally about intent
# rather than exhaustiveness — this is a tripwire that forces human review, not a
# filter we rely on to be complete.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|preceding)\b",
        r"\bdisregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|preceding|rules|instructions)\b",
        r"\bforget\s+(everything|all|your|the)\b",
        r"\b(new|updated|revised)\s+(instructions?|rules?|system\s+prompt)\b",
        r"\byou\s+(must|should|will|have\s+to)\s+(now\s+)?(select|choose|pick|return|answer|output|set)\b",
        r"\b(always|never)\s+(select|choose|pick|return|output)\b",
        r"\bset\s+(confidence|is_ambiguous|chosen_index|chosen_sku)\b",
        r"\bconfidence\s*[:=]\s*[01](\.\d+)?\b",
        r"\bchosen_(index|sku)\s*[:=]",
        r"</?\s*(system|assistant|user|instructions?|order_text)\s*>",
        r"\bsystem\s+prompt\b",
        r"\bas\s+an?\s+ai\b",
        r"\boverride\b.{0,20}\b(instruction|rule|policy|prompt)\b",
        # A role header does not need angle brackets to impersonate one. Bracketed
        # pseudo-directives read as privileged text to a model and as ordinary
        # punctuation to a tag-matching regex.
        r"\[\s*(system|admin|developer|assistant|operator|instruction)\b",
        r"\bnote\s+to\s+(the\s+)?(reviewer|reader|agent|assistant|system|auditor|approver)\b",
        # Imperatives naming this pipeline's own output vocabulary. A description of
        # a physical good has no reason to instruct anyone to return a decision.
        r"\b(return|output|respond\s+with|reply\s+with|answer|mark\s+(it\s+)?as)\s+"
        r"(with\s+)?(no_match|match|uncertain|confidence|chosen_index|chosen_sku)\b",
        r"\bconfidence\s+(of\s+|to\s+)?[01](\.\d+)?\b",
        # Claims that a control is switched off. Nothing in a purchase order line
        # is entitled to make an assertion about the state of our own guardrails.
        r"\b(verification|validation|guardrail|review|approval|checking)\s+"
        r"(is\s+|are\s+|has\s+been\s+)?(disabled|off|bypassed|skipped|waived|not\s+required)\b",
    )
]


@dataclass
class SanitizedText:
    text: str
    flags: list[str] = field(default_factory=list)

    @property
    def suspicious(self) -> bool:
        return FLAG_INJECTION in self.flags


# Our own fences. Untrusted text must never be able to open or close one, or it
# could appear to speak from outside the block that marks it as data.
_FENCE_TAG = re.compile(r"</?\s*(order_text|contract_lines)\s*>", re.IGNORECASE)

# What a SKU may look like. It is printed into the prompt and echoed back by the
# model, so a "SKU" carrying prompt syntax (| = " < >) is a payload, not a part
# number. Single spaces between tokens are allowed: real VA catalog numbers look
# like "SILQ 21400101003", and forbidding them flagged 3 of 22 real lines.
SKU_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/#()+-]| (?=[A-Za-z0-9])){0,63}")


def _sanitize(raw: str, max_chars: int, truncated_flag: str | None) -> SanitizedText:
    flags: list[str] = []
    text = raw or ""

    # Normalise unicode first, so lookalike characters cannot smuggle a pattern
    # past the regexes below.
    text = unicodedata.normalize("NFKC", text)

    # Strip control characters (keep ordinary whitespace). These are used to break
    # up keywords and to inject fake role markers.
    text = "".join(
        ch for ch in text
        if ch in "\t\n " or not unicodedata.category(ch).startswith("C")
    )

    # Collapse whitespace: newlines in a product description are noise, and they
    # are the usual way a payload tries to look like a separate prompt section.
    text = re.sub(r"\s+", " ", text).strip()

    # Neutralise our own fences so the payload cannot close a block early and
    # appear to be speaking from outside it.
    if _FENCE_TAG.search(text):
        flags.append(FLAG_INJECTION)
        text = _FENCE_TAG.sub("[tag removed]", text)

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            if FLAG_INJECTION not in flags:
                flags.append(FLAG_INJECTION)
            break

    if len(text) > max_chars:
        text = text[:max_chars]
        if truncated_flag:
            flags.append(truncated_flag)

    return SanitizedText(text=text, flags=flags)


def sanitize_order_text(raw: str) -> SanitizedText:
    """Normalise untrusted order text and flag anything that reads as an instruction.

    Returns the cleaned text plus flags. Detection never *rejects* the line — a
    real order that happens to trip a pattern still gets matched, it just cannot
    be auto-claimed. Silently dropping a line would lose recoverable money; the
    safe failure is a human looking at it.
    """
    return _sanitize(raw, MAX_ORDER_TEXT_CHARS, FLAG_TRUNCATED)


def sanitize_contract_text(raw: str, max_chars: int = MAX_CONTRACT_TEXT_CHARS
                           ) -> SanitizedText:
    """The same treatment for text that arrives in a *contract* — a description or
    a vendor name.

    Contract text used to be trusted because it was hand-written. Once vendor
    catalogs are ingested it is third-party text like any order line, and it is
    worse placed: it reaches the prompt for every order whose shortlist it joins,
    not just one. A hit here marks the row needs_verification at ingest; see
    shortlist_flags() for what that does at decision time.

    Truncation is not flagged: a long catalog description is normal, and the
    verbatim text is kept in ContractPrice.source_text for audit.
    """
    return _sanitize(raw, max_chars, None)


def shortlist_flags(candidates: list) -> list[str]:
    """Flags owed to the shortlist itself, independent of what the model answers.

    An unverified contract line that was shown to the model blocks auto-claim
    whether or not it was chosen: injected text in candidate 2 can steer the
    choice of candidate 1. (docs/WORKING_STATE.md §5.2, Decision 2.) A line that
    never reached the prompt — below the retrieval floor — owes nothing, so this
    is only called for shortlists that are actually sent.
    """
    if any(getattr(c.contract, "needs_verification", False) for c in candidates):
        return [FLAG_UNVERIFIED_CONTRACT]
    return []


# --- 4. Output verification -----------------------------------------------

@dataclass
class Verdict:
    """A validated adjudication. Constructing one means every check below passed."""
    chosen_index: int          # 0 = abstained
    chosen_sku: str | None     # None when abstained
    confidence: float
    is_ambiguous: bool
    reason: str
    flags: list[str] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.chosen_index == 0 or self.chosen_sku is None

    @property
    def must_escalate(self) -> bool:
        """True when a guardrail fired, regardless of how confident the model was."""
        return any(f in ESCALATING_FLAGS for f in self.flags)


def validate_verdict(raw: dict, candidates: list,
                     input_flags: list[str] | None = None) -> Verdict:
    """Re-derive the verdict from raw model output, trusting none of it.

    `raw` is whatever came back from the model. `candidates` is the shortlist the
    model was shown, in the same order it was numbered. The authority for *which*
    contract line was chosen is the index; `chosen_sku` is only ever used as a
    cross-check, so a model that echoes a plausible-looking SKU it was never shown
    cannot smuggle it through.

    Never raises. A malformed verdict becomes an abstention carrying the flag that
    explains it, because the caller's safe move is always "send it to a human".
    """
    flags = list(input_flags or [])

    def bail(flag: str, reason: str) -> Verdict:
        if flag not in flags:
            flags.append(flag)
        return Verdict(0, None, 0.0, False, reason, flags)

    if not isinstance(raw, dict):
        return bail(FLAG_MALFORMED, "Model response was not an object.")

    # -- index: the authoritative field --
    index = raw.get("chosen_index")
    if isinstance(index, bool) or not isinstance(index, (int, float)):
        return bail(FLAG_MALFORMED, "chosen_index missing or not a number.")
    if isinstance(index, float):
        if not index.is_integer():
            return bail(FLAG_MALFORMED, "chosen_index was not a whole number.")
        index = int(index)
    if not (0 <= index <= len(candidates)):
        return bail(
            FLAG_INDEX_OUT_OF_RANGE,
            f"Model returned candidate {index}, outside the shortlist of {len(candidates)}.",
        )

    # -- confidence --
    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return bail(FLAG_BAD_CONFIDENCE, "confidence missing or not a number.")
    confidence = float(confidence)
    if math.isnan(confidence) or not (0.0 <= confidence <= 1.0):
        return bail(FLAG_BAD_CONFIDENCE, f"confidence {confidence} outside [0, 1].")

    is_ambiguous = bool(raw.get("is_ambiguous", False))
    reason = str(raw.get("reason", "")).strip()[:500] or "(no reason given)"

    # -- abstention --
    if index == 0:
        return Verdict(0, None, confidence, is_ambiguous, reason, flags)

    # -- cross-check the echoed SKU against the one we actually showed --
    expected_sku = candidates[index - 1].contract.sku
    echoed = str(raw.get("chosen_sku", "")).strip()
    if echoed.upper() != "NONE" and echoed != expected_sku:
        if echoed not in {c.contract.sku for c in candidates}:
            note = f"Model echoed SKU {echoed!r}, which was not on the shortlist."
        else:
            note = f"Model chose candidate {index} ({expected_sku}) but echoed {echoed!r}."
        return bail(FLAG_SKU_MISMATCH, note)

    return Verdict(index, expected_sku, confidence, is_ambiguous, reason, flags)


def error_verdict(message: str, input_flags: list[str] | None = None) -> Verdict:
    """The verdict used when the call itself failed. Always abstains, always escalates."""
    flags = list(input_flags or [])
    if FLAG_LLM_ERROR not in flags:
        flags.append(FLAG_LLM_ERROR)
    return Verdict(0, None, 0.0, False, message, flags)
