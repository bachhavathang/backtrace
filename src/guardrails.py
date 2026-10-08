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

MAX_SKU_CHARS = 64
_FIELD_SYNTAX = re.compile(r"\s\|\s|\b(sku|vendor|description)\s*=", re.IGNORECASE)


def sku_is_clean(sku: str) -> bool:
    """True when a SKU is safe to print into the prompt and echo back.

    Not a character allowlist. Two allowlists in a row failed against real data:
    one rejected spaces ("SILQ 21400101003", 3 of 22 lines), the next rejected
    commas, asterisks and quotes ("MC*PB2411Y" — 174 of 24,883 lines, 0.7%, seven
    times the false-positive budget). Real catalog numbers use that punctuation.
    The prompt now prints the SKU as an escaped string literal, the way it prints
    descriptions, so punctuation cannot break the line. What remains dangerous is
    what sanitising would have to change — control characters, line breaks, our
    fence tags, injection phrasing — plus excessive length.
    """
    if not sku or len(sku) > MAX_SKU_CHARS or sku != sku.strip():
        return False
    if _FIELD_SYNTAX.search(sku):
        # Escaping stops it ending the field, but a part number has no reason to
        # imitate the candidate line's own " | vendor=" layout.
        return False
    clean = _sanitize(sku, MAX_SKU_CHARS, None)
    return not clean.suspicious and clean.text == sku


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
    # Every row of a grouped product counts: their vendor names reach the prompt too.
    rows = [r for c in candidates for r in (getattr(c, "rows", None) or [c.contract])]
    if any(getattr(r, "needs_verification", False) for r in rows):
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


# --- 5. Choice checks: deterministic second opinions on a confident pick ------
#
# Found by running the model on 600 realistic orders against 24,883 real contract
# lines (evals/real_eval.py). Every false claim at the old bar was a SIBLING
# mix-up: the order left out the one attribute on which two catalog variants
# differ, and the model picked one anyway at 0.85-0.92 —
#     "dressing symmetry 10 in"           -> the plain forceps, not the serrated
#     "suture boots ... 5 pairs ..."      -> the 3-pair variant
# The prompt already says "silence is not agreement"; the model still did it.
# These checks do not judge meaning. They test two facts about the text that a
# claim against a vendor must survive, and either one escalates.

FLAG_NUMBER_CONFLICT = "order_number_absent_from_match"
FLAG_SIBLING = "rival_candidate_not_excluded"
ESCALATING_FLAGS = ESCALATING_FLAGS | {FLAG_NUMBER_CONFLICT, FLAG_SIBLING}

# Two candidates closer than this (word-set Jaccard) are treated as variants of
# one product line, whose differences the order must address.
SIBLING_SIMILARITY = 0.5
_STOPWORDS = frozenset({"the", "and", "with", "for", "of", "per", "in", "x", "w", "a", "to"})


def _numbers(text: str) -> set[str]:
    """Numbers as written, minus trailing zero decimals ("10.0" == "10")."""
    out = set()
    for n in re.findall(r"\d+(?:\.\d+)?", text or ""):
        out.add(n.rstrip("0").rstrip(".") if "." in n else n.lstrip("0") or "0")
    return out


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def _candidate_text(candidate) -> str:
    return f"{candidate.contract.description} {candidate.contract.sku}"


def number_conflict(order_text: str, chosen) -> bool:
    """A number the order states that the chosen product does not carry.

    Sizes, gauges, lengths and pack counts are how variants differ, and a number
    written in an order is deliberate. "5 pairs" cannot be the "3 PAIRS" line.
    Errs towards escalating: "1.00" vs "1" in different notations also fires.
    """
    return not _numbers(order_text) <= _numbers(_candidate_text(chosen))


def sibling_not_excluded(order_text: str, chosen, others: list) -> bool:
    """A near-identical rival the order gives no reason to reject.

    For each rival close enough to be a variant of the chosen product, look at
    the words that tell them apart. If the order mentions none of them, it cannot
    distinguish the two; if it mentions only the RIVAL's, it points the other way.
    Either way the pick is not supported by the order text.
    """
    order = _content_words(order_text)
    mine = _content_words(_candidate_text(chosen))
    mine_desc = _content_words(chosen.contract.description)
    for rival in others:
        theirs = _content_words(_candidate_text(rival))
        # Similarity on DESCRIPTIONS only. With part numbers included, "Symmetry
        # Forceps; Dressing; 10 in" and its "...; Serrated; 10 in" twin scored 0.44
        # — the SKU tokens alone pushed a textbook sibling under the bar, and the
        # one false claim that survived this check on the real eval was exactly it.
        # Part numbers still count as EVIDENCE below: an order citing one decides it.
        theirs_desc = _content_words(rival.contract.description)
        union = mine_desc | theirs_desc
        if not union or len(mine_desc & theirs_desc) / len(union) < SIBLING_SIMILARITY:
            continue
        if number_conflict(order_text, rival):
            continue        # the order's own numbers already rule this rival out
        for_mine, for_theirs = order & (mine - theirs), order & (theirs - mine)
        if not for_mine or len(for_theirs) > len(for_mine):
            return True
    return False


def choice_flags(order_text: str, candidates: list, chosen_index: int,
                 variant_words: frozenset = frozenset()) -> list[str]:
    """Flags owed to the model's pick. Empty for an abstention."""
    if not chosen_index or chosen_index > len(candidates):
        return []
    chosen = candidates[chosen_index - 1]
    others = [c for i, c in enumerate(candidates, 1) if i != chosen_index]
    flags = []
    if number_conflict(order_text, chosen):
        flags.append(FLAG_NUMBER_CONFLICT)
    if sibling_not_excluded(order_text, chosen, others):
        flags.append(FLAG_SIBLING)
    if unconfirmed_variant(order_text, chosen, variant_words):
        flags.append(FLAG_UNCONFIRMED_VARIANT)
    return flags


# --- 5b. Unconfirmed variant word ------------------------------------------------
#
# The case that beat the sibling check on a fresh, locked run: "'wullstein' drsg.
# forecps" — the plain forceps is not in the catalog, so the only Wullstein left is
# the SERRATED one, no rival is shortlisted, and the model claimed it at 0.85. The
# pick carries a variant word the order never asked for. corpus.variant_vocabulary
# learns which words are variant words from the catalog's own one-word siblings.

FLAG_UNCONFIRMED_VARIANT = "match_has_variant_word_order_lacks"
ESCALATING_FLAGS = ESCALATING_FLAGS | {FLAG_UNCONFIRMED_VARIANT}

# Shorthand the PRODUCTION system prompt already teaches the model (prompts.py
# glossary) — deliberately not the eval's degradation table, which would grade
# this check against its own answer key.
_GLOSSARY = {
    "pf": ("powder", "free"), "lg": ("large",), "md": ("medium",), "med": ("medium",),
    "sm": ("small",), "xl": ("extra", "large"), "cath": ("catheter",),
    "syr": ("syringe",), "glv": ("gloves", "glove"), "drp": ("drape",),
}


def _one_edit_apart(a: str, b: str) -> bool:
    """Substitution, insertion, deletion or adjacent swap — a typo, not a new word."""
    if a == b or abs(len(a) - len(b)) > 1 or min(len(a), len(b)) < 4:
        return a == b
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long_ = sorted((a, b), key=len)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _confirmed(word: str, order_words: set[str]) -> bool:
    for t in order_words:
        if t == word or _one_edit_apart(t, word):
            return True
        if len(t) >= 3 and (word.startswith(t) or t.startswith(word)):
            return True
        if word in _GLOSSARY.get(t, ()):
            return True
    return False


def unconfirmed_variant(order_text: str, chosen, variant_words: frozenset) -> bool:
    """The chosen product carries a variant word the order never mentions."""
    if not variant_words:
        return False
    order_words = set(re.findall(r"[a-z]+", unicodedata.normalize("NFKC", order_text or "").lower()))
    mine = {w for w in re.findall(r"[a-z]+", chosen.contract.description.lower()) if len(w) >= 3}
    return any(not _confirmed(w, order_words) for w in mine & variant_words)
