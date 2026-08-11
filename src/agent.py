"""Stage 2 — THE CORE. The reverse-map agent.

Flow (a LangGraph StateGraph):

    retrieve ──▶ REVERSE_MAP ──(confident)──▶ recover ──▶ END
                     │
                     ├──(uncertain)──▶ human_gate ──(confirm)──▶ recover ──▶ END
                     │                            └──(deny/defer)───────────▶ END
                     └──(no_match)──────────────────────────────────────────▶ END

THE central risk this file is organised around:
  A FALSE POSITIVE is worse than a miss. If you wrongly map an order to a
  contract, you file a recovery claim against a vendor for money you aren't owed
  — a credibility hit with the customer AND the vendor. So the confidence bar for
  auto-claiming is HIGH, anything ambiguous escalates rather than guessing, and
  every failure path in this file abstains rather than assuming.

Three things live here that are easy to miss:

  Short-circuit  If nothing clears the retrieval floor, NO_MATCH is returned
                 without an LLM call at all. On a real corpus most non-catalog
                 lines genuinely aren't under contract, so this is the single
                 largest cost saving in the pipeline — it is free to be sure
                 about the obvious negatives.

  Tiering        Adjudications are routed to the cheapest model that can safely
                 make them (config.pick_tier). Close calls — the ones that
                 produce false claims — buy the stronger model.

  Deferred gate  In batch mode the human gate does not block. A scan of ten
                 thousand orders cannot stop on order three waiting for someone
                 to type 'y'. Uncertain lines are queued and reviewed after the
                 scan completes; see main.py.
"""
from __future__ import annotations

import threading
from typing import TypedDict

from .config import RETRIEVAL_K, THRESHOLDS, Thresholds, pick_tier
from .corpus import build_corpus, corpus_version, retrieve_semantic
from .llm import adjudicate
from .recovery import record_recovery
from .schema import (CandidateMatch, ContractPrice, MatchDecision,
                     OrderLine, ReverseMapResult)


class State(TypedDict, total=False):
    """Graph state.

    order        the OrderLine under adjudication
    candidates   the retrieved shortlist
    result       the ReverseMapResult being built up
    interactive  whether the human gate may block on stdin (False in batch mode)
    """
    order: OrderLine
    candidates: list[CandidateMatch]
    result: ReverseMapResult
    interactive: bool


# --- Nodes ---------------------------------------------------------------

def node_retrieve(state: State) -> State:
    """Pull candidate contract lines for this order.

    Retrieval is the *recall* half of retrieve-then-judge: cast a wide net
    cheaply. It is also what keeps the prompt small — the corpus could hold ten
    thousand contract lines and the adjudicator still only ever sees three.
    """
    corpus = build_corpus()
    order = state["order"]
    query = order.raw_description + (f" {order.sku_hint}" if order.sku_hint else "")
    state["candidates"] = retrieve_semantic(query, corpus, k=RETRIEVAL_K)
    return state


def node_reverse_map(state: State) -> State:
    """Adjudicate: does this order TRULY match a candidate, and how sure are we?

    Stage A: if nothing clears NO_MATCH_BAR, short-circuit without an LLM call.
    Stage B: route to a model tier based on how close the top two candidates are.
    Stage C: the model picks from the shortlist by index; guardrails re-validate.
    Stage D: map a validated verdict onto the confidence bands.

    Prices are copied from the order and the chosen contract record — the model
    never sees a dollar figure, so it cannot invent one.
    """
    order = state["order"]
    candidates: list[CandidateMatch] = state["candidates"]

    # --- Stage A: nothing plausible, and we can know that for free ---
    top = candidates[0] if candidates else None
    if top is None or top.similarity < THRESHOLDS.no_match_bar:
        state["result"] = ReverseMapResult(
            order_id=order.order_id,
            decision=MatchDecision.NO_MATCH,
            confidence=0.0,
            rationale=(
                "No contract candidate cleared the minimum retrieval similarity "
                f"({THRESHOLDS.no_match_bar}); not sent for adjudication."
            ),
            corpus_version=corpus_version(),
            candidates_considered=[c.contract.sku for c in candidates],
        )
        return state

    # --- Stage B: pick the cheapest model that can safely make this call ---
    runner_up = candidates[1].similarity if len(candidates) > 1 else 0.0
    tier = pick_tier(top.similarity, runner_up)

    # --- Stage C: adjudicate. Never raises; failures come back as abstentions. ---
    verdict, call = adjudicate(
        order_id=order.order_id,
        order_text=order.raw_description,
        candidates=candidates,
        tier=tier,
    )

    chosen = candidates[verdict.chosen_index - 1] if verdict.chosen_index else None

    # --- Stage D: confidence bands ---
    decision = decide(verdict, chosen is not None)

    result = ReverseMapResult(
        order_id=order.order_id,
        decision=decision,
        confidence=verdict.confidence,
        matched_sku=(chosen.contract.sku if chosen else None),
        matched_source=(chosen.contract.source if chosen else None),
        rationale=verdict.reason,
        # Provenance: what decided this, and what it was looking at.
        model=call.model,
        tier=call.tier,
        prompt_version=call.prompt_version,
        corpus_version=corpus_version(),
        candidates_considered=[c.contract.sku for c in candidates],
        guardrail_flags=verdict.flags,
        latency_ms=round(call.latency_ms, 1),
        cost_usd=round(call.cost_usd, 6),
    )

    if decision == MatchDecision.MATCH and chosen is not None:
        _fill_prices(result, order, chosen.contract)

    state["result"] = result
    return state


def decide(verdict, has_match: bool, thresholds: Thresholds | None = None
           ) -> MatchDecision:
    """Map a validated verdict onto MATCH / UNCERTAIN / NO_MATCH. Pure function.

    Pure and threshold-parameterised on purpose: evals/run_eval.py adjudicates
    the whole eval set once, then replays this function across a grid of
    thresholds to plot the precision-versus-escalation curve. The bars stay a
    measured choice rather than two literals nobody can defend, and sweeping
    them costs nothing because it re-decides recorded verdicts instead of
    re-calling the model.

    Order of checks matters. A guardrail violation outranks the model's own
    confidence: a suspected injection or a malformed verdict escalates no matter
    how certain the model claimed to be.
    """
    t = thresholds or THRESHOLDS
    if verdict.must_escalate:
        return MatchDecision.UNCERTAIN
    if verdict.is_ambiguous:
        return MatchDecision.UNCERTAIN
    if not has_match:
        return MatchDecision.NO_MATCH
    if verdict.confidence < t.low_bar:
        return MatchDecision.NO_MATCH
    if verdict.confidence >= t.high_bar:
        return MatchDecision.MATCH
    return MatchDecision.UNCERTAIN


def node_human_gate(state: State) -> State:
    """Escalate an UNCERTAIN match to a person.

    In interactive mode this prompts on stdin. In batch mode it does nothing and
    leaves human_confirmed as None, which routes to END and leaves the line in
    the review queue — a scan must not block on a human, and a human should not
    be asked to adjudicate one line at a time while a scan holds a connection
    pool open.
    """
    if not state.get("interactive", True):
        return state
    review_interactively(state["result"], state["candidates"], state["order"])
    return state


def node_recover(state: State) -> State:
    """Record the recovery (idempotent + audited)."""
    record_recovery(state["result"])
    return state


# --- Human review --------------------------------------------------------

def review_interactively(result: ReverseMapResult, candidates: list[CandidateMatch],
                         order: OrderLine) -> ReverseMapResult:
    """Ask a person to resolve one UNCERTAIN line. Mutates and returns the result.

    Two distinct cases, and conflating them is a real usability bug:
      - The agent picked a candidate but wasn't confident enough -> y/n confirm.
      - The agent abstained on genuine ambiguity -> a yes/no is meaningless, so
        the human must CHOOSE which contract line applies.
    """
    print(f"\n=== REVIEW NEEDED: order {result.order_id} ===")
    print(f"  ordered: {order.raw_description!r}  "
          f"qty {order.quantity:g} @ ${order.list_unit_price:.2f} list")
    print(f"  agent:   {result.rationale}")
    if result.blocked_by_guardrail:
        print(f"  ! guardrail: {', '.join(result.guardrail_flags)}")

    chosen_contract: ContractPrice | None = None

    if result.matched_sku is not None:
        print(f"  proposed: {result.matched_sku} from {result.matched_source} "
              f"(confidence {result.confidence:.2f})")
        if input("Confirm this match? [y/N]: ").strip().lower() == "y":
            chosen_contract = next(
                (c.contract for c in candidates
                 if c.contract.sku == result.matched_sku), None)
    else:
        print("  Agent could not choose. Candidates:")
        for i, c in enumerate(candidates, start=1):
            print(f"    [{i}] {c.contract.sku}  ${c.contract.contracted_unit_price:.2f}"
                  f"  {c.contract.description}")
        answer = input(f"Pick 1-{len(candidates)}, or N for none: ").strip().lower()
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            chosen_contract = candidates[int(answer) - 1].contract

    if chosen_contract is not None:
        result.human_confirmed = True
        result.matched_sku = chosen_contract.sku
        result.matched_source = chosen_contract.source
        _fill_prices(result, order, chosen_contract)
    else:
        result.human_confirmed = False

    return result


def candidates_for(result: ReverseMapResult) -> list[CandidateMatch]:
    """Rebuild the shortlist a result was decided against, from its provenance.

    Lets the deferred review phase in main.py show the same candidates the agent
    saw, without carrying graph state around between the scan and the review.
    """
    by_sku = {c.sku: c for c in build_corpus()}
    return [
        CandidateMatch(contract=by_sku[sku], similarity=0.0)
        for sku in result.candidates_considered if sku in by_sku
    ]


def _fill_prices(result: ReverseMapResult, order: OrderLine,
                 contract: ContractPrice) -> None:
    """Copy prices from the order and the contract record. Never from the model."""
    result.list_unit_price = order.list_unit_price
    result.contracted_unit_price = contract.contracted_unit_price
    result.quantity = order.quantity


# --- Routing -------------------------------------------------------------

def route_after_map(state: State) -> str:
    decision = state["result"].decision
    if decision == MatchDecision.MATCH:
        return "recover"
    if decision == MatchDecision.UNCERTAIN:
        return "human_gate"
    if decision == MatchDecision.NO_MATCH:
        return "end"
    raise ValueError(f"Invalid decision: {decision}")


def route_after_gate(state: State) -> str:
    return "recover" if state["result"].human_confirmed else "end"


# --- Graph ---------------------------------------------------------------

_graph = None
_graph_lock = threading.Lock()


def build_graph():
    """Compile the StateGraph once and reuse it.

    This used to be rebuilt per order line: a scan of N orders compiled N
    identical graphs before doing any work. The compiled graph is stateless
    across invocations, so one instance serves the whole scan, including
    concurrent invocations.
    """
    global _graph
    if _graph is not None:
        return _graph

    with _graph_lock:
        if _graph is not None:
            return _graph
        from langgraph.graph import END, StateGraph

        g = StateGraph(State)
        g.add_node("retrieve", node_retrieve)
        g.add_node("reverse_map", node_reverse_map)
        g.add_node("human_gate", node_human_gate)
        g.add_node("recover", node_recover)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "reverse_map")
        g.add_conditional_edges("reverse_map", route_after_map, {
            "recover": "recover",
            "human_gate": "human_gate",
            "end": END,
        })
        g.add_conditional_edges("human_gate", route_after_gate, {
            "recover": "recover",
            "end": END,
        })

        _graph = g.compile()
    return _graph


def run_one(order: OrderLine, interactive: bool = True) -> ReverseMapResult:
    """Reverse-map a single order line.

    interactive=True  the human gate blocks on stdin (single-order / forward mode)
    interactive=False uncertain lines are left for a later review pass (batch scan)
    """
    final = build_graph().invoke({"order": order, "interactive": interactive})
    return final["result"]
