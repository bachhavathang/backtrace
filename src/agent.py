"""Stage 2 — THE CORE. The reverse-map agent.

This is the file that mirrors Resolvd's "procurement-price specialist agent" and
the one that closes your gap. Everything else feeds it. Spend your best hours here.

Flow (a LangGraph StateGraph):

    retrieve ──▶ REVERSE_MAP ──(confident)──▶ recover ──▶ END
                     │
                     ├──(uncertain)──▶ human_gate ──(confirm)──▶ recover ──▶ END
                     │                            └──(deny)─────────────────▶ END
                     └──(no_match)──────────────────────────────────────────▶ END

What's built: retrieve + recover + human_gate nodes (they reuse the modules that
already work). What's YOURS (# TODO): node_reverse_map (the adjudication +
confidence policy), route_after_map (branching), build_graph (wiring). Write them
yourself before asking the AI to critique — you must defend every edge live.

THE central risk to reason about (say this in the interview):
  A FALSE POSITIVE is worse than a miss. If you wrongly map an order to a
  contract, you file a recovery claim against a vendor for money you aren't owed
  — that's a credibility hit with the customer AND the vendor. So the confidence
  bar for auto-claiming must be HIGH, and anything ambiguous (e.g. two glove
  contracts at different prices) must escalate, never guess.
"""
from __future__ import annotations

import json
from pathlib import Path
from tkinter import END
from typing import TypedDict
from unittest import result

from .schema import (CandidateMatch, ContractPrice, MatchDecision,
                     OrderLine, ReverseMapResult)
from .corpus import build_corpus, retrieve_keyword, retrieve_semantic
from .recovery import record_recovery

SYSTEMS = Path(__file__).resolve().parent.parent / "data" / "mock_systems"


class State(TypedDict, total=False):
    order: OrderLine
    candidates: list[CandidateMatch]
    result: ReverseMapResult


# --- Nodes ---------------------------------------------------------------

def node_retrieve(state: State) -> State:
    """Pull candidate contract lines for this order. Built for you.

    Uses keyword retrieval so it runs key-free. Swap to your retrieve_semantic
    once you've built it (and talk about the difference).
    """
    corpus = build_corpus()
    order = state["order"]
    query = order.raw_description + (f" {order.sku_hint}" if order.sku_hint else "")
    state["candidates"] = retrieve_semantic(query, corpus, k=3)
    return state

def _llm_adjudicate(order, candidates) -> dict:
    """Show the LLM the shortlist; let it choose the best match OR flag ambiguity.

    Retrieval narrows (recall); the LLM adjudicates the shortlist (precision).
    The LLM picks among REAL candidates by SKU — it can't invent one, and it
    never sees or sets a price.

    Returns: {"chosen_sku": str|None, "confidence": float, "is_ambiguous": bool,
              "reason": str}
    """
    import json
    from anthropic import Anthropic

    # Build a numbered list of the candidates for the prompt.
    lines = []
    for c in candidates:
        lines.append(f'- sku "{c.contract.sku}": "{c.contract.description}"')
    candidate_block = "\n".join(lines)

    prompt = f"""You match a hospital purchase order line to the correct contract item.
Choose the ONE contract item that is the same physical product (type, size, form,
packaging). Note domain shorthand: "pf" means powder-free, "lg" means large.

If two or more candidates fit equally well and you cannot distinguish them from the
order text, set is_ambiguous to true and chosen_sku to null — do NOT guess.

ORDER: "{order.raw_description}"

CANDIDATES:
{candidate_block}

Return ONLY valid JSON, no markdown:
{{"chosen_sku": "<sku or null>", "confidence": 0.0-1.0, "is_ambiguous": true/false, "reason": "<one short sentence>"}}"""

    client = Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)

    sku = data.get("chosen_sku")
    if sku in (None, "null", ""):
        sku = None
    return {
        "chosen_sku": sku,
        "confidence": float(data.get("confidence", 0.0)),
        "is_ambiguous": bool(data.get("is_ambiguous", False)),
        "reason": str(data.get("reason", "")),
    }

def _llm_confirm_match(order, contract) -> tuple[bool, float, str]:
    """ Ask the LLM: is this orfder the SAME physical item as this contract line?

    Returns: (is_match, confidence 0..1, one line reason). The LLM judges sameness only - it never sees or sets a price.
    """
    import json
    from anthropic import Anthropic
    
    prompt = f""" You compare hospital purchase order line to a contract line and decide if they are 
    the SAME physical product. Consider product type, size, form, and packaging. ignore vendor name differences.

    ORDER: "{order.raw_description}"
    CONTRACT: "{contract.description}" (sku {contract.sku})

    Return ONLY valid Json, no markdown:
    {{"is_match": boolean, "confidence": 0.0-1.0, "reason": "<one short sentence>"}}"""

    client = Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens = 200,
        messages=[{"role": "user", "content": prompt}],
    )

    text = msg.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    return bool(data["is_match"]), float(data["confidence"]), str(data["reason"])



def node_reverse_map(state: State) -> State:
    """Adjudicate: does this order TRULY match a candidate, and how sure are we?

    # TODO(you): THIS is your core deliverable. Given state["order"] and
    # state["candidates"], produce a ReverseMapResult with a decision + confidence.
    #
    # A defensible approach (build, then refine):
    #   1. If the top candidate's retrieval similarity is very low -> NO_MATCH.
    #   2. If the top TWO candidates are both plausible but point to DIFFERENT
    #      prices (the ambiguous-gloves case, PO-5004) -> UNCERTAIN (escalate).
    #      Guessing here = risking a false recovery claim.
    #   3. Otherwise call the LLM to confirm the order truly == the top candidate
    #      (semantic equality, not just token overlap), and to return a calibrated
    #      confidence + a rationale a human auditor could read.
    #   4. confidence >= HIGH_BAR -> MATCH ; mid -> UNCERTAIN ; low -> NO_MATCH.
    #
    # Set state["result"] = ReverseMapResult(...). Fill list/contracted/quantity
    # from the order + chosen candidate so recovery math works downstream.
    #
    # Defend: where's HIGH_BAR and why? Why escalate ambiguity instead of taking
    # the cheaper/pricier match? How is the LLM constrained to not invent prices?
    """
    order = state["order"]
    candidates = state["candidates"]
    
    #Stage A nothing fits well enough, NO MATCH
    NO_MATCH_BAR = 0.15
    top = candidates[0] if candidates else None
    if top is None or top.similarity < NO_MATCH_BAR:
        state["result"] = ReverseMapResult(
            order_id=order.order_id,
            decision=MatchDecision.NO_MATCH,
            confidence=0.0,
            rationale="No contract candidate cleared the minimum cimilarity bar. ",
        )
        return state

    # Stages B+C collapsed: let the LLM adjudicate the shortlist.
    HIGH_BAR = 0.85
    LOW_BAR = 0.50
    verdict = _llm_adjudicate(order, candidates)

    # Find the chosen candidate by SKU (None if the LLM abstained).
    chosen = next((c for c in candidates
                   if c.contract.sku == verdict["chosen_sku"]), None)

    if verdict["is_ambiguous"]:
        decision = MatchDecision.UNCERTAIN
    elif chosen is None or verdict["confidence"] < LOW_BAR:
        decision = MatchDecision.NO_MATCH
    elif verdict["confidence"] >= HIGH_BAR:
        decision = MatchDecision.MATCH
    else:
        decision = MatchDecision.UNCERTAIN

    result = ReverseMapResult(
        order_id=order.order_id,
        decision=decision,
        confidence=verdict["confidence"],
        matched_sku=(chosen.contract.sku if chosen else None),
        matched_source=(chosen.contract.source if chosen else None),
        rationale=verdict["reason"],
    )
    if decision == MatchDecision.MATCH and chosen is not None:
        result.list_unit_price = order.list_unit_price
        result.contracted_unit_price = chosen.contract.contracted_unit_price
        result.quantity = order.quantity

    state["result"] = result
    return state
    
    


def node_human_gate(state: State) -> State:
    """Pause for human confirmation on UNCERTAIN matches. Built for you.

    Interview point: the gate sits before a recovery CLAIM. A human confirms the
    match before any money is clawed back, because the cost of a wrong claim is
    asymmetric. In production this is a review queue + LangGraph interrupt.
    """
    r = state["result"]
    print(f"\n=== CONFIRM MATCH: order {r.order_id} ===")
    print(f"  proposed SKU {r.matched_sku} from {r.matched_source} "
          f"(confidence {r.confidence:.2f})")
    print(f"  {r.rationale}")
    ans = input("Confirm this is the same item? [y/N]: ").strip().lower()
    r.human_confirmed = ans == "y"
    return state


def node_recover(state: State) -> State:
    """Record the recovery (idempotent + audited). Built for you."""
    record_recovery(state["result"])
    return state


# --- Routing -------------------------------------------------------------

def route_after_map(state: State) -> str:
    """Conditional edge after adjudication.

    # TODO(you): return "recover", "human_gate", or "end" based on
    # state["result"].decision (MATCH / UNCERTAIN / NO_MATCH).
    """
    decision = state["result"].decision
    if decision == MatchDecision.MATCH:
        return "recover"
    elif decision == MatchDecision.UNCERTAIN:
        return "human_gate"
    elif decision == MatchDecision.NO_MATCH:
        return "end"
    raise ValueError(f"Invalid decision: {decision}")


def route_after_gate(state: State) -> str:
    return "recover" if state["result"].human_confirmed else "end"


# --- Graph ---------------------------------------------------------------

def build_graph():
    """Wire the StateGraph.

    # TODO(you): assemble it. Sketch:
    #   from langgraph.graph import StateGraph, END
    #   g = StateGraph(State)
    #   g.add_node("retrieve", node_retrieve)
    #   g.add_node("reverse_map", node_reverse_map)
    #   g.add_node("human_gate", node_human_gate)
    #   g.add_node("recover", node_recover)
    #   g.set_entry_point("retrieve")
    #   g.add_edge("retrieve", "reverse_map")
    #   g.add_conditional_edges("reverse_map", route_after_map,
    #       {"recover": "recover", "human_gate": "human_gate", "end": END})
    #   g.add_conditional_edges("human_gate", route_after_gate,
    #       {"recover": "recover", "end": END})
    #   g.add_edge("recover", END)
    #   return g.compile()
    """
    from langgraph.graph import StateGraph, END

    g = StateGraph(State)

    #2 add the nodes, name on the left, function on the right
    g.add_node("retrieve", node_retrieve)
    g.add_node("reverse_map", node_reverse_map)
    g.add_node("human_gate", node_human_gate)
    g.add_node("recover", node_recover)

    #3a. where does it start 
    g.set_entry_point("retrieve")

    #3b. staright arrows
    g.add_edge("retrieve", "reverse_map")

    #3c. conditional edges, the fork
    g.add_conditional_edges("reverse_map", route_after_map, { 
        "recover": "recover", 
        "human_gate": "human_gate", 
        "end": END
        })
    g.add_conditional_edges("human_gate", route_after_gate, {
        "recover": "recover",
        "end" : END,
    })    

    return g.compile()


def run_one(order: OrderLine) -> ReverseMapResult:
    graph = build_graph()
    final = graph.invoke({"order": order})
    return final["result"]
