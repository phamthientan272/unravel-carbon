"""LangGraph agent: router → (report pipeline | quality pipeline | ReAct tool loop).

             ┌──────────► report ─────────────┐
    START → router ─────► quality ────────────┼──► END
             └──────────► agent ⇄ tools ──────┘

- Router: structured-output LLM call that classifies intent.
- Report / quality: fixed pipelines. Deterministic tools run first, then one LLM call writes
  up the results, so these are fast, repeatable and cheap.
- Agent: free-form ReAct loop with all tools, for data questions, GHG education,
  peer benchmarking and follow-ups.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from src import analytics, benchmark, quality
from src.data import ROOT, data_card
from src.grounding import build_quality_package, format_passages, pages_in, strip_unverified_citations
from src.llm import describe_llm, get_llm, text_of
from src.prompts import QUALITY_PROMPT, REPORT_PROMPT, ROUTER_PROMPT, SYSTEM_PROMPT
from src.tools import ALL_TOOLS

Intent = Literal["data_question", "ghg_education", "quality_assessment", "peer_benchmark", "report", "other"]
HISTORY_WINDOW = 20  # messages kept in the agent's context


class Route(BaseModel):
    intent: Intent = Field(description="What the user wants")
    confidence: Literal["high", "low"] = Field("high", description="'low' if the message could fit several intents or depends on earlier turns")
    scope: Literal["all", "1", "2", "3"] = Field("all", description="Scope the question is about, if one is specific")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    scope: str
    notes: list[str]  # extra trace lines (e.g. the search queries the planner wrote)
    pages: list[int]  # pages of the provided documents retrieved during this turn; the final check allows only these
    low_confidence: bool


def _system() -> SystemMessage:
    return SystemMessage(SYSTEM_PROMPT.format(data_card=data_card()))


def _last_user(state: State) -> str:
    return next(text_of(m) for m in reversed(state["messages"]) if isinstance(m, HumanMessage))


# --- nodes -------------------------------------------------------------------------------

def router(state: State) -> dict:
    convo = "\n".join(f"{m.type}: {text_of(m)[:300]}" for m in state["messages"][-6:])
    route: Route = get_llm(0).with_structured_output(Route).invoke(ROUTER_PROMPT.format(conversation=convo))
    # reset the per-turn state, so nothing leaks from the previous turn
    return {"intent": route.intent, "scope": route.scope, "low_confidence": route.confidence == "low", "notes": [], "pages": []}


def route_next(state: State) -> str:
    if state.get("low_confidence"):  # unsure: the agent can call any tool, including assess_quality
        return "agent"
    return {"report": "report", "quality_assessment": "quality"}.get(state["intent"], "agent")


def quality_node(state: State) -> dict:
    question = _last_user(state)
    pkg = build_quality_package(question, state.get("scope") or "all")
    shown = {k: v for k, v in pkg.items() if k not in ("passages", "notes")}
    prompt = QUALITY_PROMPT.format(question=question, findings=json.dumps(shown, indent=1), passages=format_passages(pkg["passages"]))
    reply = get_llm().invoke([_system(), HumanMessage(prompt)])
    return {"messages": [AIMessage(text_of(reply), name="quality_pipeline")], "notes": pkg["notes"],
            "pages": sorted({p["page"] for p in pkg["passages"]})}


def report_node(state: State) -> dict:
    question = _last_user(state)
    facts = {
        "summary": analytics.emissions_summary(),
        "scope1_by_facility": analytics.breakdown(1, "facility"),
        "scope1_by_activity": analytics.breakdown(1, "activity_type"),
        "scope2_by_energy": analytics.breakdown(2, "energy_type"),
        "scope2_by_facility": analytics.breakdown(2, "facility"),
        "scope3_reclassified": analytics.top_scope3("reclassified_category", 6),
        "scope3_by_label": analytics.top_scope3("category", 3),
        "suppliers": analytics.supplier_priority()["suppliers"][:4],
        "quality": [{k: f[k] for k in ("id", "severity", "title")} for f in quality.run_quality_checks()["findings"]],
        "peers": benchmark.compare_peers()["ratios"],
    }
    prompt = REPORT_PROMPT.format(question=question, facts=json.dumps(facts, indent=1, default=str))
    reply = get_llm().invoke([_system(), HumanMessage(prompt)])
    return {"messages": [AIMessage(text_of(reply), name="report_pipeline")], "pages": sorted(pages_in(prompt))}


def agent_node(state: State) -> dict:
    llm = get_llm().bind_tools(ALL_TOOLS)
    reply = llm.invoke([_system(), *state["messages"][-HISTORY_WINDOW:]])
    return {"messages": [reply]}


def verify_node(state: State) -> dict:
    """Final check for every path: an answer may cite only pages that were retrieved during this turn."""
    last = state["messages"][-1]
    turn = state["messages"][next(i for i in range(len(state["messages"]) - 1, -1, -1)
                                  if isinstance(state["messages"][i], HumanMessage)) + 1:]
    allowed = set(state.get("pages") or [])
    for m in turn:
        if m.type == "tool":
            allowed |= pages_in(text_of(m))
    text, removed = strip_unverified_citations(text_of(last), allowed)
    if not removed:
        return {}
    return {"messages": [AIMessage(text, id=last.id, name=last.name)],
            "notes": [f"citation check: removed citations to pages that were not retrieved: {sorted(set(removed))}"]}


def build_graph(checkpointer=None):
    g = StateGraph(State)
    g.add_node("router", router)
    g.add_node("report", report_node)
    g.add_node("quality", quality_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(ALL_TOOLS))
    g.add_node("verify", verify_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_next, {"report": "report", "quality": "quality", "agent": "agent"})
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "verify"})
    g.add_edge("tools", "agent")
    g.add_edge("report", "verify")
    g.add_edge("quality", "verify")
    g.add_edge("verify", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def ask(graph, question: str, thread_id: str = "cli") -> dict:
    """Run one turn. Returns the answer text, the intent, and a trace of tool calls."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    trace, intent, answer = [{"step": "llm", "detail": describe_llm()}], None, ""
    for update in graph.stream({"messages": [HumanMessage(question)]}, config, stream_mode="updates"):
        for node, delta in update.items():
            if node == "router":
                intent = delta["intent"]
                trace.append({"step": "router", "detail": f"intent={delta['intent']}, scope={delta['scope']}"
                                                  + (", low confidence -> agent" if delta.get("low_confidence") else "")})
                continue
            for note in (delta or {}).get("notes", []):
                trace.append({"step": "retrieval", "detail": note})
            for m in (delta or {}).get("messages", []):
                if isinstance(m, AIMessage) and m.tool_calls:
                    for tc in m.tool_calls:
                        trace.append({"step": "tool_call", "detail": f"{tc['name']}({json.dumps(tc['args'])})"})
                elif m.type == "tool":
                    trace.append({"step": "tool_result", "detail": f"{m.name}: {text_of(m)[:300]}"})
                elif isinstance(m, AIMessage):
                    answer = text_of(m)
                    trace.append({"step": node, "detail": "answer generated"})
    return {"answer": answer, "intent": intent, "trace": trace}


if __name__ == "__main__":
    import sys

    graph = build_graph()
    out = ask(graph, " ".join(sys.argv[1:]) or "What were our highest Scope 3 emissions categories?")
    print("\n".join(f"  · {t['step']}: {t['detail']}" for t in out["trace"]))
    print("\n" + out["answer"])
