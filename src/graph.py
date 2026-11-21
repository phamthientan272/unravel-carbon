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
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from src import analytics, benchmark, quality
from src.data import ROOT, data_card
from src.kb import get_kb
from src.prompts import QUALITY_PROMPT, REPORT_PROMPT, ROUTER_PROMPT, SYSTEM_PROMPT
from src.tools import ALL_TOOLS

load_dotenv(ROOT / ".env")

Intent = Literal["data_question", "ghg_education", "quality_assessment", "peer_benchmark", "report", "other"]
HISTORY_WINDOW = 20  # messages kept in the agent's context


class Route(BaseModel):
    intent: Intent = Field(description="What the user wants")
    scope: Literal["all", "1", "2", "3"] = Field("all", description="Scope the question is about, if one is specific")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    scope: str


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL", "gemini-4-hydrogen"),
        google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
        max_retries=2,
        timeout=120,  # seconds; free-tier calls occasionally hang
    )


def text_of(msg: AnyMessage) -> str:
    """Message content as plain text (Gemini may return a list of content blocks)."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


def _system() -> SystemMessage:
    return SystemMessage(SYSTEM_PROMPT.format(data_card=data_card()))


def _last_user(state: State) -> str:
    return next(text_of(m) for m in reversed(state["messages"]) if isinstance(m, HumanMessage))


def _cite_passages(queries: list[str], k: int = 2) -> str:
    """Retrieve GHG Protocol passages for a set of queries, de-duplicated, formatted with citations."""
    kb, seen, out = get_kb(), set(), []
    for q in queries:
        for h in kb.search(q, k=k, source="ghg-protocol-revised.pdf"):
            key = (h["page"], h["text"][:50])
            if key not in seen:
                seen.add(key)
                out.append(f"[{h['title']}, p.{h['page']}]\n{h['text']}")
    return "\n\n---\n\n".join(out)


# --- nodes -------------------------------------------------------------------------------

def router(state: State) -> dict:
    convo = "\n".join(f"{m.type}: {text_of(m)[:300]}" for m in state["messages"][-6:])
    route: Route = get_llm(0).with_structured_output(Route).invoke(ROUTER_PROMPT.format(conversation=convo))
    return {"intent": route.intent, "scope": route.scope}


def route_next(state: State) -> str:
    return {"report": "report", "quality_assessment": "quality"}.get(state["intent"], "agent")


def quality_node(state: State) -> dict:
    question = _last_user(state)
    result = quality.run_quality_checks(state.get("scope") or "all")
    important = [f for f in result["findings"] if f["severity"] in ("high", "medium")]
    passages = _cite_passages([f["kb_query"] for f in important[:5]])
    # Passed checks are reported without citations; stripping their references avoids decorative, unsupported cites.
    shown = {**result, "findings": [
        {**f, "ghg_reference": "", "kb_query": ""} if f["severity"] == "pass" else f for f in result["findings"]]}
    prompt = QUALITY_PROMPT.format(question=question, findings=json.dumps(shown, indent=1), passages=passages)
    reply = get_llm().invoke([_system(), HumanMessage(prompt)])
    return {"messages": [AIMessage(text_of(reply), name="quality_pipeline")]}


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
    return {"messages": [AIMessage(text_of(reply), name="report_pipeline")]}


def agent_node(state: State) -> dict:
    llm = get_llm().bind_tools(ALL_TOOLS)
    reply = llm.invoke([_system(), *state["messages"][-HISTORY_WINDOW:]])
    return {"messages": [reply]}


def build_graph(checkpointer=None):
    g = StateGraph(State)
    g.add_node("router", router)
    g.add_node("report", report_node)
    g.add_node("quality", quality_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(ALL_TOOLS))

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_next, {"report": "report", "quality": "quality", "agent": "agent"})
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    g.add_edge("report", END)
    g.add_edge("quality", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def ask(graph, question: str, thread_id: str = "cli") -> dict:
    """Run one turn. Returns the answer text, the intent, and a trace of tool calls."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    trace, intent, answer = [], None, ""
    for update in graph.stream({"messages": [HumanMessage(question)]}, config, stream_mode="updates"):
        for node, delta in update.items():
            if node == "router":
                intent = delta["intent"]
                trace.append({"step": "router", "detail": f"intent={delta['intent']}, scope={delta['scope']}"})
                continue
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
