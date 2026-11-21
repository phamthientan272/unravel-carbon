"""Graph nodes and the full graph, run with a fake LLM (no network, no API key)."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import graph as g


class FakeLLM:
    """Stands in for the chat model: scripted replies, a fixed router decision, tool binding is a no-op."""

    def __init__(self, route: g.Route, replies: list[AIMessage]):
        self.route, self.replies = route, list(replies)
        self.prompts: list = []

    def with_structured_output(self, _schema):
        route = self.route
        return type("S", (), {"invoke": lambda _self, _prompt: route})()

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.prompts.append(messages)
        return self.replies.pop(0)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))

    def install(intent="data_question", confidence="high", replies=()):
        llm = FakeLLM(g.Route(intent=intent, confidence=confidence, scope="2"), list(replies))
        monkeypatch.setattr(g, "get_llm", lambda *_a, **_k: llm)
        return llm

    return install


def test_router_returns_per_turn_state(fake):
    fake(intent="report", confidence="low")
    out = g.router({"messages": [HumanMessage("hi")]})
    assert out == {"intent": "report", "scope": "2", "low_confidence": True, "notes": [], "pages": []}


def test_agent_path_calls_a_tool_then_answers_and_strips_unretrieved_citations(fake):
    fake(replies=[
        AIMessage("", tool_calls=[{"name": "get_emissions_summary", "args": {}, "id": "c1"}]),
        AIMessage("Totals as reported. See [GHG Protocol, p.999]."),
    ])
    out = g.ask(g.build_graph(), "What are our totals?", "t-agent")
    assert out["intent"] == "data_question"
    assert "p.999" not in out["answer"]
    steps = [t["step"] for t in out["trace"]]
    assert {"router", "tool_call", "tool_result", "retrieval"} <= set(steps)
    rec = json.loads(open(out["trace_path"]).read())
    assert any(e["step"] == "note" and "citation check" in e["detail"] for e in rec["events"])


def test_low_confidence_goes_to_the_agent_even_for_a_report_intent(fake):
    fake(intent="report", confidence="low", replies=[AIMessage("agent answer")])
    out = g.ask(g.build_graph(), "and scope 2?", "t-low")
    assert out["answer"] == "agent answer"


def test_report_pipeline_writes_up_computed_facts(fake):
    llm = fake(intent="report", replies=[AIMessage("REPORT, see [GHG Protocol, p.5]")])
    out = g.ask(g.build_graph(), "Give me an emissions report", "t-report")
    assert out["intent"] == "report" and out["answer"].startswith("REPORT")
    assert "scope1_by_facility" in llm.prompts[0][1].content  # the facts were put in the prompt


def test_quality_pipeline_keeps_citations_to_retrieved_pages(fake, monkeypatch):
    pkg = {"passages": [{"id": "P1", "cite": "[GHG, p.27]", "page": 27, "text": "text"}], "notes": ["queries[x]: y"],
           "scope": "all", "findings": []}
    monkeypatch.setattr(g, "build_quality_package", lambda question, scope: pkg)
    fake(intent="quality_assessment", replies=[AIMessage("Needs documentation [GHG, p.27]")])
    out = g.ask(g.build_graph(), "Is the data valid?", "t-quality")
    assert "p.27" in out["answer"]
    assert any(t["step"] == "retrieval" and "queries[x]" in t["detail"] for t in out["trace"])
