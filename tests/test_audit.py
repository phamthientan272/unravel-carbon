import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src import audit
from src import graph as g


class FakeGraph:
    def stream(self, *_a, **_k):
        yield {"router": {"intent": "data_question", "scope": "all", "low_confidence": False, "notes": [], "pages": []}}
        yield {"agent": {"messages": [AIMessage("", tool_calls=[{"name": "t", "args": {"a": 1}, "id": "c1"}],
                                                usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7})]}}
        yield {"tools": {"messages": [ToolMessage("full result", name="t", tool_call_id="c1")]}}
        yield {"quality": {"messages": [], "notes": ["searched X"], "pages": [3, 1]}}
        yield {"agent": {"messages": [AIMessage("done", usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4})]}}


def _load(path):
    return json.loads(next(path.glob("*.json")).read_text())


def test_ask_writes_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    out = g.ask(FakeGraph(), "q?", "th")
    rec = _load(tmp_path)
    assert out["trace_id"] == rec["trace_id"] and out["trace_path"].endswith(".json")
    assert [e["step"] for e in rec["events"]] == ["router", "tool_calls", "tool_result", "note", "answer"]
    assert rec["events"][1]["calls"][0]["tool_call_id"] == rec["events"][2]["tool_call_id"] == "c1"
    assert rec["events"][0]["next_node"] == "agent"
    assert rec["answer"] == "done" and rec["usage"]["total_tokens"] == 11
    assert rec["pages_retrieved"] == [1, 3] and rec["intent"] == "data_question" and rec["error"] is None
    assert [t["step"] for t in out["trace"]][1:] == ["router", "tool_call", "tool_result", "retrieval", "agent"]


def test_error_is_traced(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))

    class Boom:
        def stream(self, *_a, **_k):
            raise RuntimeError("llm down")
            yield

    with pytest.raises(RuntimeError):
        g.ask(Boom(), "q?")
    assert "llm down" in _load(tmp_path)["error"]


def test_disabled_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("TRACE_ENABLED", "0")
    out = g.ask(FakeGraph(), "q?")
    assert not list(tmp_path.glob("*.json")) and "trace_id" not in out


def test_long_fields_are_clipped(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    rec = audit.TraceRecorder("q", "t", "llm")
    rec.event("tool_result", result="x" * (audit.MAX_FIELD_CHARS + 5))
    rec.finish("a")
    ev = _load(tmp_path)["events"][0]
    assert len(ev["result"]) == audit.MAX_FIELD_CHARS and ev["result_truncated"] is True


def test_write_failure_is_swallowed(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("TRACE_DIR", str(blocker / "sub"))  # cannot mkdir under a file
    assert audit.TraceRecorder("q", "t", "llm").finish("a") is None


def test_git_commit_failure_returns_none(monkeypatch):
    def boom(*_a, **_k):
        raise OSError

    monkeypatch.setattr(audit.subprocess, "run", boom)
    assert audit._git_commit() is None
