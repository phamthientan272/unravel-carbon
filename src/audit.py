"""Audit trace: one JSON file per question, written to a folder (a database can replace this later).

Each file records the whole flow of a turn in order: the LLM in use, the router decision, every
tool call with its full arguments and result (linked by `tool_call_id`), retrieval notes, the citation
check, token usage, per-step timing, the final answer, and any error. Unlike the short `trace` shown in
the UI, tool results are kept in full (up to MAX_FIELD_CHARS), so an answer can be re-checked later.

Config (environment):
    TRACE_DIR      folder for the files (default: <repo>/traces)
    TRACE_ENABLED  set to 0/false to turn tracing off
Writing a trace never breaks an answer: failures are logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.data import ROOT

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1
MAX_FIELD_CHARS = 20_000


def enabled() -> bool:
    return os.getenv("TRACE_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


def trace_dir() -> Path:
    return Path(os.getenv("TRACE_DIR") or ROOT / "traces")


def _clip(text: str) -> tuple[str, bool]:
    return (text, False) if len(text) <= MAX_FIELD_CHARS else (text[:MAX_FIELD_CHARS], True)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=2)
        return out.stdout.strip() or None
    except Exception:
        return None


class TraceRecorder:
    """Collects events for one turn and writes them as a single JSON file."""

    def __init__(self, question: str, thread_id: str, llm: str):
        self.started = time.perf_counter()
        self._last = self.started
        now = datetime.now(timezone.utc)
        self.record: dict = {
            "schema_version": SCHEMA_VERSION,
            "trace_id": f"{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}",
            "timestamp_utc": now.isoformat(timespec="seconds"),
            "thread_id": thread_id,
            "question": question,
            "llm": llm,
            "code_version": _git_commit(),
            "intent": None,
            "scope": None,
            "low_confidence": None,
            "events": [],
            "answer": None,
            "pages_retrieved": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "duration_s": None,
            "error": None,
        }

    def event(self, step: str, **fields) -> None:
        """Append an event. `elapsed_s` is the time since the previous event (roughly: how long the step took)."""
        now = time.perf_counter()
        for k, v in list(fields.items()):
            if isinstance(v, str):
                fields[k], clipped = _clip(v)
                if clipped:
                    fields[f"{k}_truncated"] = True
        self.record["events"].append({"seq": len(self.record["events"]) + 1, "step": step,
                                      "elapsed_s": round(now - self._last, 3), **fields})
        self._last = now

    def add_usage(self, usage: dict | None) -> None:
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            self.record["usage"][k] += int((usage or {}).get(k) or 0)

    def finish(self, answer: str | None = None, error: BaseException | None = None) -> Path | None:
        r = self.record
        r["answer"] = answer
        r["duration_s"] = round(time.perf_counter() - self.started, 3)
        if error is not None:
            r["error"] = f"{type(error).__name__}: {str(error)[:1000]}"
        try:
            d = trace_dir()
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{r['trace_id']}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(r, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            tmp.replace(path)  # atomic: readers never see a half-written file
            return path
        except Exception:
            log.exception("could not write audit trace")
            return None
