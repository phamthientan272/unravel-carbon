"""End-to-end evaluation of the agent on the test questions.

Checks per case: (1) router picked the expected intent, (2) required facts appear in the answer,
(3) optional LLM-as-judge score for groundedness and helpfulness (--judge).
Answers are saved to eval/results/ for review and screenshots.

    uv run python eval/run_eval.py [--judge] [--only core2_scope2_validity]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph import ask, build_graph, get_llm, text_of  # noqa: E402

RESULTS = ROOT / "eval" / "results"

JUDGE_PROMPT = """\
You are grading an emissions-analysis assistant. Question: {question}

Tool trace (what the assistant looked up):
{trace}

Answer:
{answer}

Score 1-5 for each, then give one sentence of justification. Reply as JSON:
{{"grounded": n, "correct_ghg_methodology": n, "actionable": n, "concise": n, "comment": "..."}}
- grounded: numbers and claims are supported by the trace or cited documents (no invented figures)
- correct_ghg_methodology: consistent with the GHG Protocol
- actionable: gives the user clear next steps where relevant
- concise: no filler or repetition
"""


def check_facts(answer: str, must_include: list) -> list[str]:
    a = answer.lower()
    missing = []
    for item in must_include:
        options = item if isinstance(item, list) else [item]
        if not any(o.lower() in a for o in options):
            missing.append(" | ".join(options))
    return missing


def ask_with_retry(graph, case: dict, attempts: int = 3) -> dict:
    """Retry transient 503 'high demand' errors with backoff. Quota errors (429) are not retried."""
    for i in range(attempts):
        try:
            # fresh thread per attempt so a half-finished run doesn't pollute the conversation
            return ask(graph, case["question"], thread_id=f"eval-{case['id']}-{i}")
        except Exception as e:
            if "503" not in str(e) and "UNAVAILABLE" not in str(e) or i == attempts - 1:
                raise
            wait = 15 * (i + 1)
            print(f"  transient 503 on {case['id']}, retrying in {wait}s", flush=True)
            time.sleep(wait)


def judge(case: dict, out: dict) -> dict:
    trace = "\n".join(f"{t['step']}: {t['detail'][:200]}" for t in out["trace"])
    raw = text_of(get_llm(0).invoke(JUDGE_PROMPT.format(question=case["question"], trace=trace, answer=out["answer"])))
    try:
        return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except json.JSONDecodeError:
        return {"comment": f"unparseable judge output: {raw[:200]}"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--judge", action="store_true")
    p.add_argument("--only")
    args = p.parse_args()

    cases = yaml.safe_load((ROOT / "eval" / "test_questions.yaml").read_text())["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
    RESULTS.mkdir(parents=True, exist_ok=True)
    graph = build_graph()
    summary = []
    for case in cases:
        t0 = time.time()
        try:
            out = ask_with_retry(graph, case)
        except Exception as e:  # e.g. free-tier 429s: record and keep going
            print(f"ERROR {case['id']:<26} {type(e).__name__}: {str(e)[:160]}")
            summary.append({"id": case["id"], "error": f"{type(e).__name__}: {str(e)[:300]}"})
            continue
        secs = time.time() - t0
        missing = check_facts(out["answer"], case["must_include"])
        intent_ok = out["intent"] in case["expected_intent"]
        row = {"id": case["id"], "intent": out["intent"], "intent_ok": intent_ok, "missing_facts": missing,
               "facts_ok": not missing, "seconds": round(secs, 1),
               "tools": [t["detail"].split("(")[0] for t in out["trace"] if t["step"] == "tool_call"]}
        if args.judge:
            row["judge"] = judge(case, out)
        summary.append(row)
        (RESULTS / f"{case['id']}.md").write_text(
            f"# {case['question']}\n\n_intent: {out['intent']} · {secs:.1f}s_\n\n{out['answer']}\n\n---\n### Trace\n"
            + "\n".join(f"- **{t['step']}**: `{t['detail'][:300]}`" for t in out["trace"]))
        status = "PASS" if intent_ok and not missing else "FAIL"
        print(f"{status} {case['id']:<26} intent={out['intent']:<20} {secs:5.1f}s missing={missing}")

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    passed = sum(r.get("intent_ok", False) and r.get("facts_ok", False) for r in summary)
    print(f"\n{passed}/{len(summary)} cases passed. Details in {RESULTS}/")


if __name__ == "__main__":
    main()
