"""Grounding: tie data-quality findings to what the provided GHG Protocol document says.

The reusable core of the quality assessment, used by the fixed quality pipeline and by the agent's
`assess_quality` tool:

1. run the deterministic quality rules (facts about the data),
2. have an LLM write search queries from the user's question and the findings,
3. search the provided GHG Protocol document with them and link passages to findings.

`pages_in` and `strip_unverified_citations` are the shared citation check: an answer may cite only pages that were
actually retrieved during the turn.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.kb import get_kb
from src.llm import get_llm
from src.prompts import QUERY_PLAN_PROMPT
from src.quality import run_quality_checks

GHG_DOC = "ghg-protocol-revised.pdf"
_CITATION = re.compile(r"\[[^\[\]]*?p\.\s*(\d+)[^\[\]]*\]")
_PAGES = re.compile(r"\bpp?\.\s*\[?(\d+(?:\s*[,\-–]\s*\d+)*)\]?|\"page\":\s*(\d+)")


class FindingQueries(BaseModel):
    finding_id: str
    queries: list[str] = Field(description="1-2 short search queries for the GHG Protocol text")


class QueryPlan(BaseModel):
    items: list[FindingQueries]
    question_queries: list[str] = Field(default_factory=list, description="1-2 queries for the user's question itself")


def pages_in(text: str) -> set[int]:
    """Every page number mentioned in a text as 'p.N', 'p.[N, M]', 'pp.N-M' or '"page": N'."""
    pages: set[int] = set()
    for m in _PAGES.finditer(text):
        if m.group(2):
            pages.add(int(m.group(2)))
        else:
            pages.update(int(n) for n in re.findall(r"\d+", m.group(1)))
    return pages


def strip_unverified_citations(text: str, allowed_pages: set[int]) -> tuple[str, list[int]]:
    """Remove page citations whose page was not retrieved. Returns the text and the removed pages."""
    removed: list[int] = []

    def check(m: re.Match) -> str:
        page = int(m.group(1))
        if page in allowed_pages:
            return m.group(0)
        removed.append(page)
        return ""

    return _CITATION.sub(check, text), removed


def plan_queries(question: str, findings: list[dict]) -> QueryPlan:
    """Ask the LLM to write search queries from the user's question and the findings. Falls back to fixed queries."""
    brief = "\n".join(f"- {f['id']}: {f['title']}. {f['headline']}" for f in findings)
    try:
        return get_llm(0).with_structured_output(QueryPlan).invoke(QUERY_PLAN_PROMPT.format(question=question, findings=brief))
    except Exception:  # query writing must never break the assessment
        return QueryPlan(items=[FindingQueries(finding_id=f["id"], queries=[f["fallback_query"]]) for f in findings],
                         question_queries=[question])


def retrieve_for_findings(plan: QueryPlan, findings: list[dict], k: int = 3) -> tuple[dict[str, list[str]], list[dict]]:
    """Search the GHG Protocol document for every planned query. Returns {finding_id: [passage ids]} and the passages."""
    kb, passages, index = get_kb(), [], {}

    def add(h: dict, ids: list[str]) -> None:
        key = (h["page"], h["text"][:60])
        if key not in index:
            index[key] = f"P{len(index) + 1}"
            passages.append({"id": index[key], "cite": f"[{h['title']}, p.{h['page']}]", "page": int(h["page"]), "text": h["text"]})
        if index[key] not in ids:
            ids.append(index[key])

    def fetch(queries: list[str], pages: list[int] | None = None) -> list[str]:
        ids: list[str] = []
        for p in (pages or [])[:2]:  # pages the catalog says state the requirement: always retrieved, so they can be cited
            for h in kb.page(GHG_DOC, p):
                add(h, ids)
        for q in queries[:2]:
            for h in kb.search(q, k=k, source=GHG_DOC):
                add(h, ids)
        return ids

    wanted = {f["id"] for f in findings}
    by_id = {it.finding_id: it.queries for it in plan.items if it.finding_id in wanted}
    links = {f["id"]: fetch(by_id.get(f["id"]) or [f["fallback_query"]], f.get("ref_pages")) for f in findings}
    links["QUESTION"] = fetch(plan.question_queries)
    return links, passages


def build_quality_package(question: str, scope: str = "all", max_findings: int = 8, text_chars: int | None = None) -> dict:
    """Findings + the document passages linked to them + a trace of the queries used.

    `text_chars` shortens each passage (the agent tool uses it to keep the tool result small).
    """
    result = run_quality_checks(scope)
    # requirement and qa findings first: those are the ones the document speaks to
    open_ = [f for f in result["findings"] if f["severity"] in ("high", "medium")]
    open_.sort(key=lambda f: f["layer"] == "investigation")
    important = open_[:max_findings]
    plan = plan_queries(question, important)
    links, passages = retrieve_for_findings(plan, important)
    findings = [
        {k: v for k, v in f.items() if k not in ("fallback_query", "headline", "extra")}
        | ({"passage_ids": links[f["id"]]} if f["id"] in links else {})
        for f in result["findings"]
    ]
    if text_chars:
        passages = [{**p, "text": p["text"][:text_chars]} for p in passages]
    notes = [f"queries[{it.finding_id}]: {' | '.join(it.queries)}" for it in plan.items]
    notes.append(f"queries[question]: {' | '.join(plan.question_queries)}")
    notes.append(f"retrieved {len(passages)} passages, pages {sorted({p['page'] for p in passages})}")
    return {
        "scope": scope, "severity_counts": result["severity_counts"], "open_by_layer": result["open_by_layer"],
        "principle_verdicts": result["principle_verdicts"], "cannot_assess": result["cannot_assess"], "findings": findings,
        "passage_ids_for_the_question": links["QUESTION"], "passages": passages, "notes": notes,
    }


def format_passages(passages: list[dict]) -> str:
    return "\n\n---\n\n".join(f"{p['id']} {p['cite']}\n{p['text']}" for p in passages)
