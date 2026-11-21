"""LangChain tool wrappers around the deterministic modules.

Docstrings are what the LLM sees, so they say when to use each tool.
Every tool returns compact JSON.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import tool

from src import analytics, quality
from src.grounding import build_quality_package
from src.kb import get_kb


def _json(obj) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


@tool
def get_emissions_summary() -> str:
    """Total emissions by scope (tCO2e), shares, reporting periods, and an annualised estimate.
    Use for any 'total', 'by scope', 'overall footprint' or report question."""
    return _json(analytics.emissions_summary())


@tool
def get_breakdown(scope: Literal["1", "2", "3"], by: str) -> str:
    """Group one scope's emissions by a column and rank the groups.
    scope1 columns: facility, activity_type, fuel_type, month.
    scope2 columns: facility, energy_type, grid_region, month.
    scope3 columns: supplier, activity_description, category, data_quality, month, quarter."""
    return _json(analytics.breakdown(int(scope), by))  # string enum: every provider's tool schema supports it


@tool
def get_top_scope3(by: Literal["category", "activity", "reclassified_category"] = "category", top_n: int = 5) -> str:
    """Rank Scope 3 emissions, with the top activities and suppliers inside each group.
    by='category' uses the category labels in the data (92% are inconsistent with the activity);
    by='activity' ranks activity descriptions; by='reclassified_category' maps activities to the GHG Protocol category
    inferred for them (LLM-inferred mapping, unverified: indicative only). For 'highest category' questions, call it with 'category' AND 'reclassified_category'."""
    return _json(analytics.top_scope3(by, top_n))


@tool
def get_supplier_priority() -> str:
    """Rank Scope 3 suppliers for engagement (emissions, carbon intensity per USD, share of low-quality data,
    supplier/activity mismatches) with a suggested action for each. Use for supplier engagement and reduction questions."""
    return _json(analytics.supplier_priority())


@tool
def run_sql(query: str) -> str:
    """Run one read-only DuckDB SELECT over tables scope1, scope2, scope3 for questions the other tools can't answer
    (filters, time ranges, specific facilities or suppliers). Columns are lowercase snake_case, as described in the data card.
    Dates are in column `date` (timestamp); `month` is 'YYYY-MM' and `quarter` is e.g. '2024Q3'."""
    return _json(analytics.run_sql(query))


@tool
def run_quality_checks(scope: Literal["all", "1", "2", "3"] = "all") -> str:
    """Run rule-based quality checks on the inventory: arithmetic, period coverage, factor documentation, renewable
    share, double-counting risk, category and supplier labels, factor consistency, data-quality labels. Returns findings
    (facts about the data) with severity and computed evidence. It does NOT say what the GHG Protocol requires: use
    search_knowledge_base for that. Use for any validity, accuracy or quality question."""
    return _json(quality.run_quality_checks(scope))


@tool
def assess_quality(question: str, scope: Literal["all", "1", "2", "3"] = "all") -> str:
    """Assess data quality against what the provided GHG Protocol document says. Returns the quality findings (facts about
    the data) AND passages from the GHG Protocol linked to each finding by `passage_ids`. Use it for follow-ups about a
    specific quality issue or about whether the data follows the GHG Protocol. Pass the user's question, so the search
    queries fit it. Cite only the pages of the returned passages. If no passage addresses a finding, say the provided
    document does not address it."""
    pkg = build_quality_package(question, scope, text_chars=700)
    return _json({k: v for k, v in pkg.items() if k != "notes"})


@tool
def calculate(expression: str) -> str:
    """Exact arithmetic for figures you already have (ratios, sums, shares, unit conversions): + - * / ** and parentheses
    over plain numbers, e.g. '16790.1 / 449'. Use it instead of calculating yourself, with numbers taken from tool results
    or retrieved passages."""
    return _json(analytics.calculate(expression))


@tool
def search_knowledge_base(query: str, source: Literal["all", "ghg_protocol", "peer1", "peer2"] = "all") -> str:
    """Search the GHG Protocol Corporate Standard and the peer reports. Returns passages with document and page number.
    Use for definitions, methodology and scope classification, and for the peers' figures and narrative: pass
    source='peer1' (DocuSign) or 'peer2' (Shopify). Peer tables may come back as flattened text: use a figure only if
    the passage makes clear which row and year it belongs to.
    Write a specific query (e.g. 'business travel scope 3 employee-owned vehicles')."""
    src = {"all": None, "ghg_protocol": "ghg-protocol-revised.pdf", "peer1": "peer1_emissions_report.pdf",
           "peer2": "peer2_emissions_report.pdf"}[source]
    hits = get_kb().search(query, k=6, source=src)
    return _json([{"cite": f"[{h['title']}, p.{h['page']}]", "text": h["text"]} for h in hits])


ALL_TOOLS = [get_emissions_summary, get_breakdown, get_top_scope3, get_supplier_priority, run_sql,
             run_quality_checks, assess_quality, calculate, search_knowledge_base]
