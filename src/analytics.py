"""Deterministic analytics over the inventory.

Every number the agent reports comes from here (or from `run_sql`), never from the
LLM doing arithmetic. Functions return plain JSON-serialisable dicts so they can be
used both as LLM tools and by the Streamlit UI.
"""

from __future__ import annotations

import ast
import operator
from typing import Literal

import duckdb
import pandas as pd

from src.data import Inventory, LOW_QUALITY_LABELS, get_inventory

MONTHS_IN_SCOPE12 = 3  # scope 1/2 only cover Q1 2024


def _r(x: float, nd: int = 1) -> float:
    return round(float(x), nd)


def _records(df: pd.DataFrame, nd: int = 1) -> list[dict]:
    return df.round(nd).to_dict(orient="records")


def emissions_summary(inv: Inventory | None = None) -> dict:
    """Totals by scope, shares, and the annualised view needed to compare scopes fairly."""
    inv = inv or get_inventory()
    s1 = inv.scope1.co2e_tonnes.sum()
    s2 = inv.scope2.co2e_tonnes.sum()
    s3 = inv.scope3.co2e_tonnes.sum()
    total = s1 + s2 + s3
    factor = 12 / MONTHS_IN_SCOPE12
    s1_ann, s2_ann = s1 * factor, s2 * factor
    total_ann = s1_ann + s2_ann + s3
    return {
        "as_reported": {
            "scope1_tco2e": _r(s1), "scope1_period": "2024-01 to 2024-03",
            "scope2_location_based_tco2e": _r(s2), "scope2_period": "2024-01 to 2024-03",
            "scope3_tco2e": _r(s3), "scope3_period": "2024-01 to 2024-12",
            "total_tco2e": _r(total),
            "share_pct": {"scope1": _r(100 * s1 / total), "scope2": _r(100 * s2 / total), "scope3": _r(100 * s3 / total)},
        },
        "annualised_estimate": {
            "method": "Scope 1 and 2 Q1 totals multiplied by 4 (assumes no seasonality); Scope 3 as reported",
            "scope1_tco2e": _r(s1_ann), "scope2_tco2e": _r(s2_ann), "scope3_tco2e": _r(s3),
            "total_tco2e": _r(total_ann),
            "share_pct": {
                "scope1": _r(100 * s1_ann / total_ann), "scope2": _r(100 * s2_ann / total_ann), "scope3": _r(100 * s3 / total_ann),
            },
        },
    }


def breakdown(scope: int, by: str, inv: Inventory | None = None, top_n: int = 15) -> dict:
    """Group a scope's emissions by any column (e.g. facility, activity_type, energy_type, month, supplier)."""
    inv = inv or get_inventory()
    df = {1: inv.scope1, 2: inv.scope2, 3: inv.scope3}[scope]
    if by not in df.columns:
        return {"error": f"Column '{by}' not in scope{scope}. Available: {sorted(df.columns)}"}
    agg = {"co2e_tonnes": "sum"}
    if "spend_amount" in df.columns:
        agg["spend_amount"] = "sum"
    g = df.groupby(by).agg(agg).sort_values("co2e_tonnes", ascending=False)
    g["share_pct"] = 100 * g["co2e_tonnes"] / g["co2e_tonnes"].sum()
    return {
        "scope": scope, "by": by, "total_tco2e": _r(df.co2e_tonnes.sum()),
        "rows": _records(g.head(top_n).reset_index()),
    }


def top_scope3(by: Literal["category", "activity", "reclassified_category"] = "category", top_n: int = 5,
               inv: Inventory | None = None) -> dict:
    """Rank Scope 3 emissions three ways.

    - category: as labelled in the data (the client's reported view; labels are inconsistent)
    - activity: by activity_description
    - reclassified_category: activities mapped to an LLM-inferred GHG Protocol category (unverified, indicative)
    Each group lists its top contributing activities and suppliers.
    """
    inv = inv or get_inventory()
    s3 = inv.scope3
    key = {"category": ["category_num", "category_name"], "activity": ["activity_description"],
           "reclassified_category": ["inferred_category_num", "inferred_category_name"]}[by]
    total = s3.co2e_tonnes.sum()
    g = (s3.groupby(key).agg(co2e_tonnes=("co2e_tonnes", "sum"), spend_usd=("spend_amount", "sum"), rows=("co2e_tonnes", "size"))
         .sort_values("co2e_tonnes", ascending=False))
    out = []
    for idx, row in g.head(top_n).iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        sub = s3.loc[(s3[key] == pd.Series(idx, index=key)).all(axis=1)]

        def contributors(col: str) -> list[dict]:
            g2 = sub.groupby(col).agg(co2e_tonnes=("co2e_tonnes", "sum"), spend_usd=("spend_amount", "sum"))
            g2["share_of_group_pct"] = 100 * g2.co2e_tonnes / g2.co2e_tonnes.sum()
            return _records(g2.nlargest(5, "co2e_tonnes").reset_index())

        out.append({
            **dict(zip(key, idx)),
            "co2e_tonnes": _r(row.co2e_tonnes), "share_pct": _r(100 * row.co2e_tonnes / total),
            "spend_usd": _r(row.spend_usd, 0), "rows": int(row.rows),
            "top_activities": contributors("activity_description"),
            "top_suppliers": contributors("supplier"),
        })
    spread = g.co2e_tonnes
    labelled_top = None
    if by != "category":  # always surface what the raw labels say, so answers can contrast the two views
        lab = s3.groupby(["category_num", "category_name"]).co2e_tonnes.sum().nlargest(1)
        (num, name), tonnes = next(iter(lab.items()))
        labelled_top = {"category": f"Category {num}", "name": name, "co2e_tonnes": _r(tonnes),
                        "share_pct": _r(100 * tonnes / total),
                        "note": "top category according to the (unreliable) category labels in the data"}
    return {
        "ranked_by": by, "scope3_total_tco2e": _r(total), "groups_total": len(g),
        "top": out,
        "top_by_raw_labels": labelled_top,
        "spread_note": f"largest group {100 * spread.max() / total:.1f}% vs smallest {100 * spread.min() / total:.1f}% of Scope 3",
        "caveat": (f"{s3.category_mismatch.mean():.0%} of rows carry a category label inconsistent with their activity "
                   f"(random assignment would match about {1 / s3.category_num.nunique():.0%}); labels carry little information. "
                   "Compare with 'activity' and 'reclassified_category'.") if by == "category" else
                  ("The activity-to-category mapping behind this view is LLM-inferred and not verified by the client or an expert; "
                   "treat it as indicative and present it next to the labelled view." if by == "reclassified_category" else None),
    }


def supplier_priority(inv: Inventory | None = None) -> dict:
    """Rank suppliers for engagement.

    Score = 50% emissions share + 30% carbon intensity (kg/USD) + 20% share of low-quality data,
    each min-max normalised across suppliers. Recommends an engagement action per supplier.
    """
    inv = inv or get_inventory()
    s3 = inv.scope3.copy()
    s3["low_quality"] = s3.data_quality.isin(LOW_QUALITY_LABELS)
    g = s3.groupby("supplier").agg(
        co2e_tonnes=("co2e_tonnes", "sum"), spend_usd=("spend_amount", "sum"), rows=("co2e_tonnes", "size"),
        low_quality_share=("low_quality", "mean"), activity_mismatch_share=("supplier_mismatch", "mean"),
    )
    g["share_pct"] = 100 * g.co2e_tonnes / g.co2e_tonnes.sum()
    g["intensity_kg_per_usd"] = 1000 * g.co2e_tonnes / g.spend_usd

    def norm(s: pd.Series) -> pd.Series:
        return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

    g["priority_score"] = 0.5 * norm(g.co2e_tonnes) + 0.3 * norm(g.intensity_kg_per_usd) + 0.2 * norm(g.low_quality_share)
    g = g.sort_values("priority_score", ascending=False)
    top_acts = s3.groupby(["supplier", "activity_description"]).co2e_tonnes.sum().reset_index()

    def action(r: pd.Series) -> str:
        acts = []
        if r.intensity_kg_per_usd >= g.intensity_kg_per_usd.median():
            acts.append("above-median carbon intensity: discuss lower-carbon materials/logistics and set reduction targets")
        if r.low_quality_share >= 0.5:
            acts.append("mostly proxy/secondary data: request supplier-specific (primary) emission data or product carbon footprints")
        if r.activity_mismatch_share >= 0.5:
            acts.append("activity records don't match this supplier's business: fix supplier mapping first")
        return "; ".join(acts) or "monitor"

    rows = []
    for name, r in g.iterrows():
        acts = top_acts[top_acts.supplier == name].nlargest(3, "co2e_tonnes")
        rows.append({
            "supplier": name, "priority_score": _r(r.priority_score, 2), "co2e_tonnes": _r(r.co2e_tonnes),
            "share_pct": _r(r.share_pct), "spend_usd": _r(r.spend_usd, 0),
            "intensity_kg_per_usd": _r(r.intensity_kg_per_usd, 3),
            "low_quality_data_pct": _r(100 * r.low_quality_share), "activity_mismatch_pct": _r(100 * r.activity_mismatch_share),
            "top_activities": _records(acts[["activity_description", "co2e_tonnes"]]),
            "suggested_action": action(r),
        })
    return {
        "method": "score = 0.5*emissions + 0.3*intensity + 0.2*low-quality-data share (min-max normalised)",
        "intensity_caveat": ("Carbon intensity is driven by emission factors that look arbitrary (similar medians for every activity, "
                             "very wide ranges, uncorrelated with spend), and suppliers often do not match their activities. "
                             "Treat this ranking as indicative, not as evidence of which supplier is worse."),
        "spread_note": f"top supplier {g.share_pct.max():.1f}% vs bottom {g.share_pct.min():.1f}% of Scope 3: emissions are evenly spread",
        "suppliers": rows,
    }


# --- Long-tail questions: read-only SQL over the in-memory DataFrames ---------------------

_FORBIDDEN = ("insert", "update", "delete", "drop", "create", "alter", "attach", "copy", "pragma", "install", "load", "export")


def run_sql(query: str, inv: Inventory | None = None, max_rows: int = 50) -> dict:
    """Run a single read-only SELECT over tables scope1, scope2, scope3 (DuckDB, in-memory)."""
    inv = inv or get_inventory()
    q = query.strip().rstrip(";")
    lowered = q.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")) or ";" in q or any(f" {w} " in f" {lowered} " for w in _FORBIDDEN):
        return {"error": "Only a single read-only SELECT/WITH query is allowed."}
    con = duckdb.connect(database=":memory:")
    try:
        for name in ("scope1", "scope2", "scope3"):
            con.register(name, getattr(inv, name))
        df = con.execute(q).df()
    except Exception as e:  # surface SQL errors back to the model so it can fix the query
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        con.close()
    return {"row_count": len(df), "truncated": len(df) > max_rows,
            "rows": df.head(max_rows).round(3).astype(object).where(df.head(max_rows).notna(), None).to_dict(orient="records")}


# --- Exact arithmetic for the agent: the model never calculates itself ---------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression: str) -> dict:
    """Evaluate plain arithmetic (numbers, + - * / ** and parentheses). Anything else is refused."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers, + - * / ** and parentheses are allowed")

    try:
        result = ev(ast.parse(expression.strip().replace(",", ""), mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
        return {"expression": expression, "error": str(e) or type(e).__name__}
    return {"expression": expression, "result": round(result, 6) if isinstance(result, float) else result}
