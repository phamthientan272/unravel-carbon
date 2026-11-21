"""Rule-based quality assessment of a GHG inventory (Scope 1, 2 and 3 tables).

Design
------
- **Generic rules, no company-specific names.** Rules find a company's own facilities, activities and
  columns at run time. Nothing here mentions a particular facility, fuel, supplier or grid region.
- **A column spec is the only adapter.** A `QualitySpec` says which column plays which role (amount, factor,
  reported tonnes, site, source, ...). A new company needs a new spec (the LLM can propose one, a person confirms).
  Rules whose input column is missing are skipped, not guessed.
- **Rules establish facts, the document supplies the standard.** Every finding's evidence is computed from the
  data. The wording of what the GHG Protocol says is NOT stored here: the provided document is the only authority and
  it is searched when a question is asked (src/grounding.py). `quality_catalog.yaml` holds each rule's layer, the
  principle it serves, a neutral statement of the requirement and the PDF pages where the document states it.
- **The LLM never invents findings.** It explains and prioritises them, citing only retrieved passages.

Three layers, so a reader knows how much weight a finding carries (see the catalog header):
- `requirement`: the document says "shall" or defines a term the data must respect. A gap here is a compliance gap.
- `qa`: a chapter 7 quality check, or a direct consequence of a principle.
- `investigation`: our own heuristic or statistical test. Never a compliance claim.

The result also gives a verdict per principle (chapter 1) and a list of requirements the data cannot show.
The document warns that uncertainty is not an objective quality measure (PDF p.58), so there is no single score.

Every finding also states its `dimension` (what kind of quality problem) and `basis` (how it was established:
`data`, `statistical` or `heuristic`).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

from src.data import Inventory, get_inventory

Severity = Literal["high", "medium", "low", "pass"]
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "pass": 3}
CATALOG_PATH = Path(__file__).with_name("quality_catalog.yaml")


# --- column spec: the only place that knows a company's column names ----------------------------

@dataclass(frozen=True)
class TableSpec:
    """Which column plays which role in one scope's table. Optional roles can be None."""
    amount: str                      # activity amount (consumption, or spend)
    factor: str                      # emission factor
    reported: str = "co2e_tonnes"    # reported tonnes CO2e
    date: str = "date"
    entity: str | None = None        # who emits: facility (Scope 1/2) or supplier (Scope 3)
    site: str | None = None          # physical site (cross-site and overlap rules)
    source: str | None = None        # what is burned / bought / purchased (fuel, energy type, activity)
    kind: str | None = None          # broader activity type, e.g. stationary vs mobile combustion
    region: str | None = None        # grid or market region label
    renewable_pct: str | None = None # recorded renewable share
    factor_unit: str | None = None   # unit of the emission factor, if recorded
    factor_origin: str | None = None # source / year of the emission factor, if recorded
    method: Literal["activity", "spend"] = "activity"
    # Scope 3 only
    category: str | None = None
    supplier: str | None = None
    quality_label: str | None = None


@dataclass(frozen=True)
class QualitySpec:
    scope1: TableSpec
    scope2: TableSpec
    scope3: TableSpec


# Spec for the sample inventory in data/. A new company supplies its own.
DEFAULT_SPEC = QualitySpec(
    scope1=TableSpec(amount="consumption_amount", factor="emission_factor", entity="facility", site="facility",
                     source="fuel_type", kind="activity_type"),
    scope2=TableSpec(amount="consumption_amount", factor="emission_factor", entity="facility", site="facility",
                     source="energy_type", region="grid_region", renewable_pct="renewable_percentage"),
    scope3=TableSpec(amount="spend_amount", factor="emission_factor", entity="supplier", source="activity_description",
                     method="spend", category="category", supplier="supplier", quality_label="data_quality"),
)


# --- findings and catalog -----------------------------------------------------------------------

@dataclass
class Finding:
    id: str
    scope: str  # "1", "2", "3", "1,2" or "all"
    severity: Severity
    title: str
    evidence: str
    fallback_query: str = ""  # used to search the document only if the LLM cannot write a query
    dimension: str = ""  # consistency | completeness | plausibility | traceability | insight
    basis: str = ""      # data | statistical | heuristic
    layer: str = ""      # requirement | qa | investigation
    principle: str = ""  # relevance | completeness | consistency | transparency | accuracy
    requirement: str = ""  # what the document asks for (empty for investigations)
    ref_pages: list[int] = field(default_factory=list)  # PDF pages where the document states it
    headline: str = ""   # first sentence of the evidence, for compact summaries
    impact_tco2e: float | None = None
    extra: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _catalog() -> dict:
    return yaml.safe_load(CATALOG_PATH.read_text())


def _finding(rule: str, fid: str, scopes: str, severity: Severity, evidence: str,
             impact: float | None = None, **fmt) -> Finding:
    """Build a finding: computed evidence + curated advice from the catalog."""
    c = _catalog()[rule]
    first = re.split(r"\.\s+(?=[A-Z])", evidence, maxsplit=1)[0].rstrip(".")  # first sentence
    return Finding(
        id=fid, scope=scopes, severity=severity, title=c["title"].format(**fmt), evidence=evidence,
        fallback_query=c["fallback_query"], dimension=c["dimension"], basis=c["basis"], headline=first[:300],
        layer=c["layer"], principle=c["principle"], requirement=c["requirement"], ref_pages=list(c["pages"]),
        impact_tco2e=None if impact is None else round(float(impact), 1),
    )


# --- small helpers ------------------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str | None) -> pd.Series | None:
    return df[name] if name and name in df.columns else None


def _months(df: pd.DataFrame, spec: TableSpec) -> list[str]:
    return sorted(pd.to_datetime(df[spec.date]).dt.to_period("M").astype(str).unique())


def _implied_unit(amount: float, factor: float, reported: float) -> str | None:
    """Which unit of the factor reproduces the reported tonnes: t per unit, kg per unit, or neither."""
    for scale, label in ((1.0, "t CO2e per unit"), (1e-3, "kg CO2e per unit")):
        if np.isclose(amount * factor * scale, reported, rtol=0.01, atol=0.01):
            return label
    return None


def _implied_units(df: pd.DataFrame, spec: TableSpec) -> pd.Series:
    return pd.Series([_implied_unit(a, f, r) for a, f, r in zip(df[spec.amount], df[spec.factor], df[spec.reported])],
                     index=df.index)


def association(a: pd.Series, b: pd.Series, perms: int = 1000, seed: int = 0) -> dict:
    """Does knowing `a` tell you anything about `b`? Cramer's V corrected for chance, plus a permutation p-value.

    Raw Cramer's V is inflated for many categories and few rows, so the chance level (mean V over shuffled
    labels) is subtracted: `excess_v` is 0 for unrelated columns and 1 for a perfect link. Needs no mapping.
    """
    x, y = pd.factorize(a)[0], pd.factorize(b)[0]
    r, c, n = x.max() + 1, y.max() + 1, len(x)
    if r < 2 or c < 2:
        return {"excess_v": 0.0, "p": 1.0, "informative": False}

    def chi2(xx, yy):
        t = np.bincount(xx * c + yy, minlength=r * c).reshape(r, c).astype(float)
        e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / n
        m = e > 0
        return float(((t - e) ** 2 / np.where(m, e, 1))[m].sum())

    obs = chi2(x, y)
    rng = np.random.default_rng(seed)
    null = np.array([chi2(x, rng.permutation(y)) for _ in range(perms)])
    k = min(r - 1, c - 1)
    v = lambda s: float(np.sqrt(s / (n * k)))
    chance = v(null.mean())
    excess = (v(obs) - chance) / (1 - chance) if chance < 1 else 0.0
    p = (1 + int((null >= obs).sum())) / (perms + 1)
    return {"excess_v": round(max(excess, 0.0), 3), "p": round(p, 3), "informative": bool(p < 0.05 and excess >= 0.3)}


# --- rules that run on any single scope table ---------------------------------------------------

def _recalc(n: int, df: pd.DataFrame, spec: TableSpec) -> Finding:
    units = _implied_units(df, spec)
    failed = int(units.isna().sum())
    src = _col(df, spec.source)
    by_source = units.groupby(src).agg(lambda u: u.dropna().iloc[0] if u.notna().any() else "none").to_dict() if src is not None else {}
    mixed = len(set(by_source.values())) > 1
    ev = (f"{len(df) - failed}/{len(df)} rows reproduce the reported tonnes within 1%."
          + (f" Implied factor unit by source: {by_source}." if by_source else "")
          + (" The unit differs between sources." if mixed else ""))
    return _finding("RECALC", f"S{n}-RECALC", str(n), "pass" if failed == 0 else "high", ev, scope=n)


def _single_factor(n: int, df: pd.DataFrame, spec: TableSpec) -> Finding | None:
    site, src = _col(df, spec.site), _col(df, spec.source)
    if site is None or src is None:
        return None
    total = df[spec.reported].sum()
    hits = []
    for source, g in df.groupby(src):
        share = g[spec.reported].sum() / total
        if share >= 0.20 and site[g.index].nunique() > 1 and g[spec.factor].nunique() == 1:
            hits.append((source, g, share))
    if not hits:
        return None
    parts = []
    for source, g, share in hits:
        region = _col(g, spec.region)
        parts.append(f"{source} is {g[spec.reported].sum():,.0f} t ({100 * share:.0f}% of Scope {n}) but all {site[g.index].nunique()} sites use "
                     f"the same factor ({g[spec.factor].iloc[0]:g})"
                     + (f" and carry region labels ({', '.join(sorted(region.unique()))}) that do not change it" if region is not None and region.nunique() > 1 else ""))
    impact = sum(g[spec.reported].sum() for _, g, _ in hits)
    return _finding("SINGLE_FACTOR", f"S{n}-SINGLE-FACTOR", str(n), "medium", ". ".join(parts) + ".", impact)


def _concentration(n: int, df: pd.DataFrame, spec: TableSpec) -> Finding | None:
    ent = _col(df, spec.entity)
    if ent is None:
        return None
    by = df.groupby(ent)[spec.reported].sum().sort_values(ascending=False)
    share = by.iloc[0] / by.sum()
    if share < 0.50:
        return None
    top = by.index[0]
    src = _col(df, spec.source)
    drivers = ""
    if src is not None:
        s = df[ent == top].groupby(src)[spec.reported].sum().nlargest(2)
        drivers = " Largest sources there: " + ", ".join(f"{k} ({v:,.0f} t)" for k, v in s.items()) + "."
    ev = f"{top} = {by.iloc[0]:,.0f} t ({100 * share:.0f}% of Scope {n}).{drivers}"
    return _finding("CONCENTRATION", f"S{n}-CONCENTRATION", str(n), "low", ev, by.iloc[0], scope=n)


def _attribute_ignored(n: int, df: pd.DataFrame, spec: TableSpec) -> Finding | None:
    pct, site, src = _col(df, spec.renewable_pct), _col(df, spec.site), _col(df, spec.source)
    if pct is None or site is None or src is None:
        return None
    groups = []
    for (s, e), g in df.groupby([site, src]):
        if g[spec.renewable_pct].nunique() > 1 and g[spec.factor].nunique() == 1:
            groups.append(f"{s}/{e}: {g[spec.renewable_pct].min():g}% to {g[spec.renewable_pct].max():g}%")
    if not groups:
        return None
    ev = (f"In {len(groups)} site/source groups the recorded renewable share changes over time ({'; '.join(groups)}) "
          "but the emission factor never changes, so the reported tonnes do not reflect it.")
    return _finding("FACTOR_IGNORES_ATTRIBUTE", "S2-RENEW-INCONSISTENT", str(n), "medium", ev)


# --- Scope 2 ------------------------------------------------------------------------------------

def _renewable_unsupported(df: pd.DataFrame, spec: TableSpec) -> Finding | None:
    pct = _col(df, spec.renewable_pct)
    if pct is None or (pct > 0).sum() == 0:
        return None
    if any(re.search(r"certificate|\brec\b|ppa|contract|instrument|supplier", c, re.I) for c in df.columns):
        return None  # some instrument information is recorded; do not claim it is missing
    impact = float((df[spec.reported] * pct / 100).sum())
    ev = (f"A renewable share is recorded on {int((pct > 0).sum())} of {len(df)} rows (up to {pct.max():g}%), but no column records the "
          f"certificate, contract or other instrument behind it. Applying the recorded share to the reported tonnes would lower them by about {impact:,.0f} t.")
    return _finding("RENEW_UNSUPPORTED", "S2-RENEW-UNSUPPORTED", "2", "medium", ev, impact)


# --- Scope 3 ------------------------------------------------------------------------------------

def _scope3_checks(df: pd.DataFrame, spec: TableSpec) -> list[Finding]:
    out: list[Finding] = []
    total = df[spec.reported].sum()
    act, cat, sup = _col(df, spec.source), _col(df, spec.category), _col(df, spec.supplier)

    if cat is not None and act is not None:
        cat_num = cat.astype(str).str.extract(r"(\d+)")[0].astype(float)
        a = association(act, cat)
        spread = df.groupby(act).apply(lambda g: cat_num[g.index].nunique())
        ev = (f"Knowing the activity tells you {'something' if a['informative'] else 'almost nothing'} about the category label "
              f"(chance-corrected association {a['excess_v']:.2f} on a 0-1 scale, permutation p = {a['p']:.2f}). On average each activity "
              f"appears in {spread.mean():.0f} of {cat_num.nunique()} categories.")
        out.append(_finding("LABEL_NO_INFO", "S3-CATEGORY-MAP", "3", "pass" if a["informative"] else "high", ev,
                            None if a["informative"] else total))

    if sup is not None and act is not None:
        a = association(act, sup)
        ev = (f"Knowing the activity tells you {'something' if a['informative'] else 'little'} about the supplier (chance-corrected "
              f"association {a['excess_v']:.2f}, permutation p = {a['p']:.2f}). Each supplier appears with "
              f"{df.groupby(sup)[spec.source].nunique().mean():.0f} of {act.nunique()} activities on average.")
        out.append(_finding("SUPPLIER_NO_INFO", "S3-SUPPLIER-MAP", "3", "pass" if a["informative"] else "medium", ev))

    if spec.method == "spend":
        out.append(_finding("SPEND_ONLY", "S3-SPEND-BASED", "3", "medium",
                            f"All {len(df)} rows are calculated as the money spent multiplied by an emission factor. No row uses a physical quantity (units, distance, mass)."))

    if act is not None and act.nunique() >= 3:
        groups = df.groupby(act)[spec.factor]
        ss_between = float((groups.transform("mean").sub(df[spec.factor].mean()) ** 2).sum())
        eta2 = ss_between / float(((df[spec.factor] - df[spec.factor].mean()) ** 2).sum())
        disp = groups.agg(["min", "max", "median"])
        corr = df[spec.amount].corr(df[spec.factor])
        if eta2 < 0.10:
            ev = (f"Activity explains only {100 * eta2:.0f}% of the variation in the factor. Medians are similar for every activity "
                  f"({disp['median'].min():.1f} to {disp['median'].max():.1f}), the largest factor is {(disp['max'] / disp['min']).median():.0f}x the smallest "
                  f"within an activity (median), and the factor is uncorrelated with the amount (r = {corr:.2f}). The factor does not depend on the activity or the amount.")
            out.append(_finding("FACTOR_NOT_DISCRIMINATING", "S3-FACTORS", "3", "medium", ev, total))

    ql = _col(df, spec.quality_label)
    if ql is not None:
        counts = ql.value_counts()
        even = len(counts) > 2 and counts.max() / counts.min() < 1.5
        primary = ql.astype(str).str.contains(r"primary|measured|supplier[- ]specific", case=False, regex=True)
        contradiction = int(primary.sum()) if spec.method == "spend" else 0
        if even or contradiction:
            ev = (f"Labels are split {'almost evenly' if even else 'unevenly'} across {len(counts)} values "
                  f"({', '.join(f'{k} {v}' for k, v in counts.items())}).")
            if contradiction:
                ev += f" {contradiction} rows are labelled as primary or supplier-specific data although every row is calculated as spend x factor."
            out.append(_finding("DQ_LABELS", "S3-DQ-LABELS", "3", "medium", ev))

    return out


# --- cross-scope --------------------------------------------------------------------------------

def check_cross_scope(inv: Inventory, spec: QualitySpec = DEFAULT_SPEC) -> list[Finding]:
    tables = {1: (inv.scope1, spec.scope1), 2: (inv.scope2, spec.scope2), 3: (inv.scope3, spec.scope3)}
    out: list[Finding] = []

    months = {n: _months(df, s) for n, (df, s) in tables.items()}
    spans = {n: f"{m[0]} to {m[-1]} ({len(m)} months)" for n, m in months.items()}
    same = len({len(m) for m in months.values()}) == 1
    out.append(_finding("PERIOD", "ALL-PERIOD", "all", "pass" if same else "high",
                        "Scope 1: " + spans[1] + "; Scope 2: " + spans[2] + "; Scope 3: " + spans[3] + "."
                        + ("" if same else " Totals across scopes mix different periods.")))

    nulls = sum(int(df.isna().sum().sum()) for df, _ in tables.values())
    dups = sum(int(df.duplicated().sum()) for df, _ in tables.values())
    const = {f"scope{n}": [c for c in df.columns if df[c].nunique() <= 1 and len(df) > 1] for n, (df, _) in tables.items()}
    const = {k: v for k, v in const.items() if v}
    ev = f"{nulls} null cells and {dups} exact duplicate rows across the three files."
    if const:
        ev += f" Constant columns that carry no information: {const}."
    out.append(_finding("COMPLETENESS", "ALL-COMPLETENESS", "all", "pass" if not (nulls or dups or const) else ("low" if not (nulls or dups) else "medium"), ev))

    missing_unit = [f"scope{n}" for n, (df, s) in tables.items() if _col(df, s.factor_unit) is None]
    missing_origin = [f"scope{n}" for n, (df, s) in tables.items() if _col(df, s.factor_origin) is None]
    mixed = {}
    for n, (df, s) in tables.items():
        src = _col(df, s.source)
        if src is not None:
            u = _implied_units(df, s).groupby(src).agg(lambda x: x.dropna().iloc[0] if x.notna().any() else "none")
            if u.nunique() > 1:
                mixed[f"scope{n}"] = u.to_dict()
    if missing_unit or missing_origin or mixed:
        ev = (f"No factor unit column in: {missing_unit or 'none'}. No factor source or year column in: {missing_origin or 'none'}."
              + (f" Implied units differ between sources within a table: {mixed}." if mixed else ""))
        out.append(_finding("FACTOR_META", "ALL-FACTOR-META", "all", "medium", ev))

    # Fuel burned on site (Scope 1) and heat/steam bought at the same site (Scope 2): possible double count
    (s1, p1), (s2, p2) = tables[1], tables[2]
    site1, site2, src2 = _col(s1, p1.site), _col(s2, p2.site), _col(s2, p2.source)
    if site1 is not None and site2 is not None and src2 is not None:
        heat = s2[src2.astype(str).str.contains(r"steam|heat|cool|district", case=False, regex=True)]
        kind = _col(s1, p1.kind)
        stationary = s1[kind.astype(str).str.contains(r"stationary|combust|boiler", case=False, regex=True)] if kind is not None else s1
        shared = sorted(set(heat[p2.site]) & set(stationary[p1.site]))
        if shared:
            parts, impact = [], 0.0
            for site in shared:
                t = heat[heat[p2.site] == site][p2.reported].sum()
                impact += t
                fuels = sorted(stationary[stationary[p1.site] == site][p1.source].astype(str).unique()) if p1.source else []
                parts.append(f"{site}: Scope 1 burns {', '.join(fuels) or 'fuel'}; Scope 2 buys heat/steam ({t:,.0f} t)")
            ev = ("; ".join(parts) + ". The data does not say where the heat or steam comes from, so it cannot show whether the same emissions appear in both scopes. "
                  "This is a question to confirm with the client, not a confirmed error.")
            out.append(_finding("HEAT_OVERLAP", "S2-DOUBLE-COUNT", "1,2", "medium", ev, impact))
    return out


# --- requirement layer: items the document says "shall" ------------------------------------------

GAS_COLUMN = re.compile(r"(^|[^a-z])(co2|ch4|n2o|hfcs?|pfcs?|sf6)([^a-z0-9e]|$)", re.I)  # "co2e" is not a gas column
BIOGENIC_SOURCE = re.compile(r"biomass|biofuel|biodiesel|bioethanol|wood|pellet|bagasse|biogas", re.I)
NON_ELECTRICITY = re.compile(r"steam|heat|cool|district|chilled", re.I)
ELECTRICITY = re.compile(r"electric|power", re.I)


def check_requirements(inv: Inventory, spec: QualitySpec = DEFAULT_SPEC) -> list[Finding]:
    out: list[Finding] = []
    tables = {1: (inv.scope1, spec.scope1), 2: (inv.scope2, spec.scope2), 3: (inv.scope3, spec.scope3)}

    # Scopes 1 and 2 are kept apart: each has its own table with its own reported tonnes.
    apart = all(s.reported in df.columns and len(df) for df, s in (tables[1], tables[2]))
    out.append(_finding("REQ_SCOPES_APART", "REQ-SCOPES-SEPARATE", "1,2", "pass" if apart else "high",
                        "Scope 1 and Scope 2 are held in separate tables, each with its own reported tonnes." if apart
                        else "Scope 1 and Scope 2 are not both available as separate tables with reported tonnes."))

    # Six gases reported separately, in tonnes and in tonnes CO2e.
    gas_cols = {f"scope{n}": [c for c in df.columns if GAS_COLUMN.search(str(c))] for n, (df, _) in tables.items()}
    if not any(gas_cols.values()):
        ev = ("Every table gives one reported figure in tonnes CO2e. No column gives emissions per gas "
              "(CO2, CH4, N2O, HFCs, PFCs, SF6), so the breakdown by gas cannot be shown from this data. "
              "If the report holds it elsewhere, this is a gap in the data supplied, not necessarily in the report.")
        out.append(_finding("REQ_GAS_BREAKDOWN", "ALL-GAS-BREAKDOWN", "all", "medium", ev))
    else:
        found = sorted({c.lower() for cols in gas_cols.values() for c in cols})
        out.append(_finding("REQ_GAS_BREAKDOWN", "ALL-GAS-BREAKDOWN", "all", "pass",
                            f"Gas-level columns are present ({', '.join(found)}). Whether all six gases are covered is not checked."))

    # Biomass or biofuel burned must be reported separately from Scope 1.
    s1, p1 = tables[1]
    src1 = _col(s1, p1.source)
    if src1 is not None:
        bio = s1[src1.astype(str).str.contains(BIOGENIC_SOURCE)]
        if len(bio):
            names = ", ".join(sorted(src1[bio.index].astype(str).unique()))
            out.append(_finding("REQ_BIOGENIC", "S1-BIOGENIC", "1", "medium",
                                f"{len(bio)} Scope 1 rows ({bio[p1.reported].sum():,.0f} t) have a biomass or biofuel source ({names}). "
                                "The data does not show whether their CO2 is excluded from Scope 1 and reported separately.",
                                bio[p1.reported].sum()))
        else:
            out.append(_finding("REQ_BIOGENIC", "S1-BIOGENIC", "1", "pass",
                                "No biomass or biofuel source was found in the Scope 1 source column (keyword match)."))

    # Scope 2 is defined as purchased electricity.
    s2, p2 = tables[2]
    src2 = _col(s2, p2.source)
    if src2 is not None:
        other = s2[src2.astype(str).str.contains(NON_ELECTRICITY) & ~src2.astype(str).str.contains(ELECTRICITY)]
        if len(other):
            names = ", ".join(sorted(src2[other.index].astype(str).unique()))
            t = other[p2.reported].sum()
            out.append(_finding("REQ_SCOPE2_DEFINITION", "S2-DEFINITION", "2", "medium",
                                f"{len(other)} of {len(s2)} Scope 2 rows ({t:,.0f} t) are for energy other than electricity ({names}). "
                                "In the provided document Scope 2 covers purchased electricity only, so these rows need a scope decision.", t))
    return out


# What the data cannot show at all. Each item is a requirement of the document that no table column can evidence.
def cannot_assess(inv: Inventory, spec: QualitySpec = DEFAULT_SPEC) -> list[dict]:
    years = {int(y) for n, s in ((1, spec.scope1), (2, spec.scope2), (3, spec.scope3))
             for y in pd.to_datetime(getattr(inv, f"scope{n}")[s.date]).dt.year.unique()}
    items = [
        {"item": "Base year and base-year recalculation policy", "needs": "a base year, its reason and a recalculation policy", "pages": [37, 65]},
        {"item": "Organizational boundary and consolidation approach", "needs": "the chosen approach (equity share or control)", "pages": [65]},
        {"item": "Methodologies and exclusions disclosed", "needs": "a methodology statement and a list of excluded sources", "pages": [65]},
        {"item": "Scope 2 emission factor basis", "needs": "whether factors are at generation or at consumption", "pages": [89]},
        {"item": "Relevance to the users' decisions", "needs": "the purpose of the inventory and its users", "pages": [10]},
    ]
    if len(years) < 2:
        items.append({"item": "Year-on-year change in activity data", "pages": [55],
                      "needs": f"data for at least two years (only {sorted(years)} present)"})
    return items


PRINCIPLES = ("relevance", "completeness", "consistency", "transparency", "accuracy")


def principle_verdicts(findings: list[Finding]) -> dict:
    """One verdict per principle. Investigations alone never make a gap: they say 'review'.

    gap = a requirement or QA check failed (high or medium); review = only investigations or low-severity items;
    no_issues_found = checks ran and passed; not_assessable = no check in this module covers the principle.
    """
    out = {}
    for p in PRINCIPLES:
        fs = [f for f in findings if f.principle == p]
        failed = [f for f in fs if f.layer in ("requirement", "qa") and f.severity in ("high", "medium")]
        review = [f for f in fs if f.severity != "pass" and f not in failed]
        status = "gap" if failed else "review" if review else "no_issues_found" if fs else "not_assessable"
        out[p] = {"status": status, "gaps": [f.id for f in failed], "to_review": [f.id for f in review],
                  "checks_passed": [f.id for f in fs if f.severity == "pass"]}
    return out


def check_scope1(inv: Inventory, spec: QualitySpec = DEFAULT_SPEC) -> list[Finding]:
    return _table_checks(1, inv.scope1, spec.scope1)


def check_scope2(inv: Inventory, spec: QualitySpec = DEFAULT_SPEC) -> list[Finding]:
    out = _table_checks(2, inv.scope2, spec.scope2)
    attr = _attribute_ignored(2, inv.scope2, spec.scope2)
    method = _renewable_unsupported(inv.scope2, spec.scope2)
    return out + [f for f in (method, attr) if f]


def check_scope3(inv: Inventory, spec: QualitySpec = DEFAULT_SPEC) -> list[Finding]:
    return [_recalc(3, inv.scope3, spec.scope3)] + _scope3_checks(inv.scope3, spec.scope3)


def _table_checks(n: int, df: pd.DataFrame, spec: TableSpec) -> list[Finding]:
    found = [_recalc(n, df, spec), _single_factor(n, df, spec), _concentration(n, df, spec)]
    return [f for f in found if f]


def run_quality_checks(scope: str = "all", inv: Inventory | None = None, spec: QualitySpec | None = None) -> dict:
    """Run all checks, optionally filtered to one scope ('1', '2', '3' or 'all').

    Returns the findings (requirement, qa and investigation layers), a verdict per principle and the list of
    requirements the data cannot show.
    """
    inv, spec = inv or get_inventory(), spec or DEFAULT_SPEC
    findings = (check_requirements(inv, spec) + check_cross_scope(inv, spec) + check_scope1(inv, spec)
                + check_scope2(inv, spec) + check_scope3(inv, spec))
    if scope != "all":
        findings = [f for f in findings if scope in f.scope.split(",") or f.scope == "all"]
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], -(f.impact_tco2e or 0)))
    counts = {s: sum(f.severity == s for f in findings) for s in SEVERITY_ORDER}
    layers = {l: sum(f.layer == l and f.severity != "pass" for f in findings) for l in ("requirement", "qa", "investigation")}
    return {"scope": scope, "severity_counts": counts, "open_by_layer": layers,
            "principle_verdicts": principle_verdicts(findings), "cannot_assess": cannot_assess(inv, spec),
            "findings": [asdict(f) for f in findings]}
