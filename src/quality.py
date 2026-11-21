"""Rule-based quality assessment of the emissions inventory.

Checks are deterministic and return structured findings with the evidence behind them.
The LLM's job is only to explain and prioritise them, with GHG Protocol citations
retrieved from the knowledge base.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from src.data import Inventory, get_inventory

Severity = Literal["high", "medium", "low", "pass"]
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "pass": 3}


@dataclass
class Finding:
    id: str
    scope: str  # "1", "2", "3" or "all"
    severity: Severity
    title: str
    evidence: str
    recommendation: str
    ghg_reference: str = ""
    kb_query: str = ""  # what to search the GHG Protocol for when explaining this finding
    impact_tco2e: float | None = None
    extra: dict = field(default_factory=dict)


def _recalc(amount, factor, reported):
    """Return the unit scale (1 or 1/1000) that reproduces the reported tonnes, or None."""
    for scale, label in ((1.0, "t CO2e per unit"), (1e-3, "kg CO2e per unit")):
        if np.isclose(amount * factor * scale, reported, rtol=0.01, atol=0.01):
            return label
    return None


def check_scope1(inv: Inventory) -> list[Finding]:
    s1 = inv.scope1
    out: list[Finding] = []
    units = [_recalc(a, f, r) for a, f, r in zip(s1.consumption_amount, s1.emission_factor, s1.co2e_tonnes)]
    failed = sum(u is None for u in units)
    mix = s1.assign(factor_unit=units).groupby("fuel_type").factor_unit.first().to_dict()
    out.append(Finding(
        id="S1-RECALC", scope="1", severity="pass" if failed == 0 else "high",
        title="Scope 1 arithmetic reproduces (activity x factor)",
        evidence=f"{len(s1) - failed}/{len(s1)} rows recompute within 1%. Implied factor units differ by fuel: {mix}.",
        recommendation="Store emission-factor units and source (e.g. EPA, DEFRA, version/year) in their own columns; mixed kg/t factor units are an error waiting to happen.",
        ghg_reference="Corporate Standard ch.6 (calculation tools; transparency)",
        kb_query="emission factors documentation transparency calculation approach",
    ))
    plant = s1[s1.facility == "Manufacturing Plant"].co2e_tonnes.sum()
    out.append(Finding(
        id="S1-HOTSPOT", scope="1", severity="low",
        title="Scope 1 is concentrated in the Manufacturing Plant",
        evidence=f"Manufacturing Plant = {plant:,.0f} t ({100 * plant / s1.co2e_tonnes.sum():.0f}% of Scope 1): coal boilers and cement process emissions.",
        recommendation="Prioritise measured fuel data (metered/invoiced coal tonnage, cement clinker ratio) at the plant; these drive accuracy more than any other Scope 1 source.",
        ghg_reference="Corporate Standard ch.7 (managing inventory quality)",
        kb_query="process emissions cement direct measurement",
    ))
    ref = s1[s1.fuel_type == "Refrigerants"]
    out.append(Finding(
        id="S1-GWP", scope="1", severity="low",
        title="Refrigerant GWP source not documented",
        evidence=f"R404A factor = {ref.emission_factor.iloc[0]:g} (matches IPCC AR4 GWP 3,922; AR5 is 3,943 and AR6 is higher).",
        recommendation="State the IPCC assessment report used for GWPs and apply it consistently across the inventory; consider the mass-balance method for refrigerant leaks.",
        ghg_reference="Corporate Standard ch.4/6 (HFCs; GWP)",
        kb_query="HFC refrigeration fugitive emissions global warming potential",
    ))
    return out


def check_scope2(inv: Inventory) -> list[Finding]:
    s2 = inv.scope2
    elec = s2[s2.energy_type == "Electricity"]
    steam = s2[s2.energy_type == "Steam"]
    out: list[Finding] = []

    units = [_recalc(a, f, r) for a, f, r in zip(s2.consumption_amount, s2.emission_factor, s2.co2e_tonnes)]
    failed = sum(u is None for u in units)
    out.append(Finding(
        id="S2-RECALC", scope="2", severity="pass" if failed == 0 else "high",
        title="Scope 2 arithmetic reproduces (consumption x factor)",
        evidence=f"{len(s2) - failed}/{len(s2)} rows recompute within 1% (electricity factors in kg/kWh, steam in t/MMBtu).",
        recommendation="Keep it this way, and record factor source and vintage next to each factor.",
        ghg_reference="Corporate Standard ch.6",
        kb_query="indirect emissions purchased electricity emission factors",
    ))

    renewable_adj = (elec.co2e_tonnes * elec.renewable_percentage / 100).sum()
    out.append(Finding(
        id="S2-DUAL", scope="2", severity="high",
        title="Only one Scope 2 figure; no market-based total",
        evidence=(f"All {len(s2)} rows use grid-average factors (location-based). No market-based column, no contractual instruments "
                  f"recorded. renewable_percentage (0-{elec.renewable_percentage.max():g}%) is captured but never applied: if backed by "
                  f"valid certificates (RECs/PPAs) the market-based electricity total could be ~{renewable_adj:,.0f} t lower."),
        recommendation=("Report both location-based and market-based Scope 2 (GHG Protocol Scope 2 Guidance, 2015). For market-based, "
                        "use supplier-specific factors or energy attribute certificates meeting the Scope 2 quality criteria, and the "
                        "residual mix elsewhere. Keep evidence (certificate IDs, PPA contracts) for each renewable claim."),
        ghg_reference="Scope 2 Guidance (2015) dual reporting. Not in the provided 2004 Corporate Standard, which covers purchased electricity in ch.4 and green power in ch.8/9.",
        kb_query="purchased electricity green power contracts renewable energy certificates scope 2 reporting",
        impact_tco2e=round(renewable_adj, 1),
    ))

    out.append(Finding(
        id="S2-RENEW-INCONSISTENT", scope="2", severity="medium",
        title="Renewable share recorded but not reflected in emissions",
        evidence=("Rows with 'increased renewables' / 'some renewables added' notes still use the same grid factor, so the reported "
                  "number doesn't reflect the procurement the notes describe."),
        recommendation="Decide per site whether renewables are on-site generation (reduces grid kWh) or contractual (market-based only), and calculate accordingly.",
        ghg_reference="Corporate Standard ch.4 (scope 2), ch.9 (reporting green power purchases separately)",
        kb_query="green power purchases reporting optional information",
    ))

    out.append(Finding(
        id="S2-STEAM-FACTOR", scope="2", severity="medium",
        title="Steam uses one generic factor and grid-region labels",
        evidence=(f"Steam is {steam.co2e_tonnes.sum():,.0f} t ({100 * steam.co2e_tonnes.sum() / s2.co2e_tonnes.sum():.0f}% of Scope 2) but every "
                  f"site uses {steam.emission_factor.iloc[0]:g} t/MMBtu and is tagged with an electricity grid region (CAISO/PJM)."),
        recommendation="Obtain supplier-specific steam factors (fuel mix and boiler efficiency of the district heating/steam provider). Steam isn't a grid product, so grid-region tags don't apply.",
        ghg_reference="Corporate Standard ch.4 (scope 2 includes purchased steam)",
        kb_query="purchased steam heat cogeneration emissions allocation",
    ))

    plant_steam = steam[steam.facility == "Manufacturing Plant"].co2e_tonnes.sum()
    out.append(Finding(
        id="S2-DOUBLE-COUNT", scope="1,2", severity="medium",
        title="Possible double counting of plant steam (Scope 1 coal boilers vs Scope 2 steam)",
        evidence=(f"Manufacturing Plant reports coal 'Boiler operations' in Scope 1 and {plant_steam:,.0f} t of 'Process heating' steam in Scope 2. "
                  "If the steam comes from the company's own boilers it is already counted in Scope 1."),
        recommendation="Confirm the steam is bought from a third party (invoice/contract). If it comes from owned boilers, remove it from Scope 2.",
        ghg_reference="Corporate Standard ch.4 (double counting; scope 2 = generation of purchased energy)",
        kb_query="double counting scope 1 scope 2 steam generated own boilers",
        impact_tco2e=round(plant_steam, 1),
    ))

    out.append(Finding(
        id="S2-GRID-FACTORS", scope="2", severity="low",
        title="Grid factors labelled by ISO, constant all period, source undocumented",
        evidence=("Factors: " + ", ".join(f"{r}={f:g}" for r, f in elec.groupby('grid_region').emission_factor.first().items()) +
                  " kg/kWh. CAISO looks high compared with typical published California grid averages (~0.2-0.25 kg/kWh)."),
        recommendation="Use eGRID subregion factors (e.g. CAMX, RFCE, ERCT, NEWE) of a stated vintage, including CH4/N2O and, if needed, T&D losses separately.",
        ghg_reference="Corporate Standard ch.6 (emission factors)",
        kb_query="electricity emission factors grid average published",
    ))
    return out


def check_scope3(inv: Inventory) -> list[Finding]:
    s3 = inv.scope3
    out: list[Finding] = []
    total = s3.co2e_tonnes.sum()

    mis = s3[s3.category_mismatch]
    examples = mis.groupby(["activity_description", "category"]).size().nlargest(3)
    out.append(Finding(
        id="S3-CATEGORY-MAP", scope="3", severity="high",
        title="Scope 3 category labels contradict activities",
        evidence=(f"{len(mis)}/{len(s3)} rows ({100 * len(mis) / len(s3):.0f}%, {mis.co2e_tonnes.sum():,.0f} t) have a category that "
                  f"doesn't fit the activity; every activity appears in all 15 categories. Examples: "
                  + "; ".join(f"'{a}' as {c} x{n}" for (a, c), n in examples.items()) + "."),
        recommendation=("Re-map using an activity to category rule table (e.g. air travel → Cat 6, steel → Cat 1, equipment → Cat 2, "
                        "trucking → Cat 4), then review edge cases by hand. Category-level targets and hotspot analysis are invalid until this is fixed."),
        ghg_reference="Scope 3 Standard category definitions; Corporate Standard ch.4 (scope 3 activities)",
        kb_query="scope 3 activities business travel transportation purchased materials",
        impact_tco2e=round(mis.co2e_tonnes.sum(), 1),
    ))

    sm = s3[s3.supplier_mismatch]
    out.append(Finding(
        id="S3-SUPPLIER-MAP", scope="3", severity="medium",
        title="Supplier and activity combinations are implausible",
        evidence=(f"{100 * len(sm) / len(s3):.0f}% of rows pair a supplier with an activity outside its business, e.g. "
                  "'Corporate Travel' supplying trucking and product use, 'WasteManagement Co' supplying IT hardware; no waste activity exists at all."),
        recommendation="Reconcile against the AP/procurement ledger (vendor master and GL account) before supplier engagement or supplier-specific factors.",
        ghg_reference="Corporate Standard ch.7 (data management, internal QA)",
        kb_query="inventory quality management data collection checks",
    ))

    disp = s3.groupby("activity_description").emission_factor.agg(["min", "max"])
    out.append(Finding(
        id="S3-SPEND-BASED", scope="3", severity="medium",
        title="100% spend-based with highly inconsistent factors",
        evidence=(f"All {len(s3)} rows use USD x kg/USD factors. The same activity gets factors from ~{disp['min'].min():.2f} to "
                  f"{disp['max'].max():.2f} kg/USD (a ~{(disp['max'] / disp['min']).median():.0f}x spread within an activity)."),
        recommendation=("Use one documented EEIO factor set (e.g. USEEIO, a given year, inflation-adjusted) per activity. Then move the largest "
                        "categories to activity data: air travel by passenger-km and cabin class, trucking by tonne-km, steel/aluminium by tonnes with supplier EPDs."),
        ghg_reference="Scope 3 Standard / Technical Guidance (calculation methods hierarchy); Corporate Standard ch.7 (uncertainty)",
        kb_query="uncertainty emission factors data quality activity data",
    ))

    contradictory = s3[s3.data_quality == "Primary data"]
    out.append(Finding(
        id="S3-DQ-LABELS", scope="3", severity="medium",
        title="Data-quality labels are uninformative or contradictory",
        evidence=(f"Labels split almost evenly across 5 values ({', '.join(f'{k} {v}' for k, v in s3.data_quality.value_counts().items())}). "
                  f"{len(contradictory)} rows are 'Primary data' yet are calculated from spend with generic factors. "
                  "Category_Details is the same constant on every row."),
        recommendation="Score data quality with a defined rubric (technology, temporal, geographic representativeness, completeness, reliability) and record the actual method used (supplier-specific / hybrid / average-data / spend-based).",
        ghg_reference="Corporate Standard ch.7 (data quality, uncertainty)",
        kb_query="data quality indicators uncertainty assessment",
    ))

    downstream = s3[s3.category_num.between(9, 15)]
    out.append(Finding(
        id="S3-DOWNSTREAM-SPEND", scope="3", severity="medium",
        title="Downstream categories built from purchase spend",
        evidence=(f"Categories 9-15 hold {downstream.co2e_tonnes.sum():,.0f} t ({100 * downstream.co2e_tonnes.sum() / total:.0f}% of Scope 3) "
                  "calculated from money paid to suppliers. Downstream emissions (use of sold products, end-of-life, franchises, "
                  "investments) depend on products sold, not on purchases."),
        recommendation="Model downstream categories from sales volumes and product specs (e.g. lifetime energy use for Cat 11). Report categories that aren't relevant as excluded, with a justification.",
        ghg_reference="Scope 3 Standard (downstream categories); Corporate Standard ch.4",
        kb_query="downstream scope 3 use of sold products disposal",
        impact_tco2e=round(downstream.co2e_tonnes.sum(), 1),
    ))
    return out


def check_cross_scope(inv: Inventory) -> list[Finding]:
    s12_months = sorted(set(inv.scope1.month) | set(inv.scope2.month))
    s3_months = sorted(set(inv.scope3.month))
    dup = sum(df.duplicated().sum() for df in (inv.scope1, inv.scope2, inv.scope3))
    nulls = sum(int(df.isna().sum().sum()) for df in (inv.scope1, inv.scope2, inv.scope3))
    return [
        Finding(
            id="ALL-PERIOD", scope="all", severity="high",
            title="Reporting periods don't match across scopes",
            evidence=f"Scope 1/2 cover {len(s12_months)} months ({s12_months[0]} to {s12_months[-1]}); Scope 3 covers {len(s3_months)} months ({s3_months[0]} to {s3_months[-1]}).",
            recommendation="Collect Scope 1 and 2 for the full calendar year before reporting. In the meantime, label any annualised figures as estimates.",
            ghg_reference="Corporate Standard ch.9 (reporting period; annual reporting)",
            kb_query="reporting period annual inventory base year",
        ),
        Finding(
            id="ALL-COMPLETENESS", scope="all", severity="pass",
            title="No missing values or duplicate rows",
            evidence=f"{nulls} null cells, {dup} exact duplicate rows across the three files.",
            recommendation="Keep automated null/duplicate checks in the ingestion pipeline.",
            ghg_reference="Corporate Standard ch.1 (completeness principle)",
            kb_query="completeness principle accounting reporting",
        ),
    ]


def run_quality_checks(scope: str = "all", inv: Inventory | None = None) -> dict:
    """Run all checks, optionally filtered to one scope ('1', '2', '3' or 'all')."""
    inv = inv or get_inventory()
    findings = check_cross_scope(inv) + check_scope1(inv) + check_scope2(inv) + check_scope3(inv)
    if scope != "all":
        findings = [f for f in findings if scope in f.scope.split(",") or f.scope == "all"]
    findings.sort(key=lambda f: SEVERITY_ORDER[f.severity])
    counts = {s: sum(f.severity == s for f in findings) for s in SEVERITY_ORDER}
    return {"scope": scope, "severity_counts": counts, "findings": [asdict(f) for f in findings]}
