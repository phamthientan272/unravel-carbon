"""Load and normalise the emissions inventory CSVs.

The CSVs remain the source of truth. They are loaded once into pandas DataFrames;
everything downstream (analytics tools, quality checks, SQL tool) reads from the
`Inventory` object built here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# GHG Protocol Scope 3 category names (Corporate Value Chain Standard).
SCOPE3_CATEGORIES = {
    1: "Purchased goods and services",
    2: "Capital goods",
    3: "Fuel- and energy-related activities",
    4: "Upstream transportation and distribution",
    5: "Waste generated in operations",
    6: "Business travel",
    7: "Employee commuting",
    8: "Upstream leased assets",
    9: "Downstream transportation and distribution",
    10: "Processing of sold products",
    11: "Use of sold products",
    12: "End-of-life treatment of sold products",
    13: "Downstream leased assets",
    14: "Franchises",
    15: "Investments",
}

# Activities a supplier could plausibly provide, used to detect supplier/activity mismatches.
EXPECTED_SUPPLIER_ACTIVITIES = {
    "SteelCorp Inc": {"Raw Materials - Steel"},
    "MetalSupply Co": {"Raw Materials - Steel", "Raw Materials - Aluminum"},
    "PackagePro Ltd": {"Packaging Materials"},
    "OfficeMax": {"Office Supplies", "IT Hardware"},
    "TechEquip Corp": {"IT Hardware", "Manufacturing Equipment"},
    "Corporate Travel": {"Business Travel - Air"},
    "LogisticsPro": {"Inbound Logistics - Truck"},
    "WasteManagement Co": set(),  # no waste activity exists in the data at all
}

LOW_QUALITY_LABELS = {"Proxy data", "Industry average", "Secondary data"}


def _snake(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.strip().lower())


@dataclass(frozen=True)
class Inventory:
    scope1: pd.DataFrame
    scope2: pd.DataFrame
    scope3: pd.DataFrame

    @property
    def combined(self) -> pd.DataFrame:
        """Long table across scopes with a common set of columns."""
        s1 = self.scope1.assign(
            scope=1, source=self.scope1["facility"], activity=self.scope1["activity_type"] + " - " + self.scope1["fuel_type"]
        )
        s2 = self.scope2.assign(
            scope=2, source=self.scope2["facility"], activity="Purchased " + self.scope2["energy_type"]
        )
        s3 = self.scope3.assign(scope=3, source=self.scope3["supplier"], activity=self.scope3["activity_description"])
        cols = ["scope", "source", "activity", "co2e_tonnes", "date"]
        return pd.concat([s1[cols], s2[cols], s3[cols]], ignore_index=True)


def load_inventory(data_dir: Path = DATA_DIR) -> Inventory:
    s1 = _snake(pd.read_csv(data_dir / "scope1.csv", parse_dates=["Date"]))
    s2 = _snake(pd.read_csv(data_dir / "scope2.csv", parse_dates=["Date"]))
    s3 = _snake(pd.read_csv(data_dir / "scope3.csv", parse_dates=["Date"]))

    for df in (s1, s2, s3):
        df["month"] = df["date"].dt.to_period("M").astype(str)
        df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    s3["category_num"] = s3["category"].str.extract(r"(\d+)").astype(int)
    s3["category_name"] = s3["category_num"].map(SCOPE3_CATEGORIES)
    # A supplier the mapping does not know is "unknown", not "wrong".
    s3["supplier_mismatch"] = [
        sup in EXPECTED_SUPPLIER_ACTIVITIES and act not in EXPECTED_SUPPLIER_ACTIVITIES[sup]
        for sup, act in zip(s3["supplier"], s3["activity_description"])
    ]
    # Spend-based factor is kg CO2e per USD: tonnes = spend * factor / 1000
    s3["intensity_kg_per_usd"] = s3["emission_factor"]
    return Inventory(scope1=s1, scope2=s2, scope3=s3)


@lru_cache(maxsize=1)
def get_inventory() -> Inventory:
    return load_inventory()


def data_card(inv: Inventory | None = None) -> str:
    """Compact description of the inventory that goes into the system prompt.

    Everything here is computed from the loaded data and from the quality findings, so it describes whichever
    company's inventory is loaded. The model gets the shape and caveats of the data, never the raw rows.
    """
    from src.quality import run_quality_checks  # local import: quality imports this module

    inv = inv or get_inventory()

    def describe(n: int, df: pd.DataFrame) -> str:
        ent = next((c for c in ("facility", "supplier") if c in df.columns), None)
        names = f" Distinct {ent} values: {', '.join(sorted(map(str, df[ent].unique()))[:12])}." if ent else ""
        return (f"- scope{n} table: {len(df)} rows, {df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}, "
                f"total {df['co2e_tonnes'].sum():,.1f}. Columns: {', '.join(c for c in df.columns if c not in ('month', 'quarter'))}.{names}")

    findings = [f for f in run_quality_checks("all", inv)["findings"] if f["severity"] in ("high", "medium")]
    caveats = "\n".join(f"- {f['headline']}." for f in findings)
    return f"""\
DATA CARD (company emissions inventory; units tCO2e)
{describe(1, inv.scope1)}
{describe(2, inv.scope2)}
{describe(3, inv.scope3)}
KNOWN CAVEATS (computed by the quality checks; always mention the relevant ones, call run_quality_checks for evidence):
{caveats}
- Annualised figures extrapolate shorter scopes to 12 months and assume no seasonality, so they are estimates.
"""
