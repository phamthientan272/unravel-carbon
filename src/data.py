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

# Where each activity *should* sit under the GHG Protocol, used to detect mislabelling
# and to offer a reclassified view of Scope 3.
EXPECTED_CATEGORY = {
    "Raw Materials - Steel": 1,
    "Raw Materials - Aluminum": 1,
    "Packaging Materials": 1,
    "Office Supplies": 1,
    "IT Hardware": 2,
    "Manufacturing Equipment": 2,
    "Inbound Logistics - Truck": 4,
    "Business Travel - Air": 6,
    "Employee Commuting": 7,
    "Product Use Phase": 11,
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
    s3["expected_category_num"] = s3["activity_description"].map(EXPECTED_CATEGORY)
    s3["expected_category_name"] = s3["expected_category_num"].map(SCOPE3_CATEGORIES)
    s3["category_mismatch"] = s3["category_num"] != s3["expected_category_num"]
    s3["supplier_mismatch"] = [
        act not in EXPECTED_SUPPLIER_ACTIVITIES.get(sup, set())
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

    The model gets the shape and caveats of the data, never the raw rows.
    """
    inv = inv or get_inventory()
    s1, s2, s3 = inv.scope1, inv.scope2, inv.scope3

    def period(df: pd.DataFrame) -> str:
        return f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}"

    return f"""\
DATA CARD (company emissions inventory; units tCO2e)
- scope1 table: {len(s1)} rows, {period(s1)}, total {s1.co2e_tonnes.sum():,.1f}. Columns: facility, activity_type, fuel_type, consumption_amount, consumption_unit, emission_factor, co2e_tonnes, date, notes.
  Facilities: {', '.join(sorted(s1.facility.unique()))}.
- scope2 table: {len(s2)} rows, {period(s2)}, total {s2.co2e_tonnes.sum():,.1f} (location-based only). Columns: facility, energy_type, consumption_amount, consumption_unit, grid_region, emission_factor, co2e_tonnes, date, renewable_percentage, notes.
- scope3 table: {len(s3)} rows, {period(s3)}, total {s3.co2e_tonnes.sum():,.1f}. Columns: category (e.g. 'Category 6'), category_num, category_name, activity_description, spend_amount (USD), emission_factor (kg CO2e/USD), co2e_tonnes, date, supplier, data_quality, expected_category_num, category_mismatch, supplier_mismatch.
KNOWN CAVEATS (always mention the relevant ones):
- Scope 1 and 2 cover Jan-Mar 2024 only (3 months); Scope 3 covers Jan-Dec 2024. Totals across scopes mix periods.
- Scope 3 is 100% spend-based. {s3.category_mismatch.mean():.0%} of rows have a category label inconsistent with the activity (e.g. air travel labelled Category 11), so label-based category totals are unreliable; tools offer a reclassified view by activity.
- {s3.supplier_mismatch.mean():.0%} of Scope 3 rows pair a supplier with an implausible activity (e.g. 'Corporate Travel' supplying trucking).
- Scope 2 reports only one (location-based) figure even though renewable_percentage is recorded.
"""
