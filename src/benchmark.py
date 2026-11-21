"""Peer benchmarking against figures extracted from the peer reports (data/derived/peers.json)."""

from __future__ import annotations

import json
from functools import lru_cache

from src.analytics import emissions_summary
from src.data import DATA_DIR, Inventory, get_inventory

PEERS_PATH = DATA_DIR / "derived" / "peers.json"


@lru_cache(maxsize=1)
def load_peers() -> dict:
    return json.loads(PEERS_PATH.read_text())


def _ratio(a: float | None, b: float | None) -> float | None:
    return round(a / b, 1) if a and b else None


def compare_peers(inv: Inventory | None = None) -> dict:
    """Compare the company's (annualised) Scope 1 and 2 with peer-reported figures."""
    inv = inv or get_inventory()
    peers = load_peers()
    ann = emissions_summary(inv)["annualised_estimate"]
    s1, s2 = ann["scope1_tco2e"], ann["scope2_tco2e"]

    s1_drivers = inv.scope1.groupby(["facility", "activity_type"]).co2e_tonnes.sum().nlargest(3) * 4
    s2_drivers = inv.scope2.groupby(["facility", "energy_type"]).co2e_tonnes.sum().nlargest(3) * 4

    table = [{
        "company": "Our company (annualised from Q1 2024)", "industry": "Manufacturing (coal boilers, cement process, fleet)",
        "scope1_tco2e": s1, "scope2_location_based_tco2e": s2, "scope2_market_based_tco2e": None,
        "scope1_plus_2_tco2e": round(s1 + s2, 1), "scope3_tco2e": ann["scope3_tco2e"],
    }]
    ratios = []
    for p in peers["peers"]:
        s2_lb = p.get("scope2_location_based_tco2e") or p.get("scope2_reported_tco2e")
        table.append({
            "company": p["company"], "industry": p["industry"], "period": p["reporting_period"],
            "scope1_tco2e": p["scope1_tco2e"], "scope2_location_based_tco2e": s2_lb,
            "scope2_market_based_tco2e": p.get("scope2_market_based_tco2e"),
            "scope1_plus_2_tco2e": round(p["scope1_tco2e"] + (s2_lb or 0), 1),
            "scope3_tco2e": p.get("scope3_tco2e"), "scope3_coverage": p["scope3_coverage"],
            "scope2_method": p["scope2_method"], "renewables": p["renewables"], "intensity": p["intensity"],
            "source": f"{p['source_file']} p.{p['pages']}",
        })
        ratios.append({
            "peer": p["company"],
            "our_scope1_vs_peer_x": _ratio(s1, p["scope1_tco2e"]),
            "our_scope2_location_vs_peer_x": _ratio(s2, s2_lb),
            "our_scope1_plus_2_vs_peer_x": _ratio(s1 + s2, p["scope1_tco2e"] + (s2_lb or 0)),
        })
    return {
        "comparison_table": table,
        "ratios": ratios,
        "our_scope1_drivers_annualised": {" / ".join(k): round(v, 0) for k, v in s1_drivers.items()},
        "our_scope2_drivers_annualised": {" / ".join(k): round(v, 0) for k, v in s2_drivers.items()},
        "peer_context": {p["company"]: {"targets": p.get("targets"), "drivers_of_change": p.get("drivers_of_change"),
                                        "trend_scope1": p.get("trend_scope1_tco2e"), "trend_scope2": p.get("trend_scope2_tco2e")}
                         for p in peers["peers"]},
        "comparability_notes": peers["comparability_notes"],
    }
