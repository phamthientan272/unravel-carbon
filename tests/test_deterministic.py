"""Unit tests for the deterministic layer (no LLM, no API key needed)."""

import pytest

from src import analytics, benchmark
from src.data import get_inventory
from src.quality import run_quality_checks


def test_scope_totals_match_hand_calculation():
    s = analytics.emissions_summary()["as_reported"]
    assert s["scope1_tco2e"] == pytest.approx(1692.5, abs=0.1)
    assert s["scope2_location_based_tco2e"] == pytest.approx(2505.0, abs=0.1)
    assert s["scope3_tco2e"] == pytest.approx(374485.5, abs=0.1)


def test_annualisation_only_applies_to_scope_1_and_2():
    a = analytics.emissions_summary()["annualised_estimate"]
    assert a["scope1_tco2e"] == pytest.approx(1692.47 * 4, abs=0.1)
    assert a["scope3_tco2e"] == pytest.approx(374485.5, abs=0.1)


def test_top_scope3_views():
    assert analytics.top_scope3("category", 1)["top"][0]["category_num"] == 3
    assert analytics.top_scope3("activity", 1)["top"][0]["activity_description"] == "Business Travel - Air"
    top = analytics.top_scope3("reclassified_category", 1)["top"][0]
    assert top["expected_category_num"] == 1 and top["co2e_tonnes"] == pytest.approx(133282.7, abs=0.1)


def test_supplier_priority_covers_all_suppliers():
    rows = analytics.supplier_priority()["suppliers"]
    assert len(rows) == 8
    assert rows == sorted(rows, key=lambda r: -r["priority_score"])


def test_run_sql_reads_and_blocks_writes():
    ok = analytics.run_sql("SELECT SUM(co2e_tonnes) AS t FROM scope1")
    assert ok["rows"][0]["t"] == pytest.approx(1692.47, abs=0.01)
    assert "error" in analytics.run_sql("DROP TABLE scope1")
    assert "error" in analytics.run_sql("SELECT 1; DELETE FROM scope1")
    assert "error" in analytics.run_sql("SELECT * FROM missing_table")


def test_quality_flags_known_issues():
    ids = {f["id"]: f for f in run_quality_checks()["findings"]}
    assert ids["S2-DUAL"]["severity"] == "high"
    assert ids["ALL-PERIOD"]["severity"] == "high"
    assert ids["S3-CATEGORY-MAP"]["severity"] == "high"
    assert ids["S1-RECALC"]["severity"] == "pass" and ids["S2-RECALC"]["severity"] == "pass"


def test_quality_scope_filter():
    scopes = {f["scope"] for f in run_quality_checks("2")["findings"]}
    assert scopes <= {"2", "1,2", "all"}


def test_category_mismatch_rate():
    assert get_inventory().scope3.category_mismatch.mean() == pytest.approx(0.92, abs=0.01)


def test_peer_comparison():
    r = {x["peer"]: x for x in benchmark.compare_peers()["ratios"]}
    assert r["DocuSign"]["our_scope1_vs_peer_x"] == pytest.approx(15.1, abs=0.1)
    assert r["Shopify"]["our_scope1_vs_peer_x"] == pytest.approx(18.8, abs=0.1)
