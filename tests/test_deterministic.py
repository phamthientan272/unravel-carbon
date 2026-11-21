"""Unit tests for the deterministic layer (no LLM, no API key needed)."""

import pytest

from src import analytics
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
    assert ids["S2-RENEW-UNSUPPORTED"]["severity"] == "medium"
    assert ids["S2-RENEW-INCONSISTENT"]["severity"] == "medium"
    assert ids["ALL-PERIOD"]["severity"] == "high"
    assert ids["S3-CATEGORY-MAP"]["severity"] == "high"
    assert ids["S1-RECALC"]["severity"] == "pass" and ids["S2-RECALC"]["severity"] == "pass"


def test_quality_scope_filter():
    scopes = {f["scope"] for f in run_quality_checks("2")["findings"]}
    assert scopes <= {"2", "1,2", "all"}


def test_scope3_findings_are_computed_not_asserted():
    ids = {f["id"]: f for f in run_quality_checks("3")["findings"]}
    assert ids["S3-RECALC"]["severity"] == "pass"
    assert "uncorrelated" in ids["S3-FACTORS"]["evidence"]
    assert "permutation p" in ids["S3-CATEGORY-MAP"]["evidence"] and ids["S3-CATEGORY-MAP"]["basis"] == "statistical"


def test_every_finding_has_dimension_basis_and_catalog_advice():
    for f in run_quality_checks()["findings"]:
        assert f["dimension"] and f["basis"] and f["headline"] and f["fallback_query"]


def test_catalog_states_nothing_from_outside_the_provided_document():
    from src.quality import CATALOG_PATH

    text = CATALOG_PATH.read_text()
    assert "recommendation" not in text.replace("Nothing in this file", "")  # no stored advice
    assert "ghg_reference" not in text
    for outside in ("Scope 2 Guidance", "USEEIO", "eGRID", "DEFRA", "EPA", "market-based", "IPCC"):
        assert outside not in text, outside


def test_unverified_citations_are_removed():
    from src.grounding import strip_unverified_citations

    text = "Double counting is addressed [GHG Protocol, p.27] and also [GHG Protocol, p.999]."
    out, removed = strip_unverified_citations(text, {27})
    assert "[GHG Protocol, p.27]" in out and "p.999" not in out and removed == [999]


def test_query_planner_failure_falls_back_to_fixed_queries(monkeypatch):
    from src import grounding

    def boom(*a, **k):
        raise RuntimeError("no llm")

    monkeypatch.setattr(grounding, "get_llm", boom)
    findings = [f for f in run_quality_checks()["findings"] if f["severity"] == "high"]
    plan = grounding.plan_queries("Are my scope 2 emissions valid?", findings)
    assert {i.finding_id for i in plan.items} == {f["id"] for f in findings}
    assert plan.question_queries == ["Are my scope 2 emissions valid?"]


def test_retrieval_uses_only_the_ghg_document_and_links_passages_to_findings():
    from src import grounding

    findings = [f for f in run_quality_checks()["findings"] if f["id"] in ("ALL-PERIOD", "S2-DOUBLE-COUNT")]
    plan = grounding.QueryPlan(items=[], question_queries=["scope 2 purchased electricity"])  # planner gave nothing: fallback queries
    links, passages = grounding.retrieve_for_findings(plan, findings)
    assert passages and all("GHG" in p["cite"] or "Greenhouse" in p["cite"] or "Protocol" in p["cite"] for p in passages)
    assert all(links[f["id"]] for f in findings) and links["QUESTION"]


def test_label_ranking_is_caveated_by_a_statistical_test_not_a_mapping():
    caveat = analytics.top_scope3("category", 1)["caveat"]
    assert "almost no information" in caveat and "unreliable" in caveat
    assert analytics.top_scope3("activity", 1)["caveat"] is None


def test_supplier_ranking_is_caveated():
    assert "indicative" in analytics.supplier_priority()["intensity_caveat"]


def test_calculate_is_exact_and_refuses_anything_but_arithmetic():
    assert analytics.calculate("(1692.47 * 4) / 449")["result"] == pytest.approx(15.08, abs=0.01)
    assert analytics.calculate("2 ** 10 + 1")["result"] == 1025
    for bad in ("__import__('os').system('ls')", "abs(-1)", "1 +", "9 ** 9 ** 9", "x + 1"):
        assert "error" in analytics.calculate(bad), bad


# --- the quality rules must work on a different company's data, with different names ----------------

def _new_company_inventory():
    """Synthetic inventory: other column names, sites, fuels and suppliers than the sample data."""
    import numpy as np
    import pandas as pd

    from src.data import Inventory

    m = pd.date_range("2023-01-15", periods=6, freq="MS")
    s1 = pd.DataFrame({
        "site": ["Berlin Office"] * 6 + ["Hamburg Depot"] * 6,
        "fuel": ["Diesel"] * 6 + ["Propane"] * 6, "qty": [100.0] * 12,
        "ef": [2.7] * 6 + [1.5] * 6, "tco2e": [0.27] * 6 + [150.0] * 6, "date": list(m) * 2,   # diesel in kg/unit, propane in t/unit
    })
    s2 = pd.DataFrame({
        "site": ["Berlin Office"] * 6 + ["Hamburg Depot"] * 6, "fuel": ["Heat"] * 12, "qty": [1000.0] * 12,
        "ef": [0.2] * 12, "tco2e": [200.0] * 12, "date": list(m) * 2,
    })
    rng = np.random.default_rng(1)
    n = 300
    s3 = pd.DataFrame({
        "activity": rng.choice(["Cloud hosting", "Laptops", "Flights", "Catering"], n),
        "vendor": rng.choice(["A", "B", "C"], n), "cat": rng.choice([f"Category {i}" for i in range(1, 8)], n),
        "spend": rng.uniform(1e3, 1e5, n), "ef": rng.uniform(0.05, 3.0, n), "date": pd.to_datetime("2023-06-15"),
    })
    s3["tco2e"] = (s3.spend * s3.ef / 1000).round(2)
    return Inventory(scope1=s1, scope2=s2, scope3=s3)


def test_quality_rules_generalise_to_a_new_company():
    from src.quality import QualitySpec, TableSpec

    spec = QualitySpec(
        scope1=TableSpec(amount="qty", factor="ef", reported="tco2e", entity="site", site="site", source="fuel"),
        scope2=TableSpec(amount="qty", factor="ef", reported="tco2e", entity="site", site="site", source="fuel"),
        scope3=TableSpec(amount="spend", factor="ef", reported="tco2e", entity="vendor", source="activity",
                         method="spend", category="cat", supplier="vendor"),
    )
    result = run_quality_checks("all", inv=_new_company_inventory(), spec=spec)
    ids = {f["id"]: f for f in result["findings"]}
    assert ids["ALL-PERIOD"]["severity"] == "high"                      # 6 vs 1 month, found without any hardcoded dates
    assert "kg CO2e per unit" in ids["S1-RECALC"]["evidence"] and "t CO2e per unit" in ids["S1-RECALC"]["evidence"]
    assert ids["ALL-FACTOR-META"]["severity"] == "medium"               # mixed implied units, no unit column
    assert ids["S2-DOUBLE-COUNT"]["id"] and "Berlin Office" in ids["S2-DOUBLE-COUNT"]["evidence"]  # heat bought where fuel is burned
    assert ids["S2-SINGLE-FACTOR"]["severity"] == "medium"
    assert ids["S3-CATEGORY-MAP"]["severity"] == "high"                 # random labels are detected by the statistical test
    assert "Manufacturing" not in str(result) and "Steam" not in str(result)   # nothing carried over from the sample data


def test_pages_in_reads_every_citation_style():
    from src.grounding import pages_in

    text = '[GHG Protocol, p.27] peer1_emissions_report.pdf p.[49, 50] pp.12-14 {"page": 63}'
    assert pages_in(text) == {27, 49, 50, 12, 14, 63}


def test_verify_node_removes_citations_to_pages_not_retrieved_in_the_turn():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from src.graph import verify_node

    state = {"messages": [
        HumanMessage("how is business travel classified?"),
        ToolMessage(content='[{"cite": "[GHG Protocol, p.27]", "text": "..."}]', tool_call_id="1", name="search_knowledge_base"),
        AIMessage("Scope 3 [GHG Protocol, p.27]. Also [GHG Protocol, p.500].", id="a1"),
    ], "pages": []}
    out = verify_node(state)
    text = out["messages"][0].content
    assert "p.27" in text and "p.500" not in text and out["messages"][0].id == "a1"
    assert "citation check" in out["notes"][0]
    clean = {"messages": state["messages"][:2] + [AIMessage("No citations here.", id="a2")], "pages": []}
    assert verify_node(clean) == {}


def test_low_confidence_routes_to_the_agent():
    from src.graph import route_next

    assert route_next({"intent": "quality_assessment", "low_confidence": False}) == "quality"
    assert route_next({"intent": "quality_assessment", "low_confidence": True}) == "agent"
    assert route_next({"intent": "report"}) == "report"


def test_assess_quality_tool_is_registered_and_links_passages(monkeypatch):
    import json

    from src import grounding
    from src.tools import ALL_TOOLS, assess_quality

    monkeypatch.setattr(grounding, "get_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no llm")))  # planner falls back
    assert "assess_quality" in {t.name for t in ALL_TOOLS}
    out = json.loads(assess_quality.invoke({"question": "Are my scope 2 emissions valid?", "scope": "2"}))
    assert out["passages"] and all(len(p["text"]) <= 700 for p in out["passages"])
    assert any(f.get("passage_ids") for f in out["findings"])


# --- LLM provider selection (no network, no API keys needed) -------------------------------------

@pytest.fixture
def clean_llm_env(monkeypatch):
    for v in ("LLM_MODEL", "LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
              "OLLAMA_BASE_URL"):
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def test_llm_provider_is_auto_detected_from_the_api_key(clean_llm_env):
    from src.llm import resolve_llm

    clean_llm_env.setenv("ANTHROPIC_API_KEY", "x")
    cfg = resolve_llm()
    assert cfg.provider == "anthropic" and cfg.model == "claude-sonnet-5-5" and "ANTHROPIC_API_KEY" in cfg.source
    clean_llm_env.setenv("GEMINI_API_KEY", "x")  # both set: Google comes first, as before
    assert resolve_llm().provider == "google_genai"


def test_llm_model_prefix_and_provider_override(clean_llm_env):
    from src.llm import resolve_llm

    clean_llm_env.setenv("GEMINI_API_KEY", "x")
    clean_llm_env.setenv("LLM_MODEL", "anthropic:claude-test")
    cfg = resolve_llm()
    assert (cfg.provider, cfg.model) == ("anthropic", "claude-test")
    clean_llm_env.setenv("LLM_MODEL", "my-model")
    clean_llm_env.setenv("LLM_PROVIDER", "openai")
    assert (resolve_llm().provider, resolve_llm().model) == ("openai", "my-model")


def test_ollama_model_names_with_a_colon_are_not_mistaken_for_a_prefix(clean_llm_env):
    from src.llm import resolve_llm

    clean_llm_env.setenv("LLM_PROVIDER", "ollama")
    clean_llm_env.setenv("LLM_MODEL", "llama3.1:8b")
    cfg = resolve_llm()
    assert (cfg.provider, cfg.model) == ("ollama", "llama3.1:8b")


def test_llm_configuration_errors_say_how_to_fix_them(clean_llm_env):
    from src.llm import LLMConfigError, describe_llm, get_llm, resolve_llm

    with pytest.raises(LLMConfigError, match="API key"):
        resolve_llm()
    assert describe_llm().startswith("not configured")
    clean_llm_env.setenv("OPENAI_API_KEY", "x")
    with pytest.raises(LLMConfigError, match="LLM_MODEL"):  # openai has no default model
        resolve_llm()
    clean_llm_env.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(LLMConfigError, match="Unknown LLM_PROVIDER"):
        resolve_llm()
    clean_llm_env.setenv("LLM_PROVIDER", "ollama")
    clean_llm_env.setenv("LLM_MODEL", "llama3.1")
    try:
        get_llm()
    except LLMConfigError as e:  # package not installed here: the message must name it
        assert "langchain-ollama" in str(e)


# --- layers, requirements and principle verdicts -------------------------------------------------------

def test_every_finding_has_layer_principle_and_pages_for_requirements():
    for f in run_quality_checks()["findings"]:
        assert f["layer"] in ("requirement", "qa", "investigation")
        assert f["principle"] in ("relevance", "completeness", "consistency", "transparency", "accuracy")
        if f["layer"] == "requirement":
            assert f["requirement"] and f["ref_pages"]
        if f["layer"] == "investigation":
            assert f["requirement"] == ""


def test_requirement_checks_on_the_sample_data():
    ids = {f["id"]: f for f in run_quality_checks()["findings"]}
    assert ids["REQ-SCOPES-SEPARATE"]["severity"] == "pass"
    assert ids["ALL-GAS-BREAKDOWN"]["severity"] == "medium" and "per gas" in ids["ALL-GAS-BREAKDOWN"]["evidence"]
    assert ids["S1-BIOGENIC"]["severity"] == "pass"
    assert ids["S2-DEFINITION"]["severity"] == "medium" and "Steam" in ids["S2-DEFINITION"]["evidence"]


def test_investigations_alone_never_make_a_gap():
    from src.quality import Finding, principle_verdicts

    inv = Finding(id="X", scope="3", severity="high", title="t", evidence="e", layer="investigation", principle="accuracy")
    qa = Finding(id="Y", scope="1", severity="medium", title="t", evidence="e", layer="qa", principle="completeness")
    ok = Finding(id="Z", scope="1", severity="pass", title="t", evidence="e", layer="qa", principle="consistency")
    v = principle_verdicts([inv, qa, ok])
    assert v["accuracy"]["status"] == "review"           # only an investigation is open
    assert v["completeness"]["status"] == "gap"          # a qa check failed
    assert v["consistency"]["status"] == "no_issues_found"
    assert v["relevance"]["status"] == "not_assessable"  # nothing checks it


def test_result_lists_what_the_data_cannot_show():
    r = run_quality_checks()
    items = " ".join(c["item"] for c in r["cannot_assess"])
    assert "Base year" in items and "Year-on-year" in items   # sample data covers one year only
    assert set(r["principle_verdicts"]) == {"relevance", "completeness", "consistency", "transparency", "accuracy"}
    assert r["principle_verdicts"]["completeness"]["status"] == "gap"   # gas breakdown missing


def test_new_company_gets_biogenic_and_gas_checks_without_hardcoded_names():
    import pandas as pd
    from src.quality import QualitySpec, TableSpec

    inv = _new_company_inventory()
    inv.scope1.loc[0, "fuel"] = "Wood pellets"
    inv.scope1["co2_t"] = 1.0   # a per-gas column exists
    spec = QualitySpec(
        scope1=TableSpec(amount="qty", factor="ef", reported="tco2e", entity="site", site="site", source="fuel"),
        scope2=TableSpec(amount="qty", factor="ef", reported="tco2e", entity="site", site="site", source="fuel"),
        scope3=TableSpec(amount="spend", factor="ef", reported="tco2e", entity="vendor", source="activity",
                         method="spend", category="cat", supplier="vendor"),
    )
    ids = {f["id"]: f for f in run_quality_checks("all", inv=inv, spec=spec)["findings"]}
    assert ids["S1-BIOGENIC"]["severity"] == "medium" and "Wood pellets" in ids["S1-BIOGENIC"]["evidence"]
    assert ids["ALL-GAS-BREAKDOWN"]["severity"] == "pass"
    assert ids["S2-DEFINITION"]["severity"] == "medium"   # heat in the scope 2 table


def test_requirement_pages_are_retrieved_so_they_can_be_cited():
    from src import grounding

    findings = [f for f in run_quality_checks()["findings"] if f["id"] == "ALL-GAS-BREAKDOWN"]
    plan = grounding.QueryPlan(items=[], question_queries=[])
    links, passages = grounding.retrieve_for_findings(plan, findings)
    assert 65 in {p["page"] for p in passages} and links["ALL-GAS-BREAKDOWN"]
