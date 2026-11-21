"""Tools, data, analytics, quality, grounding and llm: the branches the main tests do not reach."""

import json
import types

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from src import analytics, grounding, llm, quality, tools
from src.data import get_inventory, load_inventory


# --- tools -------------------------------------------------------------------------------------

def test_every_tool_returns_json(monkeypatch):
    monkeypatch.setattr(grounding, "get_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no llm")))
    calls = [
        (tools.get_emissions_summary, {}),
        (tools.get_breakdown, {"scope": "1", "by": "facility"}),
        (tools.get_top_scope3, {"by": "activity", "top_n": 2}),
        (tools.get_supplier_priority, {}),
        (tools.run_sql, {"query": "SELECT COUNT(*) AS n FROM scope1"}),
        (tools.run_quality_checks, {"scope": "2"}),
        (tools.assess_quality, {"question": "Is scope 2 valid?", "scope": "2"}),
        (tools.calculate, {"expression": "16790.1 / 449"}),
    ]
    for tool, args in calls:
        assert isinstance(json.loads(tool.invoke(args)), (dict, list)), tool.name


def test_search_knowledge_base_maps_source_names(monkeypatch):
    seen = {}

    class FakeKB:
        def search(self, query, k, source):
            seen.update(query=query, source=source)
            return [{"title": "Doc", "page": 4, "text": "passage"}]

    monkeypatch.setattr(tools, "get_kb", lambda: FakeKB())
    out = json.loads(tools.search_knowledge_base.invoke({"query": "scope 2", "source": "peer1"}))
    assert out == [{"cite": "[Doc, p.4]", "text": "passage"}] and seen["source"] == "peer1_emissions_report.pdf"
    tools.search_knowledge_base.invoke({"query": "x"})
    assert seen["source"] is None


# --- analytics / data ---------------------------------------------------------------------------

def test_breakdown_reports_unknown_columns_and_includes_spend_for_scope3():
    assert "not in scope1" in analytics.breakdown(1, "nope")["error"]
    s3 = analytics.breakdown(3, "supplier")
    assert s3["rows"] and "spend_amount" in s3["rows"][0]


def test_calculate_rejects_bad_input():
    assert "error" in analytics.calculate("1 / 0")
    assert "error" in analytics.calculate("2 ** 1000")
    assert "error" in analytics.calculate("__import__('os')")
    assert analytics.calculate("2 + 3")["result"] == 5
    assert analytics.calculate("-5 + +3")["result"] == -2  # unary operators


def test_combined_table_has_common_columns_across_scopes():
    df = get_inventory().combined
    assert set(df["scope"]) == {1, 2, 3} and list(df.columns) == ["scope", "source", "activity", "co2e_tonnes", "date"]


def test_data_card_describes_the_loaded_inventory():
    from src.data import data_card

    card = data_card()
    assert "DATA CARD" in card and "scope3 table" in card and "KNOWN CAVEATS" in card and "unverified" in card


def test_data_card_without_inferred_mapping_omits_that_caveat():
    from src.data import Inventory, data_card

    inv = get_inventory()
    plain = Inventory(inv.scope1, inv.scope2, inv.scope3.drop(columns=["inferred_category_num"]))
    assert "LLM-inferred" not in data_card(plain)


def test_load_inventory_reads_csvs_from_a_folder(tmp_path):
    from src.data import DATA_DIR

    for name in ("scope1.csv", "scope2.csv", "scope3.csv"):
        (tmp_path / name).write_text((DATA_DIR / name).read_text())
    assert len(load_inventory(tmp_path).scope1) == len(get_inventory().scope1)


# --- quality rules on small hand-made tables ---------------------------------------------------

def test_implied_unit_is_none_when_no_unit_reproduces_the_total():
    assert quality._implied_unit(10, 2, 20) == "t CO2e per unit"
    assert quality._implied_unit(1000, 2, 2) == "kg CO2e per unit"
    assert quality._implied_unit(1, 1, 999) is None


def test_association_needs_at_least_two_categories_on_each_side():
    out = quality.association(pd.Series(["a", "a", "a"]), pd.Series(["x", "y", "z"]))
    assert out == {"excess_v": 0.0, "p": 1.0, "informative": False}


SPEC = quality.TableSpec(amount="amt", factor="f", reported="t", entity="ent", site="site", source="src",
                         renewable_pct="renew")


def _df(**cols):
    return pd.DataFrame(cols)


def test_single_factor_and_concentration_skip_missing_columns():
    df = _df(t=[1.0, 2.0], f=[1, 1], amt=[1, 2])
    no_cols = quality.TableSpec(amount="amt", factor="f", reported="t")
    assert quality._single_factor(1, df, no_cols) is None
    assert quality._concentration(1, df, no_cols) is None


def test_concentration_is_silent_when_no_entity_dominates():
    df = _df(t=[1.0, 1.0, 1.0], ent=["a", "b", "c"], f=[1, 1, 1], amt=[1, 1, 1])
    assert quality._concentration(1, df, SPEC) is None


def test_attribute_ignored_is_silent_when_the_attribute_never_changes():
    df = _df(t=[1.0, 1.0], site=["s", "s"], src=["e", "e"], renew=[10, 10], f=[1, 1], amt=[1, 1], ent=["s", "s"])
    assert quality._attribute_ignored(2, df, SPEC) is None


def test_renewable_claim_is_not_flagged_when_instrument_columns_exist():
    df = _df(t=[10.0, 10.0], renew=[50, 50], certificate_id=["a", "b"], f=[1, 1], amt=[1, 1])
    assert quality._renewable_unsupported(df, SPEC) is None
    assert quality._renewable_unsupported(df.drop(columns="certificate_id"), SPEC) is not None


# --- grounding -----------------------------------------------------------------------------------

def test_format_passages_joins_with_ids_and_citations():
    text = grounding.format_passages([{"id": "P1", "cite": "[Doc, p.1]", "text": "a"},
                                      {"id": "P2", "cite": "[Doc, p.2]", "text": "b"}])
    assert text == "P1 [Doc, p.1]\na\n\n---\n\nP2 [Doc, p.2]\nb"


# --- llm client construction -------------------------------------------------------------------

@pytest.fixture
def fake_provider(monkeypatch):
    for v in ("LLM_MODEL", "LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
              "OLLAMA_BASE_URL", "LLM_MAX_TOKENS"):
        monkeypatch.delenv(v, raising=False)
    made = {}

    def fake_import(_module):
        class Chat:
            def __init__(self, **kwargs):
                made.update(kwargs)

        return types.SimpleNamespace(**{p.cls: Chat for p in llm.PROVIDERS.values()})

    monkeypatch.setattr(llm.importlib, "import_module", fake_import)
    return monkeypatch, made


@pytest.mark.parametrize("provider,key,extra", [
    ("google_genai", "GEMINI_API_KEY", {"google_api_key": "k", "timeout": llm.TIMEOUT_S}),
    ("anthropic", "ANTHROPIC_API_KEY", {"max_tokens": 4096}),
    ("openai", "OPENAI_API_KEY", {"max_retries": llm.MAX_RETRIES}),
])
def test_get_llm_passes_provider_specific_settings(fake_provider, provider, key, extra):
    mp, made = fake_provider
    mp.setenv(key, "k")
    mp.setenv("LLM_PROVIDER", provider)
    mp.setenv("LLM_MODEL", "m1")
    llm.get_llm(0.1)
    assert made["model"] == "m1" and made["temperature"] == 0.1
    assert extra.items() <= made.items()


def test_get_llm_ollama_uses_base_url_when_set(fake_provider):
    mp, made = fake_provider
    mp.setenv("LLM_PROVIDER", "ollama")
    mp.setenv("LLM_MODEL", "llama3.1")
    mp.setenv("OLLAMA_BASE_URL", "http://host:11434")
    llm.get_llm()
    assert made["base_url"] == "http://host:11434"


def test_text_of_handles_string_and_block_content():
    assert llm.text_of(AIMessage("plain")) == "plain"
    blocks = AIMessage(content=[{"type": "text", "text": "a"}, {"type": "thinking", "x": 1}, "b"])
    assert llm.text_of(blocks).startswith("a") and llm.text_of(blocks).endswith("b")
