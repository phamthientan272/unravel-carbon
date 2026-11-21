"""Streamlit UI for the Emissions Analysis & Insights Agent.

    uv run streamlit run app.py
"""

from __future__ import annotations

import uuid

import streamlit as st

from src.kb import index_exists

st.set_page_config(page_title="Emissions Insights Agent", page_icon="🌍", layout="wide")

CORE_QUESTIONS = [
    "Should employee business travel be classified as Scope 1 or Scope 3? Explain the reasoning and describe how I can calculate my business travel emissions?",
    "Are my scope 2 emissions calculation valid according to the Greenhouse Gas Protocol?",
    "How do my scope 1 & 2 emissions compare with other companies in my industry (refer to the peer reports), and what insights can I derive from this comparison?",
]
BONUS_QUESTIONS = [
    "What is our highest emitting Scope 3 category and what specific activities contribute to it?",
    "Which suppliers should I prioritise to engage for emissions reduction efforts?",
    "Generate a summary report of our total emissions by scope with key insights",
]


@st.cache_resource
def get_graph():
    from src.graph import build_graph

    return build_graph()


def init_state() -> None:
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("chat", [])  # list of {role, content, trace?, intent?}
    st.session_state.setdefault("pending", None)


def sidebar() -> None:
    with st.sidebar:
        st.header("🌍 Emissions Insights Agent")
        st.caption("Ask about your inventory, the GHG Protocol, data quality or peers. "
                   "Figures come from deterministic tools; methodology is cited from the GHG Protocol.")
        st.subheader("Core questions")
        for i, q in enumerate(CORE_QUESTIONS):
            if st.button(q, key=f"core{i}", width="stretch"):
                st.session_state.pending = q
        st.subheader("Bonus questions")
        for i, q in enumerate(BONUS_QUESTIONS):
            if st.button(q, key=f"bonus{i}", width="stretch"):
                st.session_state.pending = q
        st.divider()
        if st.button("🗑️ New conversation", width="stretch"):
            st.session_state.chat, st.session_state.thread_id = [], str(uuid.uuid4())
            st.rerun()


def render_trace(msg: dict) -> None:
    if msg.get("trace"):
        with st.expander(f"How I answered · intent: {msg.get('intent')}"):
            for t in msg["trace"]:
                st.markdown(f"- **{t['step']}** · `{t['detail'][:400]}`")


def chat_tab() -> None:
    if not index_exists():
        st.warning("Knowledge base not built yet. Run `uv run python -m src.kb` once, then reload.")
        return
    if not st.session_state.chat:
        st.info("Pick a question in the sidebar or type your own below. Follow-up questions keep the conversation context.")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_trace(msg)

    question = st.chat_input("Ask about your emissions…") or st.session_state.pending
    if question:
        st.session_state.pending = None
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                from src.graph import ask

                try:
                    out = ask(get_graph(), question, st.session_state.thread_id)
                except Exception as e:  # e.g. rate limits on the free tier
                    out = {"answer": f"⚠️ The model call failed: `{type(e).__name__}: {str(e)[:300]}`. Please retry.",
                           "intent": None, "trace": []}
            msg = {"role": "assistant", "content": out["answer"] or "_(no answer)_", "trace": out["trace"], "intent": out["intent"]}
            st.markdown(msg["content"])
            render_trace(msg)
        st.session_state.chat.append(msg)


def main() -> None:
    init_state()
    sidebar()
    chat_tab()


main()
