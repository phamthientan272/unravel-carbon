# Emissions Analysis & Insights Agent

A prototype LangGraph agent for sustainability teams: inventory questions, GHG Protocol education with page citations,
peer benchmarking, summary reports and a data-quality assessment. Stack: LangGraph + Gemini, Chroma + BM25 retrieval,
pandas/DuckDB analytics, Streamlit.

**Models used:** LLM `gemini-3.5-flash-lite` (Google AI Studio free tier; the brief's `gemini-4-hydrogen` ran out of
quota). OCR of the image-only Shopify report: the built-in **macOS Vision** engine, run locally once and cached.
Embeddings: Chroma's local ONNX model.

# One-page overview

**Solution.** One chat for sustainability analysts. A router sends each question to one of four capabilities:
inventory questions (exact numbers), GHG Protocol education (page citations), summary report with peer benchmarking, and
a data-quality assessment with prioritised fixes. I prioritised reliability (grounded numbers, citations, traceability)
over breadth. A "How I answered" expander shows the route and every tool call.

**Approach and architecture.** LangGraph router (structured-output LLM) → fixed pipelines for reports and quality
(tools first, then one LLM write-up) or a ReAct agent with tools for open questions. The LLM never does arithmetic:
numbers come from pandas tools or read-only DuckDB SQL. Context is kept small with a ~300-token data card (schema,
periods, totals, caveats) instead of raw rows. A post-check strips citations to pages that were not retrieved. The model
is swappable through environment variables.

**Key observations about the data.**
- Scope 1/2 cover Jan-Mar 2024 only, Scope 3 covers all of 2024, so the agent shows as-reported and annualised views.
- Scope 1 is 94% one site (coal boilers, cement process). Scope 2 is location-based only, the recorded renewable share
  is never applied, and steam in Scope 2 may double-count coal boilers in Scope 1.
- The Scope 3 category labels look synthetic (no association with the activity), so the agent reports them with a
  warning and also ranks by activity.
- The peers are SaaS and e-commerce companies, not manufacturers, and Shopify's emissions tables are images (OCR'd once).

**PDF and CSV in the LLM's context.** CSVs are reached through typed tools and a SQL tool, never embedded, because
semantic search cannot sum or rank. PDFs go through hybrid retrieval: page-bounded chunks in Chroma plus BM25, merged
by reciprocal rank fusion, cited as `[document, p.N]`. Peer figures are used only when the retrieved passage shows the
row and year.

**Evaluation and scaling.** Built: 67 unit tests (deterministic tools against hand-computed values, quality rules,
graph nodes, retrieval, trace) and a per-turn JSON trace with tokens, latency and tool calls. Planned: a golden set
scored on retrieval (Recall@K, MRR), generation (faithfulness, relevance, correctness) and routing. To scale: a document
registry with per-peer fan-out and reranking, tables stored in DuckDB, a VLM for image pages, a server vector DB, and
Parquet or a warehouse behind the SQL tool. Details are in section 5.

**Ideas for improvement.** A data-quality tab that turns findings into actions with owners and status, a dashboard
with drill-down, an exportable report module, reduction scenarios, market-based Scope 2 with a factor library,
disclosure mapping (CDP/CSRD) and sector-matched peers. Details are in section 6.

**Challenges.** The brief gives no method for judging inventory quality, so the quality checks are a working scaffold
from the GHG Protocol's "shall" statements, not a validated method. The suggested model ran out of free-tier quota, so
I used `gemini-3.5-flash-lite` and kept LLM calls per question low. Shopify's image tables needed OCR. Details are in
section 7.

*The sections below are the full write-up behind this summary.*

---

# Detailed overview

## 1. Proposed solution
A single chat for analysts. The agent routes each question to one of four capabilities: **questions over the inventory**
(exact numbers), **GHG Protocol education** (cited pages), **summary report + peer benchmarking**, and **quality
assessment** with a prioritised list of fixes. Priority went to the agent's reliability (grounded numbers, citations,
traceability) over breadth of features. A "How I answered" expander shows the route and every tool call, so answers are
auditable.

## 2. Technical approach and architecture decisions
```
Streamlit ─► LangGraph router (LLM, structured output) ─┬─► report pipeline  (tools → 1 LLM write-up)
                                                        ├─► quality pipeline (rules + retrieved passages → 1 LLM write-up)
                                                        └─► ReAct agent ⇄ tools → citation check
Tools: emissions_summary · breakdown · top_scope3 · supplier_priority · run_sql (read-only DuckDB) · calculate
       run_quality_checks · assess_quality · search_knowledge_base (Chroma + BM25, RRF fusion, page citations)
```
- **The LLM never does arithmetic.** It plans, picks tools and writes; every number comes from a deterministic pandas
  tool or SELECT-only SQL.
- **Fixed pipelines for repeatable, high-stakes tasks** (reports, quality): 2 LLM calls instead of 4-8, consistent
  structure, easy to evaluate. A ReAct loop handles open questions.
- **Context engineering:** the system prompt carries a ~300-token *data card* (schema, periods, totals, caveats), not raw
  rows; tool outputs are compact JSON; history is windowed.
- **Citations are enforced:** references to pages that were not retrieved are stripped, and the model must say so when
  the document does not address a point.
- **Swappable model** via environment variables (see Setup).
- **Current assumptions:** two peers and one standard; documents small enough to read passages at question time; one
  inventory with the provided column layout; the PDF text layer (plus cached OCR) is the truth.

### RAG architecture (`src/kb.py`, `src/grounding.py`)
```
INGEST (once)   PDF ─► page text (pymupdf; Shopify = cached OCR) ─► clean ─► chunk (~1.5k chars, 250 overlap, within a page)
                  └─► Chroma (local ONNX embeddings, cosine) + metadata {source, title, page}     BM25 index built in memory at load

QUERY           query ─┬─► vector search  (top 30, optional source filter) ─┐
                       └─► BM25 keyword   (top 30, same filter)             ├─► RRF merge (k=60) ─► top 6 ─► LLM
                                                                            ┘     passages as [document, p.N]
```
- **Where the query comes from.** On the ReAct route the LLM writes it when it calls `search_knowledge_base(query, source)`
  and can pin `source` to the standard, peer1 or peer2. On the quality route one LLM call (structured output) writes
  all the queries at once: 1-2 short queries per finding plus 1-2 for the user's question (a second query gives another
  wording of the same requirement a chance to match). If that call fails, a fixed fallback query per finding is used.
  Each query is then searched with no further LLM call. The same string
  goes to both retrievers: no rewriting or expansion inside the retrieval layer.
- **Why hybrid.** Embeddings capture meaning ("what counts as a direct emission"), BM25 catches exact tokens ("Category 6",
  "R404A", "tCO2e"). Reciprocal rank fusion merges ranks, not scores, so the two incomparable scales need no tuning.
- **Why page-bounded chunks.** A chunk never crosses a page, so every passage maps to one citable page. Overlap keeps
  sentences that straddle a chunk boundary retrievable.
- **Grounding rules.** The model may cite only pages it was given; a post-check strips any other citation, and it must say
  so when no passage addresses the point. Peer numbers are used only when the passage shows the row and year.
- **Quality route.** Rules compute the findings from the data first, retrieval then finds the protocol text behind each
  finding, and the LLM only explains and prioritises. Retrieval supports the answer, it never produces the numbers.
- **Not implemented (deliberate, for a prototype):** reranking, query rewriting inside the retriever, per-peer fan-out,
  and table-aware chunking. With three documents the fused top 6 is enough; section 5 covers fan-out, table handling and reranking.

## 3. Key observations about the data
- **Periods differ:** Scope 1/2 cover Jan-Mar 2024 only, Scope 3 all of 2024. The agent shows as-reported and annualised views.
- **Scope 1** is 94% the Manufacturing Plant (coal boilers, cement process).
- **Scope 2** is location-based only; `Renewable_Percentage` is recorded but never applied; the plant reports coal
  boilers in Scope 1 and "process heating" steam in Scope 2 (possible double count).
- **Scope 3 labels look synthetic:** the category carries no information about the activity (a permutation test finds no
  association). The agent reports the labelled view, flags it as unreliable, and also ranks by activity.
- **Peers are not industry peers** (SaaS and e-commerce vs. a coal/cement manufacturer), and Shopify's emissions tables
  are images.

## 4. PDF and CSV integration into the LLM's context
- **CSV → tools, not embeddings.** Semantic search cannot sum or rank, so rows are never embedded. The agent reaches the
  tables through typed tools plus a DuckDB SQL tool for long-tail questions.
- **PDF → hybrid RAG** (details in section 2): chunked per page, embedded in Chroma and indexed in BM25, merged by RRF,
  cited as `[document, p.N]`.
- **Peer figures come from retrieval only**, filtered by `source=peer1|peer2`. A number is used only if the passage shows
  its row and year; ratios use `calculate`. Shopify's image tables were OCR'd once (macOS Vision or Gemini vision) and
  cached in `data/derived/`.
- **GHG Protocol is the only authority** for protocol claims.
- **Quality assessment** is rule-based (`src/quality.py`): checks produce severity and evidence computed from the data,
  then retrieval finds the protocol passages that state each requirement and the LLM explains and prioritises. Findings
  are labelled requirement, QA or investigation, and there is no single score.

## 5. Evaluation and scaling in production
### Evaluation
**Implemented today:** unit tests on every deterministic tool against hand-computed values, the quality rules on a
synthetic company, graph nodes, the KB and the audit trace (`uv run pytest`). **Planned (not built):** the golden set and
metrics below. The tool layer is deterministic, so evaluation focuses on the two probabilistic parts: retrieval and the
LLM's generation.

**Golden set.** ~50-100 labelled questions across the four routes, each with the expected route, the expected
tool calls, the passages (document + page) that answer it, the required facts or numbers, and a reference answer. Build
it from the test questions, from real user questions once live, and from failed traces.

**1. Retrieval (offline, no LLM needed).** Run `KnowledgeBase.search` on each question's query and compare with the
labelled relevant pages.
- **Recall@K:** share of the relevant pages found in the top K (K=3, 6, 10). The key metric: if the page is not
  retrieved, the answer cannot be grounded.
- **Precision@K:** share of the top K that is relevant. It shows how much noise reaches the context. Also track **MRR**
  (rank of the first relevant hit).
- **Use:** compare BM25-only, vector-only and hybrid to justify the hybrid; tune chunk size, K and `rrf_k`; decide
  whether a reranker earns its cost (rerank the top 30, check whether Precision@6 rises). Re-run after every change
  to chunking, embeddings or documents. Labels are page-level, so a hit means the right page, not the right sentence.

**2. Generation (LLM-as-judge plus checks).** Judge on the final answer, the retrieved context and the tool results.
- **Faithfulness (groundedness):** every claim is supported by the retrieved passages or tool outputs. Split the answer
  into claims and have the judge check each against the context. Cheap code checks run alongside: cited pages are in the
  retrieved set (already enforced), and every number in the answer appears in a tool result.
- **Answer relevance:** the answer addresses the question asked, without padding or drifting to another topic.
- **Answer correctness:** against the reference answer and required facts. Numbers are checked exactly by code (e.g. top
  Scope 3 activity, totals), prose by the judge.
- **Product checks:** actionability of the quality recommendations, concision, and whether the agent says "the document
  doesn't address this" when it should.
- **Judge hygiene:** a different model from the generator, temperature 0, a fixed rubric, and a small human-labelled
  sample to measure agreement with the judge.

**3. Agent and routing.** Router accuracy (intent vs label), tool-call accuracy and wasted calls from the trace, and
LLM calls per question (cost proxy).

**Dev vs production.** In **dev**, run the golden set on every prompt, model, chunking or tool change as a regression
gate (retrieval and correctness must not drop). In **production**, no labels exist for live traffic, so sample traces
for the judge (faithfulness and relevance on e.g. 5-10% of turns), collect thumbs up/down and "answer was wrong"
flags, and turn every failure into a new golden-set case.

### Observability
Every turn already writes one JSON trace (`src/audit.py`, `TRACE_DIR`, off with `TRACE_ENABLED=0`). It is the basis for
both debugging and the metrics above.
- **Traceability:** the ordered events of the turn (router decision, each tool call with full arguments and result linked
  by `tool_call_id`, retrieval queries and `pages_retrieved`, the citation check, the final answer), plus `trace_id`,
  `thread_id`, the LLM and provider, and the git `code_version`. Any answer can be replayed and checked against the
  exact evidence it saw.
- **Token usage and cost:** input, output and total tokens per turn, so cost per question and per route can be tracked.
- **Latency:** per-step `elapsed_s` and total `duration_s`, to see whether time goes to the LLM, retrieval or SQL.
- **Quality signals:** `low_confidence`, `error`, citations removed by the check, empty or low-score retrievals.
- **In production:** ship the same events to a tracing backend (LangSmith, Langfuse or OpenTelemetry) instead of files,
  then dashboard token cost, p50/p95 latency, error rate, tool-call counts, retrieval hit rate and judge scores, with
  alerts on regressions. Redact or restrict access to inventory data in traces.

**Scaling, bigger PDFs:**
- *Many peers:* a document registry (company, doc_type, period, needs-OCR) replaces the hardcoded two; metadata filters;
  a `search_documents(query, companies, doc_type)` tool that fans out per peer (top 2-3 each) so one company cannot fill
  the top-k; a reranker. Beyond ~10 peers, extract metrics at ingestion into a verified table (company, metric, value,
  year, page, quote) so comparison becomes a query.
- *Images, tables, charts:* classify each page, send only image regions to a VLM, store tables as rows in DuckDB (vector
  hit → exact query), keep `page`, `bbox`, `image_path`, `extraction_method` in metadata so the UI can show the original crop.
- *Infrastructure:* server vector DB (Qdrant/pgvector) with per-client namespaces, layout-aware parsing, async ingestion.

**Scaling, bigger tables:** Parquet or a warehouse behind the same SQL tool, with a semantic layer of governed metrics.
**Cost and latency:** prompt caching, a small model for routing, cached deterministic tool outputs.

## 6. Ideas for improvement
Product: three modules around the chat, which stays the entry point for open questions and "why". Each module reuses the
existing deterministic tools, so numbers match everywhere.
1. **Data quality tab → actions.** Findings by principle and severity with evidence and the protocol page behind each,
   a recommended action, owner and status (open / in progress / resolved), and drafted data requests to site or supplier
   owners. A "must fix / disclose as limitation" view shows what blocks sign-off.
2. **Dashboard tab.** Scope totals (as reported and annualised), trends, top sites, activities and suppliers, with
   filters, drill-down to rows, and a warning badge where a quality finding affects a figure.
3. **Report module.** Template + period → draft with summary, charts, methodology, limitations and actions, exported as
   PDF/Word. Figures come from tools; the LLM writes only the narrative; limitations come from the quality findings.

Next steps behind them: reduction scenarios (renewable PPA, coal-to-gas) · human-confirmed activity→category mapping ·
factor library (eGRID, DEFRA, USEEIO) and market-based Scope 2 · disclosure mapping (CDP/CSRD) · intensity benchmarks and
sector-matched peers once revenue and volume data exist · a quality methodology designed with business input.
Technical items (reranking, per-peer fan-out, multimodal ingestion, document edit/delete) are in section 5.

## 7. Challenges and how I solved them
- **Quality assessment (biggest gap).** The brief does not say how to judge inventory quality and I had a knowledge gap
  on the GHG Protocol. For now I worked from an assumption proposed by the coding assistant: scan the protocol for what it
  requires ("shall") and recommends (chapter 7), and turn that into rule-based checks. It is a working scaffold, not a
  validated method, and not the focus here: designing it properly needs business insight.
- **Unreliable Scope 3 labels:** tested statistically, report both views, no silent reclassification.
- **Peer PDF with image tables:** one-time OCR, cached, indexed with the rest. The brief's model (`gemini-4-hydrogen`)
  was out of free-tier quota, so I ran the OCR with the built-in macOS Vision engine instead (local, free, no API
  calls). Gemini vision is still available with `--engine gemini`.
- **Gemini free tier (~20 requests/day/model):** `gemini-4-hydrogen` ran out of quota, so the answers and screenshots
  here were produced with **`gemini-3.5-flash-lite`** (Google AI Studio, free tier). The model is configurable, and fixed
  pipelines keep calls per question low.
- **Gemini tool-schema quirks and hanging requests:** string-literal enums and explicit timeouts.

---

# Setup

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
uv sync                          # create .venv, install dependencies
cp .env.example .env             # paste GEMINI_API_KEY into .env
uv run python -m src.kb          # build the knowledge-base index (~1 min, local embeddings)
uv run streamlit run app.py      # http://localhost:8501
```

- **CLI:** `uv run python -m src.graph "Which suppliers should I prioritise?"`
- **Tests:** `uv run pytest`
- **OCR (optional):** `data/derived/peer2_ocr.json` is committed; regenerate with `uv run python scripts/ocr_peer2.py`.
- **LLM:** Gemini via `GEMINI_API_KEY`. The model is set with `LLM_MODEL` (for example `LLM_MODEL=gemini-3.5-flash-lite`).
  If the free tier runs out, switch to another Gemini model. Check the setup with `uv run python scripts/check_llm.py`.

# Layout
```
app.py            Streamlit UI              src/graph.py      LangGraph router, pipelines, ReAct agent
src/data.py       CSV loading, data card    src/tools.py      LangChain tool wrappers
src/analytics.py  pandas + DuckDB tools     src/kb.py         PDF chunking, Chroma + BM25 search
src/quality.py    rule-based checks         src/grounding.py  findings → protocol passages, citation check
src/prompts.py    prompts                   src/llm.py, audit.py   LLM client, per-turn trace
scripts/          OCR, LLM check            notebooks/        data profiling        tests/  unit tests
```
