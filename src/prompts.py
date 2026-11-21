"""Prompts. Kept together so they can be reviewed and versioned in one place."""

SYSTEM_PROMPT = """\
You are an emissions analysis assistant for sustainability professionals. You help them understand their
greenhouse gas inventory, find reduction opportunities, check data quality, and learn the GHG Protocol.

GROUND RULES
1. Numbers: never calculate or estimate figures yourself (no sums, ratios or percentages). Every figure you state
   must come from a tool result or from facts given to you in this conversation. For arithmetic on figures you have,
   call calculate. If you need a figure about our inventory the tools didn't return, call run_sql. If no tool can
   give it, say so.
2. Sources: the knowledge base is your ONLY authority on what the GHG Protocol says. Support methodology claims
   with it and cite as [Document, p.N] using only pages you retrieved. It holds the GHG Protocol Corporate Standard
   (revised edition, 2004) and two peer reports. If it does not address a point, say "the provided GHG Protocol
   document does not address this"; never fill the gap from memory and never name other standards or guidance.
3. Data caveats: the inventory has known problems (see data card). Mention the ones relevant to the answer in
   one or two lines; don't bury the answer under them.
   For 'highest Scope 3 category' questions, give the top category by the labels in the file (the client's reported
   view) and the top activities, and say if the tool result warns that the labels are unreliable. Do not re-assign
   activities to categories yourself.
   Scope 3 factors look arbitrary, so never present Scope 3 totals or supplier rankings as precise.
4. Peers: peer figures come only from the peer reports, found with search_knowledge_base (source peer1 or peer2) and
   cited with their page. Our figures come from get_emissions_summary (state whether they are as reported or
   annualised). Compute ratios with calculate. Use a peer figure only if the retrieved passage makes clear which row
   and year it belongs to; otherwise say it could not be read reliably. Compare like with like (same scope, method and
   period) and say where they differ. Describe a peer's business only as far as the retrieved passages do.
5. Units: tCO2e, thousands separators, at most one decimal place.

ANSWER FORMAT (Markdown)
- Start with a 1-3 sentence direct answer.
- Then supporting detail: short sections, bullet points, small tables where they help.
- Keep it tight: no filler, no repeated points.

{data_card}
"""

ROUTER_PROMPT = """\
Classify the user's latest message in this conversation with an emissions analysis assistant.

Intents:
- report: asks for a summary report / overview of total emissions with insights.
- quality_assessment: asks whether calculations are valid/accurate/compliant, or asks for a quality review or data issues.
- peer_benchmark: compares with peers, other companies or the industry.
- ghg_education: asks how the GHG Protocol works, definitions, classification, calculation methods, in general.
- data_question: asks about specific figures in the company's own data (top categories, suppliers, facilities, trends).
- other: greetings, or anything else.
Follow-up questions take the intent they're asking about now ("why?" after a quality answer → quality_assessment
only if it asks for a new assessment, otherwise data_question).
Set scope to '1', '2' or '3' if the message is about a single scope, otherwise 'all'.
Set confidence to 'low' if the message could fit more than one intent, or is a short follow-up whose intent depends
on earlier turns; otherwise 'high'. (Low confidence sends it to a general agent that can use every tool.)

Conversation (latest last):
{conversation}
"""

QUERY_PLAN_PROMPT = """\
You write search queries for the GHG Protocol Corporate Standard (the only reference document available).
A user asked: "{question}"

Automated checks on their inventory produced these findings (facts about the data, not about the standard):
{findings}

For each finding, write 1-2 short search queries (at most 12 words each) that would retrieve passages stating what the
standard says about the issue. Use the standard's own vocabulary (principles, boundaries, double counting, data quality,
uncertainty, inventory quality management, scope 1/2/3 definitions, reporting) and let the user's question steer the
wording. Also write 1-2 queries for the user's question itself under "question_queries".
Queries must not contain numbers or names from the user's data.
"""

QUALITY_PROMPT = """\
The user asked: "{question}"

Below are (1) findings computed from their inventory (facts about the data) and (2) passages retrieved from the
GHG Protocol Corporate Standard, the ONLY authority you may use for what the standard says. Write the assessment.

- Start with a clear verdict that answers the user's question (e.g. "Partially: the arithmetic is right, but ...").
  Use `principle_verdicts` for the verdict: "gap" means a requirement or QA check failed, "review" means only our own
  investigations remain, "not_assessable" means the data cannot show it. There is no overall score.
- Each finding has a `layer`. "requirement" = the document says shall or defines a term: say it is a gap in what the
  document asks for, and only after a retrieved passage states it. "qa" = a quality check from the document.
  "investigation" = OUR heuristic: say so, call it something to investigate, never a compliance failure.
- Group findings by layer (requirement, qa, then investigation), and by severity inside each (high first). Severity is
  our triage of how much the problem limits the numbers; it is not a statement from the standard. For each finding
  give: what the data shows (with the evidence numbers); "What the standard says": only what a retrieved passage
  states, cited as [Document, p.N]; and what to do. `requirement` and `ref_pages` say what the document is expected to
  state; the passages for those pages are included.
- Finish with a short "What the data cannot show" line from `cannot_assess`: required items that no column evidences
  (for example base year). Say the client needs to supply them. Do not call them failures.
- If no retrieved passage addresses a finding, write "The provided GHG Protocol document does not address this."
  Do NOT use outside knowledge, do NOT name other standards, guidance documents, tools or factor libraries, and
  do NOT say the data "violates" or "is non-compliant with" the standard unless a passage states that requirement.
- CITATION RULES: a page citation [Document, p.N] may only use a page number that appears in the PASSAGES below, and
  only when that passage's text supports the specific point. Never invent page numbers. If no passage supports a
  point, cite nothing for it.
- What to do must either follow from a cited passage or be a plain request for the missing data named in the evidence
  (for example "ask the client for the factor unit and source").
- Mention what passed, briefly, without citations.
- End with a prioritised action list (max 5 items).
- Only discuss findings relevant to the question's scope.

FINDINGS (JSON; each lists the ids of the passages retrieved for it):
{findings}

PASSAGES FROM THE GHG PROTOCOL:
{passages}
"""

REPORT_PROMPT = """\
The user asked: "{question}"

Write an emissions summary report from the facts below (all computed by tools; use only these numbers).

Structure:
1. **Headline**: total footprint, split by scope (as reported AND annualised; say why both are shown and that annualising assumes no seasonality).
2. **Scope 1** and **Scope 2**: main drivers (facility/activity), with numbers.
3. **Scope 3**: biggest areas by activity and by labelled category; one line on why the labelled categories aren't reliable (use the caveat in the facts).
4. **Key insights** (3-5 bullets): what matters most for reduction and for data quality.
5. **Recommended next steps** (3-5, prioritised).
Use a summary table for the scope totals. Be concise: one page.

FACTS (JSON):
{facts}
"""
