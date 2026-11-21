# How much Scope 3 emissions came from IT Hardware purchased in Q3 2024?

_intent: data_question · 5.5s_

In Q3 2024, Scope 3 emissions from purchased IT Hardware were 13,884.3 tCO2e.

### Supporting Detail
- **Activity Description:** IT Hardware
- **Quarter:** 2024Q3
- **Emissions:** 13,884.3 tCO2e

### Caveats
- Scope 3 emissions are calculated entirely using spend-based data.
- In the inventory, 92% of Scope 3 rows have a category label inconsistent with the activity, and 88% pair suppliers with implausible activities; activity descriptions provide a more reliable view than raw category labels.

---
### Trace
- **router**: `intent=data_question, scope=3`
- **tool_call**: `run_sql({"query": "SELECT sum(co2e_tonnes) AS total_co2e FROM scope3 WHERE quarter = '2024Q3' AND (activity_description ILIKE '%IT Hardware%' OR activity_description ILIike '%hardware%')"})`
- **tool_result**: `run_sql: {"error": "ParserException: Parser Error: syntax error at or near \"ILIike\"\n\nLINE 1: ..._description ILIKE '%IT Hardware%' OR activity_description ILIike '%hardware%')\n                                                                      ^^^^^^"}`
- **tool_call**: `run_sql({"query": "SELECT sum(co2e_tonnes) AS total_co2e FROM scope3 WHERE quarter = '2024Q3' AND (activity_description ILIKE '%IT Hardware%' OR activity_description ILIKE '%hardware%')"})`
- **tool_result**: `run_sql: {"row_count": 1, "truncated": false, "rows": [{"total_co2e": 13884.25}]}`
- **tool_call**: `run_sql({"query": "SELECT distinct activity_description FROM scope3 WHERE activity_description ILIKE '%hardware%' OR activity_description ILIKE '%IT%'"})`
- **tool_result**: `run_sql: {"row_count": 1, "truncated": false, "rows": [{"activity_description": "IT Hardware"}]}`
- **agent**: `answer generated`