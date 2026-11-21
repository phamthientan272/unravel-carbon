# What is our highest emitting Scope 3 category and what specific activities contribute to it?

_intent: data_question · 4.4s_

Your highest emitting Scope 3 category depends on whether you look at the raw category labels in the data or the GHG Protocol reclassified view. Under the raw labels, **Category 3 (Fuel- and energy-related activities)** is the highest at 31,794.3 tCO2e; however, because 92% of category labels are inconsistent with their underlying activities, the **reclassified view** is the correct basis for decisions, placing **Category 1 (Purchased goods and services)** highest at **133,282.7 tCO2e** (35.6% of Scope 3).

### Reclassified Category Breakdown
When activities are correctly mapped to GHG Protocol categories, Category 1 comprises four main activities:

| Activity Description | Emissions (tCO2e) | Share of Category 1 | Spend (USD) |
| :--- | :--- | :--- | :--- |
| **Raw Materials - Steel** | 37,511.9 | 28.1% | $22,702,705.7 |
| **Office Supplies** | 34,799.2 | 26.1% | $21,539,361.2 |
| **Packaging Materials** | 33,412.0 | 25.1% | $21,433,187.4 |
| **Raw Materials - Aluminum** | 27,559.6 | 20.7% | $17,848,897.7 |

### Why the Two Views Differ
The raw category labels in the inventory are 92% inconsistent with the activities they describe (e.g., air travel is frequently mislabelled under Category 11). The reclassified view sorts rows by their actual activity descriptions to align with GHG Protocol definitions.

### Caveats
- Scope 3 covers Jan-Dec 2024 (374,485.5 tCO2e total), whereas Scope 1 and 2 cover Jan-Mar 2024 only.
- Scope 3 is 100% spend-based, and 88% of rows pair a supplier with an implausible activity (e.g., 'Corporate Travel' supplying trucking).

### Next Steps
- Prioritize supplier engagement with high-spend suppliers in Category 1, such as *MetalSupply Co*, *LogisticsPro*, and *SteelCorp Inc*.
- Transition from spend-based emission factors to primary activity data (weight, volume, or supplier-specific footprints) for steel, aluminum, and packaging materials.

---
### Trace
- **router**: `intent=data_question, scope=3`
- **tool_call**: `get_top_scope3({"by": "category", "top_n": 5})`
- **tool_call**: `get_top_scope3({"by": "reclassified_category", "top_n": 5})`
- **tool_result**: `get_top_scope3: {"ranked_by": "category", "scope3_total_tco2e": 374485.5, "groups_total": 15, "top": [{"category_num": 3, "category_name": "Fuel- and energy-related activities", "co2e_tonnes": 31794.3, "share_pct": 8.5, "spend_usd": 20293884.0, "rows": 75, "top_activities": [{"activity_description":`
- **tool_result**: `get_top_scope3: {"ranked_by": "reclassified_category", "scope3_total_tco2e": 374485.5, "groups_total": 6, "top": [{"expected_category_num": 1, "expected_category_name": "Purchased goods and services", "co2e_tonnes": 133282.7, "share_pct": 35.6, "spend_usd": 83524152.0, "rows": 349, "top_activities":`
- **agent**: `answer generated`