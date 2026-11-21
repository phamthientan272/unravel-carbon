# Generate a summary report of our total emissions by scope with key insights

_intent: report · 4.6s_

### 1. Headline

The company’s total greenhouse gas inventory is **378,683.0 tCO₂e** as reported, dominated by Scope 3 emissions (**98.9%**). 

Because Scope 1 and Scope 2 cover only Q1 2024 while Scope 3 covers the full year 2024, both as-reported and annualised figures are shown below to enable fair annual comparisons and highlight the difference caused by mismatched reporting periods.

| Scope | As Reported (tCO₂e) | Share (%) | Annualised Estimate (tCO₂e) | Annualised Share (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Scope 1** | 1,692.5 | 0.4% | 6,769.9 | 1.7% |
| **Scope 2 (Location-based)** | 2,505.0 | 0.7% | 10,020.2 | 2.6% |
| **Scope 3** | 374,485.5 | 98.9% | 374,485.5 | 95.7% |
| **Total** | **378,683.0** | **100.0%** | **391,275.5** | **100.0%** |

---

### 2. Scope 1 and Scope 2

* **Scope 1 (1,692.5 tCO₂e):** Driven primarily by the **Manufacturing Plant** (94.4% or 1,597.4 tCO₂e). By activity, **Stationary Combustion** accounts for 53.3% (902.6 tCO₂e) and **Process Emissions** for 41.5% (702.0 tCO₂e).
* **Scope 2 (2,505.0 tCO₂e):** Also concentrated at the **Manufacturing Plant** (84.6% or 2,118.6 tCO₂e). By energy type, **Steam** represents 84.5% (2,116.4 tCO₂e) and **Electricity** 15.5% (388.6 tCO₂e).

---

### 3. Scope 3

Using the reclassified view by activity, Scope 3 is led by **Purchased goods and services** at 133,282.7 tCO₂e (35.6%), followed by **Capital goods** at 82,814.0 tCO₂e (22.1%) and **Business travel** at 46,312.1 tCO₂e (12.4%). 

*Note: Raw category labels are unreliable because 92% of rows carry a category label inconsistent with their underlying activity (for instance, top-level raw labels incorrectly point to Category 3 as the largest at 31,794.3 tCO₂e).*

---

### 4. Key Insights

* **Scope dominance:** Scope 3 comprises nearly 96% to 99% of the inventory, making supply chain engagement essential for any meaningful reduction strategy.
* **Facility hotspots:** Direct operational emissions (Scopes 1 and 2) are heavily concentrated at the Manufacturing Plant, specifically via stationary combustion, process emissions, and steam consumption.
* **Data quality barrier:** 92% of Scope 3 category labels and 88% of supplier-activity pairings contain contradictions, requiring data mapping clean-up before setting reduction targets.
* **Missing market-based view:** Scope 2 is reported solely on a location basis, leaving renewable energy purchases unreflected in the figures despite recorded renewable percentages.

---

### 5. Recommended Next Steps

1. **Clean Scope 3 mapping:** Fix supplier and activity mismatches to resolve the 88% error rate and ensure reliable baseline categories.
2. **Engage top supply chain partners:** Target key suppliers (such as Corporate Travel, WasteManagement Co, OfficeMax, and MetalSupply Co) to replace 100% spend-based proxy data with primary, supplier-specific carbon data.
3. **Investigate Manufacturing Plant hotspots:** Audit stationary combustion, process emissions, and steam use at the manufacturing facility to identify direct reduction opportunities.
4. **Implement Scope 2 dual reporting:** Incorporate market-based Scope 2 accounting to properly account for recorded renewable energy percentages.
5. **Align reporting periods:** Standardise all scopes to a full 12-month calendar year in future inventories.

---

### Caveats
* Scope 1 and 2 cover Jan–Mar 2024 only, while Scope 3 covers Jan–Dec 2024; totals mix periods.
* Scope 3 is 100% spend-based with significant label and supplier mismatches.

---
### Trace
- **router**: `intent=report, scope=all`
- **report**: `answer generated`