# Should employee business travel be classified as Scope 1 or Scope 3? Explain the reasoning and describe how I can calculate my business travel emissions?

_intent: ghg_education · 5.3s_

Employee business travel should be classified as **Scope 3**, specifically under Category 6 (Employee Business Travel) of the GHG Protocol Corporate Standard. 

### Reasoning
- **Ownership and Control:** Scope 1 covers direct emissions from sources that are owned or controlled by the reporting company (such as company-owned fleet vehicles or facilities) [GHG Protocol Corporate Standard, p.27]. Employee business travel typically involves commercial airlines, rental cars, or employee-owned vehicles where the company does not own or directly control the asset. 
- **Value Chain Indirect Impact:** Because these emissions are a consequence of company activities but occur from sources owned or controlled by third parties (e.g., airlines or hotel chains), they fall under **Scope 3: Other indirect GHG emissions** [GHG Protocol Corporate Standard, p.27, p.31].
*(Note: If a vehicle is leased and falls under the company's financial or operational control boundary, those specific vehicle emissions may be reported under Scope 1 [GHG Protocol Corporate Standard, p.30-33].)*

### How to Calculate Business Travel Emissions
According to GHG Protocol guidance (outside provided documents), calculating business travel generally follows two primary methods:

1. **Fuel-Based Method:**
   - Collect primary fuel consumption data from rental car agencies or reimbursed employee-owned vehicle mileage logs.
   - Multiply the fuel consumed (or distance travelled by vehicle type) by appropriate combustion emission factors.

2. **Distance-Based Method:**
   - Collect activity data on travel distance (e.g., passenger-kilometers or passenger-miles broken down by mode of transport: air travel by short/long-haul, rail, or taxi).
   - Multiply the distance travelled by relevant secondary emission factors (such as DEFRA or EPA passenger transport emission factors).

3. **Spend-Based Method (Alternative):**
   - If distance or fuel data is unavailable, multiply financial expenditure on travel (e.g., hotel stays or airfare spend) by environmentally extended input-output (EEIO) emission factors per currency unit (similar to how the inventory's current Scope 3 spend-based data is structured).

---
### Next Steps
- Review current expense reports or travel management system (TMS) data to extract disaggregated activity data (such as flight classes and distances or hotel nights) rather than relying purely on financial spend.
- Apply recognized emission factor databases (e.g., DEFRA conversion factors) to calculate accurate tCO2e totals for each trip type.

---
### Trace
- **router**: `intent=ghg_education, scope=3`
- **tool_call**: `search_knowledge_base({"source": "ghg_protocol", "query": "business travel scope 3 scope 1 employee-owned vehicles leased vehicles"})`
- **tool_result**: `search_knowledge_base: [{"cite": "[GHG Protocol Corporate Accounting and Reporting Standard (Revised Edition), p.32]", "text": "s. Each partner is required, as an element\nof the subcontract payment scheme, to enter data on vehicles used,\ndistance traveled, fuel efficiency, and background data. Thi`
- **agent**: `answer generated`