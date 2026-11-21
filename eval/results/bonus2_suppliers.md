# Which suppliers should I prioritise to engage for emissions reduction efforts?

_intent: data_question · 5.4s_

To prioritise suppliers for engagement, rank them by emissions volume, carbon intensity per USD spent, and data quality gaps. 

Emissions are quite evenly spread across your supplier base—ranging from **40,352.6 tCO2e** (10.8% of Scope 3) to **50,966.2 tCO2e** (13.6%)—meaning engagement cannot focus on just one or two outliers. 

### Prioritised Supplier Ranking

| Priority | Supplier | Share of Scope 3 | Emissions (tCO2e) | Intensity (kg/USD) | Low-Quality Data |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Corporate Travel | 13.4% | 50,296.3 | 1.66 | 58.2% |
| **2** | WasteManagement Co | 13.2% | 49,390.9 | 1.62 | 58.8% |
| **3** | OfficeMax | 13.6% | 50,966.2 | 1.53 | 60.4% |
| **4** | MetalSupply Co | 13.0% | 48,603.0 | 1.55 | 63.2% |
| **5** | SteelCorp Inc | 12.6% | 47,246.6 | 1.54 | 64.1% |
| **6** | TechEquip Corp | 11.9% | 44,445.1 | 1.46 | 60.6% |
| **7** | PackagePro Ltd | 11.5% | 43,184.7 | 1.51 | 55.6% |
| **8** | LogisticsPro | 10.8% | 40,352.6 | 1.51 | 59.6% |

---

### Caveats
* **Scope 3 data is 100% spend-based**, so emission figures reflect monetary spend and secondary intensity factors rather than physical quantities.
* **Severe supplier mapping issues:** 88% of Scope 3 rows pair a supplier with an implausible activity (e.g. *WasteManagement Co* supplying IT Hardware, or *Corporate Travel* supplying raw steel), meaning supplier names currently do not reflect actual goods or services purchased.

### Next Steps
1. **Fix supplier mapping:** Audit and clean your procurement database to correct supplier-to-activity assignments before launching formal engagement.
2. **Upgrade data quality:** Move away from generic spend-based emission factors by requesting primary, supplier-specific activity data or product carbon footprints from your top spend partners (*OfficeMax*, *Corporate Travel*, *WasteManagement Co*).
3. **Target intensity reductions:** Engage suppliers with above-median carbon intensities (such as *Corporate Travel* and *WasteManagement Co*) to discuss lower-carbon operational practices and set reduction targets.

---
### Trace
- **router**: `intent=data_question, scope=3`
- **tool_call**: `get_supplier_priority({})`
- **tool_result**: `get_supplier_priority: {"method": "score = 0.5*emissions + 0.3*intensity + 0.2*low-quality-data share (min-max normalised)", "spread_note": "top supplier 13.6% vs bottom 10.8% of Scope 3: emissions are evenly spread", "suppliers": [{"supplier": "Corporate Travel", "priority_score": 0.83, "co2e_tonne`
- **agent**: `answer generated`