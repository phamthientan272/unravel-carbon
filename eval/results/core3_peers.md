# How do my scope 1 & 2 emissions compare with other companies in my industry (refer to the peer reports), and what insights can I derive from this comparison?

_intent: peer_benchmark · 5.6s_

Our annualised Scope 1 and 2 emissions (16,790.1 tCO2e) are significantly higher than both DocuSign (8,994.0 tCO2e) and Shopify (679.0 tCO2e), driven entirely by our heavy manufacturing operations rather than typical office-based software activities [compare_with_peers]. 

### Comparison & Peer Context

* **Scope 1 (Direct):** Our annualised Scope 1 is 6,769.9 tCO2e—about 15 times DocuSign's (449.0 tCO2e) and nearly 19 times Shopify's (360.0 tCO2e). Our main drivers are process emissions (2,808.0 tCO2e) and stationary combustion (3,582.0 tCO2e) at our Manufacturing Plant.
* **Scope 2 (Electricity/Steam, Location-Based):** Our annualised Scope 2 is 10,020.2 tCO2e. While comparable to DocuSign's location-based figure (8,545.0 tCO2e), it is over 31 times higher than Shopify's (319.0 tCO2e), largely due to heavy steam (7,548.0 tCO2e) and electricity (926.0 tCO2e) consumption at the Manufacturing Plant [compare_with_peers].
* **Decarbonization Maturity:** Both peers rely heavily on renewable energy certificates (RECs) and carbon removal to achieve near-zero or offset operational footprints (e.g., DocuSign purchases 100% renewable electricity; Shopify achieves net-zero operational emissions via RECs and carbon removal) [compare_with_peers, peer1_emissions_report.pdf p.49, peer2_emissions_report.pdf p.18].

### Insights

* **Activity Misalignment:** Direct comparisons of absolute totals or basic ratios are misleading because DocuSign and Shopify operate as asset-light software/SaaS companies, whereas our footprint includes an energy-intensive manufacturing facility [compare_with_peers]. 
* **Data & Disclosure Gap:** Unlike DocuSign, which follows GHG Protocol dual-reporting by disclosing both location- and market-based Scope 2 figures alongside a comprehensive Scope 3 inventory [peer1_emissions_report.pdf p.50], our inventory currently reports only location-based figures and relies entirely on spend-based Scope 3 estimations.

### Caveats & Next Steps
* **Caveats:** Our figures are annualised from a 3-month sample (Jan–Mar 2024), which may introduce seasonal bias. Furthermore, peer comparisons lack intensity normalisation (such as tCO2e per $M revenue or per employee) because our company's revenue and headcount data are not yet integrated into the inventory.
* **Next Steps:**
  1. Calculate emissions intensity metrics (tCO2e per $1M revenue and per employee) to enable a fair, normalized benchmark against industry peers.
  2. Implement dual-reporting for Scope 2 by accounting for contractual renewable energy purchases (market-based), mirroring DocuSign's disclosure standard [peer1_emissions_report.pdf p.50].
  3. Prioritize decarbonization interventions at the Manufacturing Plant, specifically targeting stationary combustion, steam efficiency, and process emissions.

---
### Trace
- **router**: `intent=peer_benchmark, scope=all`
- **tool_call**: `compare_with_peers({})`
- **tool_result**: `compare_with_peers: {"comparison_table": [{"company": "Our company (annualised from Q1 2024)", "industry": "Manufacturing (coal boilers, cement process, fleet)", "scope1_tco2e": 6769.9, "scope2_location_based_tco2e": 10020.2, "scope2_market_based_tco2e": null, "scope1_plus_2_tco2e": 16790.1, "scope3`
- **agent**: `answer generated`