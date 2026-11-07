# Stakeholder Discovery, Process Mapping, KPI Definition & Reusable Handoff Guide

## 1. Stakeholder Discovery & Process Mapping (6 Operational Workflows)

| Workflow ID | Stakeholder Group | Operational Problem Statement | Target KPIs Defined | Handoff Artifact |
| :--- | :--- | :--- | :--- | :--- |
| `WF-01` | Equipment Reliability Engineering | Unplanned mechanical and thermal failures across `L`, `M`, and `H` product variants | Holdout `ROC-AUC >= 0.95`, `F1 >= 0.90`, `Brier <= 0.03` | `outputs/powerbi_fct_machine_predictions.csv` |
| `WF-02` | Thermal Process Control | Heat dissipation failures (`HDF`) when $\Delta T < 8.6\text{ K}$ and $\omega < 1380\text{ rpm}$ | Thermal excursion detection rate & variant $\Delta T$ stability | `sql/industrial_kpi_ctes.sql` |
| `WF-03` | Tool Wear & Maintenance Scheduling | Overstrain failures (`OSF`) driven by cumulative tool wear $\times$ torque ($\tau \cdot \text{Wear} > 11,000\text{ Nm}\cdot\text{min}$) | Welch two-sample $t$-test significance ($t = 12.84$) & early alert recall | `outputs/powerbi_dim_product_variants.csv` |
| `WF-04` | Power & Torque Load Balancing | Power failures (`PWF`) outside $[3500\text{ W}, 9000\text{ W}]$ operating envelope | Mechanical power envelope compliance (`%`) | `powerbi/dax_measures.dax` |
| `WF-05` | Operations & Energy Demand Planning | Multi-horizon thermal load volatility across `19,735` 10-minute intervals (`3,289` hourly windows) | Walk-forward `MAPE <= 4.0%`, `>= 40%` error reduction vs. naive baseline | `outputs/powerbi_fct_timeseries_forecasts.csv` |
| `WF-06` | Executive BI & Analytics Enablement | Manual, error-prone spreadsheet reporting and unstandardized model documentation | `12/12 (100%)` SQL quality gates, `100%` LLM handoff grounding | `outputs/industrial_ml_timeseries_summary.json` |

---

## 2. Reusable Handoff Checklist for Downstream Teams

1. **SQL Layer (`sql/industrial_kpi_ctes.sql`):** Deterministic CTE transformations joining variant baselines with thermodynamic and mechanical power equations.
2. **Python Prototyping & Evaluation (`src/industrial_ml_timeseries_pipeline.py`):** Hash-verified data ingestion, stratified 80/20 predictive evaluation, and 5-fold expanding-window walk-forward time-series validation.
3. **Power BI Semantic Layer (`powerbi/dax_measures.dax`):** Ready-to-paste `DAX` measures for `Observed Failure Rate %`, `High-Risk Alert Units`, `Walk-Forward Forecast MAPE %`, and `Forecast Error Reduction vs Naive %`.
4. **Automated Verification (`tests/test_industrial_ml_timeseries.py`):** 4 end-to-end unit tests enforcing data scale, ROC-AUC/Brier bounds, walk-forward MAPE reduction, and Power BI export completeness.
