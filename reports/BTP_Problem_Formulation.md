# Problem Formulation
## Cash-Flow Early Warning System for MSME Liquidity Risk

---

### 1. Problem Statement

A business can be solvent — its total assets worth more than what it owes — and still fail, because solvency and liquidity are not the same thing. Liquidity is narrower and more immediate: it is whether a business has enough cash on hand, *right now*, to meet the obligations that are due *right now*. A business can be fundamentally healthy on paper and still collapse because it cannot bridge a short-term gap between money owed to it and money it owes.

This gap is especially acute for micro, small and medium enterprises (MSMEs). Unlike large firms, MSMEs typically operate with thin cash buffers, sell heavily on credit with irregular and often-delayed repayment, and rarely have a dedicated finance function actively monitoring cash position on a rolling basis. Delayed receivables, upcoming supplier payments, inventory purchases that tie up working capital, and volatile month-to-month revenue combine to create liquidity stress that is often invisible until it is already a crisis — even when the underlying business is otherwise performing well.

The core problem this project addresses is therefore: **a business can appear financially healthy by conventional indicators while still heading toward an imminent cash shortfall it has no structured way to foresee.** An early-warning system that flags this risk with sufficient lead time gives a business owner room to act — collecting an overdue payment sooner, delaying a discretionary purchase, or arranging short-term financing — before the shortfall becomes unavoidable. The value of such a system lies specifically in *how early* and *how reliably* it can raise that flag, particularly for businesses that, being small, often have limited historical financial data for a model to learn from.

---

### 2. Precise Definition of Liquidity Stress

For this project, liquidity stress is defined operationally, not descriptively, so that it can serve as a trainable and measurable target:

> **Liquidity stress** occurs when a business's forecasted cash balance is predicted to fall below a business-specific minimum operating buffer at any point within a defined future horizon (e.g., the next 30 days).

The operating buffer is business-specific rather than a fixed value, since minimum viable cash reserves differ meaningfully by business size, expense structure, and sector. This definition deliberately keeps the target short-term and transaction-grounded, in contrast to the one-year-horizon, firm-level bankruptcy/default definitions used throughout the financial-distress literature reviewed for this project (e.g., Cheraghali & Molnár, 2025, define default via Chapter 7/11 filings over a one-year horizon).

---

### 3. Research Questions

This project is organized around two research questions, in addition to the core system-building objective:

**RQ1 — Detection horizon:** How many days in advance can transaction-level data reliably signal an approaching liquidity-stress event, and at what precision?

**RQ2 — Robustness under data scarcity:** How does prediction reliability degrade as the amount of available transaction history for a business shrinks — from a full year, down to a few months, down to only a few weeks?

A supporting objective, tied to both questions, is explainability: for any flagged risk, identifying which factors (e.g., overdue receivables, upcoming payables, revenue decline) are driving the prediction, so the output is actionable rather than a bare probability.

---

### 4. Positioning Against Existing Work

A review of eight papers spanning SME financial-distress prediction, cash-flow forecasting, and transaction-level financial modeling shows that no existing work combines all four elements of this project's approach.

Closest in **objective** is a recent SME financial-management system (2026) that combines accounts-receivable delay prediction with modular cash-flow forecasting, and which tests one sparse-data scenario — forecasting eleven months of cash flow from a single month of history. This project extends that single data point into a systematic study across multiple history lengths, and adds a unified liquidity-risk score with an explicit lead-time analysis, neither of which the existing system attempts.

Closest in **data granularity** is Kotios et al. (2022), which works directly with individual SME banking transactions — over 3.5 million records — before aggregating them for cash-flow forecasting. However, its objective is transaction categorization and aggregate forecasting, not an explicit, explained liquidity-stress classification.

Several papers (Medianovskyi et al., 2023; Cheraghali & Molnár, 2025; Qi, 2025) independently establish that gradient-boosted models — particularly LightGBM, XGBoost, and CatBoost — combined with SHAP-based explainability, are the strongest and most interpretable approach for SME financial-risk prediction. This finding directly motivates the modeling approach adopted here. All three, however, operate at firm-level using annual or quarterly financial ratios, predicting distress over a roughly one-year horizon, rather than short-term liquidity stress from day-to-day transaction behavior.

A broader survey of the field (Zhao, Ouenniche & De Smedt, 2024) independently identifies improved interpretability and better handling of imbalanced, heterogeneous data as open future-research directions — both of which this project addresses directly through its SHAP-based explanation layer and its explicit study of performance under sparse, limited-history conditions.

The gap this project fills, precisely: no reviewed work combines (1) transaction-level, day-to-day input data, (2) a unified liquidity-stress risk score rather than separate forecast and classification outputs, (3) an explicit measurement of achievable detection lead time, and (4) a systematic study of how reliability degrades as available history shrinks.

---

### 5. Approach (Summary)

Semester 1 establishes the forecasting foundation: a transaction-to-daily-panel data pipeline, a synthetic data generator calibrated against real public MSME datasets, and a comparison of baseline and gradient-boosted (XGBoost/LightGBM) forecasting models, evaluated via walk-forward validation. Semester 2 builds on this foundation to address RQ1 and RQ2 directly — the early-detection-horizon study and the data-scarcity robustness study — alongside SHAP-based explainability and, as a stretch goal, decision-support scenario simulation. Full scope and week-by-week planning are detailed separately in the Semester 1 project plan.
