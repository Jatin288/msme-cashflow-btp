# MSME Cash-Flow BTP — Complete Project Context

> **Last updated:** October 1, 2026  
> **Purpose:** Single source of truth synthesized from the full conversation history. Read this before every session.

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Full Title** | Cash-Flow Early Warning System for MSME Liquidity Risk |
| **Institution** | IIIT Delhi (IIITD) |
| **Advisor** | Prof. Pankaj Vajpayee |
| **Type** | B.Tech Project (BTP) — Thesis track |
| **Student** | Jatin Pawar (GitHub: `Jatin288`) |
| **Working alone** | Yes — solo project, scope adjusted accordingly |
| **Registration term** | Semester 5 |
| **GitHub repo** | `https://github.com/Jatin288/msme-cashflow-btp` |
| **Local path** | `C:\Users\Jatin Pawar\msme-cashflow-btp\` |

---

## 2. The Core Idea (One Paragraph)

Small businesses in India do not fail because they are unprofitable. They fail because money comes in late and goes out on time. A shop can be profitable on paper while cash is stuck in uncollected receivables, with supplier payments due tomorrow. The project builds a system that takes a business's past transaction history — sales, payments received, bills paid — and predicts their cash position several weeks ahead, flags when a shortfall is likely, explains *why* (overdue receivables? upcoming payables? falling revenue?), and optionally simulates what the business could do to avoid it. The scientific contribution is not the system itself but two specific, measurable research questions embedded inside it.

---

## 3. The Two Research Questions (RQ1 and RQ2)

These are the core academic contribution — everything else is infrastructure.

**RQ1 — Early-Detection Horizon:**  
*How many days before a liquidity-stress event can the system reliably flag it, and at what precision?*  
Target result format: "The model detects X% of stress events at least Y days before the event, at Z% precision."

**RQ2 — Robustness Under Data Scarcity:**  
*How does prediction reliability degrade as the available transaction history shrinks?*  
Target result format: An accuracy-vs-history-length curve (12 months to 6 months to 3 months to 6 weeks), showing where the system becomes unreliable. This is the finding that addresses the real-world MSME constraint that most small businesses don't have years of clean financial records.

---

## 4. Precise Definition of Liquidity Stress

> **Liquidity stress** = the predicted cash balance falling below the business's minimum operating buffer within a defined horizon (e.g., 30 days).

- Cash balance is computed as the running sum of all inflows minus outflows — it is never estimated directly, it *emerges* from the transaction log, same as a real bank account.
- The minimum operating buffer is a business-specific parameter (not a universal constant).
- "Stress event" = the first day the forecast crosses below this threshold.

---

## 5. How the Economics Grounds This

For when Prof. Vajpayee asks about theoretical foundation:

- **Solvency vs. Liquidity** (core concept): A business is *solvent* if total assets exceed total liabilities. A business is *liquid* only if it has enough cash *right now* to meet obligations *right now*. These can diverge — a profitable, solvent business can still become illiquid. This gap is the phenomenon being predicted.
- **Cash Conversion Cycle** (CCC): The number of days cash is tied up before it returns — time inventory sits unsold + time customers take to pay minus time the business can delay paying its own suppliers. The features chosen (receivables, payables, inventory spend) are the direct components of the CCC.
- **Keynesian Precautionary Motive**: Why businesses hold a cash buffer at all — to survive unexpected gaps between money in and money out. The risk threshold in the model is a data-driven estimate of this precautionary buffer, not a round-number guess.
- **Trade Credit**: Larger buyers systematically delay payment to smaller suppliers because of negotiating power — a known Indian MSME dynamic, relevant to why payment delay is a structural problem, not just individual behavior.

---

## 6. How the CS/ML Grounds This

For when Prof. Vajpayee asks about technical depth:

- **Monte Carlo** — used at the decision-support layer: simulates thousands of possible cash-flow futures with different random payment-timing draws, to estimate how much an intervention (e.g., collecting a late payment 10 days sooner) changes the probability of shortfall.
- **Shapley Values / SHAP** — from cooperative game theory: decomposes any individual risk score into per-feature contributions ("receivables added +24% to the risk score; upcoming payables added +18%"). Implemented via the `shap` Python library on trained XGBoost/LightGBM models.
- **Walk-Forward Validation** — the correct way to evaluate time-series models: train on months 1–9, test on month 10; retrain on 1–10, test on 11; etc. Randomly splitting time-series into train/test leaks future information into training (data leakage), which is the most common mistake in this space.
- **Class Imbalance** — most days for most businesses are not heading into a crisis, so a naive classifier gets high accuracy by always predicting "no stress." Correct evaluation uses precision/recall/PR-AUC, not accuracy. Class weighting or threshold tuning preferred over SMOTE-style oversampling (risky on time series).
- **Survival Analysis Framing** — the early-detection-horizon study is structurally equivalent to survival analysis: how long before the "event" (liquidity crisis) did the model first cross the risk threshold?
- **KS-Test (Kolmogorov-Smirnov)** — used to validate that the synthetic data's distributions are statistically indistinguishable from the real data, so the generator is calibrated, not invented.

---

## 7. Two-Semester Technical Roadmap (Updated Oct 2026)

### Strategic Architecture: "Predictive Platform" (Sem 1) ➔ "Prescriptive Execution" (Sem 2)
The project avoids artificial elongation by establishing a clean separation between prediction and action:
- **Semester 1 (Weeks 1–18):** Complete Predictive Foundation + Early Warning Engine + Empirical Studies (RQ1/RQ2) + **Working Web Application**.
  *Rationale:* Evaluators grade what is *visible*. An interactive web application (CSV upload, cash runway graphs, risk alert badges, SHAP root-cause breakdown) makes the engineering and research undeniable.
- **Semester 2 (Weeks 19–36):** Open-Ended Prescriptive Action Layer. Natural progression: from predicting a cash crunch to mathematically recommending *how to prevent it* (Counterfactual What-If Simulation, TReDS invoice discounting optimizer, or Financial Copilot). To be finalized with Prof. Vajpayee following the Sem 1 live demo.

### Semester 1 (Aug 15 – Dec 15, 2026): The Predictive Early Warning Platform

| Phase | Weeks | Work | Status |
|---|---|---|---|
| Literature + problem formulation | 1–2 | 8 papers read, synthesis, problem formulation doc written | DONE |
| Data pipeline | 3–4 | Daily cash-flow panel from raw invoices, feature engineering | DONE |
| Synthetic data + KS validation | 4–5 | Calibrated generator, two-part payment model, KS tests | DONE |
| Walk-forward validation + baselines | 5 | 4 baselines, expanding-window eval, MAE/RMSE/sMAPE | DONE |
| **ARIMA evaluation (0,1,0 random walk)** | **6** | **2,860 ARIMA fits across 20 businesses; proves univariate limits** | **DONE** |
| Multivariate ML Models (LightGBM/XGBoost) | 7–9 | Train gradient boosting on 22 rolling features; multi-horizon forecast | NEXT |
| Stress Classifier & Empirical Studies (RQ1/RQ2) | 10–11 | Define liquidity threshold, train classifier, lead-time curve (RQ1), scarcity curve (RQ2) | Pending |
| Explainability (TreeSHAP) | 12 | Decompose risk scores into human-readable driver cards | Pending |
| Working Web Application (FastAPI + Dashboard) | 13–15 | Interactive UI: CSV upload, cash runway chart, risk alert banner, SHAP cards | Pending |
| Integration, Demo Polish & Defense Prep | 16–18 | End-to-end testing, BTP-1 interim report, live defense demo rehearsal | Pending |

**Semester 1 deliverable:** A working, visible Web Application + Comprehensive Interim Thesis Report with full benchmark metrics (Baselines vs ARIMA vs LightGBM) and RQ1/RQ2 empirical curves.

### Semester 2 (Jan – May 2027): Prescriptive Actions & System Optimization (Open-Ended)

| Candidate Direction | Focus | Description |
|---|---|---|
| **Option A: Prescriptive Decision Engine** | Operations Research | Counterfactual Monte Carlo simulation ("What if customer X pays 5 days early?"), action optimization for insolvency prevention. |
| **Option B: Fintech & TReDS Financing** | Credit Underwriting | Dynamic cash-flow credit scoring, automated invoice discounting matching for working-capital shortfall. |
| **Option C: Agentic Financial Copilot** | GenAI / Agents | Plain-language conversational advisory agent integrated with tool-calling on the cash-flow engine. |

*Strategy:* Retain flexibility to select the exact Semester 2 direction with Prof. Vajpayee after the live Semester 1 demo.

---

## 8. Technical Pipeline (Full Stack)

```
Data Layer:       data/raw/  — raw CSVs, gitignored, never edited
                  data/processed/  — cleaned daily panels

Feature Layer:    Python (pandas) — rolling features, no data leakage
                  Rolling revenue (7/30-day avg, volatility)
                  Receivables ageing, % overdue, receivables/revenue ratio
                  Upcoming payables (next 7/14/30 days)
                  Inventory spend vs. historical average
                  Seasonality (day-of-week, month)

Model Layer:      scikit-learn, XGBoost, LightGBM
                  Evaluation: walk-forward validation (NOT random split)
                  Metrics: MAE/RMSE (forecasting), AUC/PR-AUC (classification)

Explanation:      SHAP — decomposes each prediction into per-feature contributions

Serving Layer:    FastAPI (endpoint: upload transactions, get forecast + risk + explanation)

Demo UI:          CSV-upload interface — no live business integration needed
```

---

## 9. Data Sources

### Real Data (Level 1)

| Dataset | Filename | Rows | Key findings |
|---|---|---|---|
| IBM Watson Analytics (Late Payment Histories) | `WA_Fn-UseC_-Accounts-Receivable.csv` | 2,466 | Right-skewed DaysLate, symmetric amounts. Likely tutorial/curated. |
| SAP-style invoice export (Kaggle) | `dataset.csv` | 50,000 raw, 36,818 after filtering | More realistic. 38.7% early / 22.2% on-time / 39.1% late. Negative delays are real. |

### Critical Data Findings (Week 2 EDA)

1. **IBM dataset floors at 0** — never captures early payment. Right-skewed (gamma fits). Amounts are *symmetric* — NOT lognormal (mean ~ median ~ 60).
2. **SAP dataset has genuine three-way split** (early/on-time/late). Min = -89 days. Max = +204 days. A single gamma distribution *cannot* model this.
3. **clear_date NaN == isOpen=1** is exact: 10,000 open (unpaid) invoices have no settlement date — censored data, not dirty data.
4. **area_business is 100% empty** — drop this column during cleaning.
5. **Filter to USD only** — 46,081 USD vs. 3,919 CAD. CAD rows dropped.
6. **Customer payment personality is real**, not noise: CCU013 (n=539) consistently averages +42 days late; 100054980 averages -24.5 days early with std 3.56. The synthetic generator must assign per-customer-relationship parameters, not one global distribution.

### Synthetic Data (Level 2)

Purpose: generate controlled experiments the real data can't support (e.g., "what if a business only has 6 weeks of history?").

How it works:
1. Learn rules from real data — measure actual payment-delay distributions, amount spreads, default rates via `scipy.stats` fitting.
2. Assign each simulated business a "personality" — sampled around calibrated parameters with variation.
3. Simulate day-by-day — for each day: Poisson-sample number of sales, lognormal-sample amounts, gamma/lognormal-sample payment delays per customer-relationship, periodic fixed expenses.
4. Cash balance *emerges naturally* from summing events — never directly generated.
5. Validate with KS-test — compare synthetic vs. real distributions.

**Key design decision (from real data):** Payment timing must use a **two-part model**: (1) classify invoice as early/on-time/late, then (2) model the magnitude within each category. Gamma alone fails because 39% of invoices are early (negative delay).

**Rule:** Final accuracy claims are always validated on *real* data. Synthetic data is only used for controlled experiments, explicitly labeled as synthetic.

### Real-World Validation (Level 3, stretch goal)

Approach: anonymized data from local MSMEs, IIITD incubator startups, or family businesses. Even basic "Month | Revenue | Expenses | Receivables | Payables" tables would help. Project does *not depend* on this.

---

## 10. Literature Review Summary

Eight papers read and rigorously fact-checked in Week 1.

### Closest Prior Work

| Paper | Overlap | What's missing |
|---|---|---|
| arXiv 2511.03631 (SME AR + Cash Flow, 2026) | Most similar system — SVM for AR prediction, modular cash-flow forecasting; even tested one sparse-data scenario | No unified liquidity-risk score; no lead-time analysis; only one sparse-data point, not a systematic curve |
| Medianovskyi et al. 2023 (Interpretable ML, CatBoost + SHAP) | Closest on explainability — gradient boosting + SHAP for SME risk | Firm-level financial ratios, not transaction-level; no detection-horizon study |
| Cheraghali & Molnar (SME default, 6,100+ model combos) | LightGBM confirmed strongest performer — validates model choice | Annual/quarterly financial data, not day-to-day transactions; no early-warning framing |

### The Gap Our Project Fills

No paper combines all four:
1. Transaction-level, day-to-day input data
2. A unified liquidity-stress risk score (not separate forecast/classification outputs)
3. An explicit study of how many days in advance risk can be flagged
4. A systematic robustness study across different history lengths

### Benchmark Expectations

Based on published literature:
- SME distress prediction with gradient-boosted models: AUC 0.80–0.94+ range
- LightGBM consistently the strongest performer
- Our "wow" metric equivalent: "detects X% of stress events at least Y days in advance at Z% precision"

### Critical Fact-Check Notes

- The 0.94 AUC cited early was from a **different paper** (1.8M+ firm-year observations), NOT Pizzi et al. Do not attribute to Pizzi.
- IEEE paper 2 (lit review) is accessible only via abstract — no specific model names or gap statements can be claimed from it.
- General rule: never state a number as fact without verifying it from the actual paper.

---

## 11. Current Project State (as of Sept 13, 2026)

### DONE — Week 1

- Python 3.12 + venv + VS Code environment working
- Git repo initialized, committed, pushed to `Jatin288/msme-cashflow-btp`
- Folder structure: `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `reports/`
- 8 papers read and rigorously fact-checked, saved to `reports/literature_notes_week1.md`
- Problem formulation document: `reports/BTP_Problem_Formulation.md`
- Weekly update email sent to Prof. Vajpayee with the PDF attached

### DONE — Week 2

- IBM and SAP datasets downloaded to `data/raw/`
- Jupyter notebook `notebooks/01_eda_ibm_dataset.ipynb` created, kernel = venv
- Full EDA on both datasets completed
- Three key findings logged in `reports/data_notes_week2.md`
- Weekly update email drafted (findings inline; no attachment; customer-personality hook added)
- Everything committed and pushed to GitHub

### DONE — Week 3

- Built `src/build_daily_panel.py` — cleans SAP dataset, builds daily cash-flow panel for U001
- Panel: 510 rows × 17 columns (revenue_booked, cash_collected, receivables_outstanding, 13 rolling features)
- All features backward-looking (no data leakage)
- Reconciliation verified: ₹0.00 difference between total revenue and total collections
- Added README.md to GitHub repo
- Weekly email sent to Prof. Vajpayee (with GitHub link)
- Bug found and fixed: `due_in_date` stored as integer (YYYYMMDD), not date string

### DONE — Week 4

- Built `src/fit_real_distributions.py` — fits lognormal (amounts), Poisson (counts), gamma (delays), customer personalities from real data
- Built `src/generate_synthetic_data.py` — simulates 20 MSME businesses × 365 days with complete cash flow (inflows + outflows + cash balance)
- 12/20 synthetic businesses experience liquidity stress events (cash balance < 0)
- Built `src/validate_synthetic.py` — KS-test validation comparing synthetic vs real distributions
- Validation results: invoice amounts and payment delays well-calibrated (KS < 0.15); daily revenue differences explained by MSME scale vs large business (structural, not error)
- Key design decisions: two-part payment model with gamma magnitudes, per-customer payment personalities, COGS 55-75%, initial buffer 7-30 days

### DONE — Week 5

- Built `src/walk_forward.py` — expanding-window walk-forward validation engine
- Built `src/models/baselines.py` — 4 baseline forecasters (Last Value, 7-Day Avg, 30-Day Avg, Linear Trend)
- Evaluated across 4 horizons (t+7, t+14, t+30, t+60) on all 20 synthetic businesses
- 11,440 total predictions; results saved to `reports/baseline_results.csv` and `reports/model_comparison.md`
- Key findings: 7-Day Avg and Last Value closely competitive (MAE ~₹3.5M at t+7); Linear Trend worst; all baselines sMAPE 37–74%
- Framework designed for plug-in models — ARIMA/XGBoost will use the same evaluation engine

### DONE — Week 6

- Built `src/models/arima_model.py` — ARIMA forecaster with auto-order selection (pmdarima)
- Built `src/run_arima_eval.py` — combined baselines + ARIMA walk-forward evaluation
- Auto-ARIMA selected order (0,1,0) for 17/20 businesses = random walk model
- 0% fit failure rate across 2,860 ARIMA fits
- **Key finding: ARIMA does NOT beat baselines** — lags behind by 2.7–6.3% vs best baseline at every horizon
- Root cause: ARIMA is univariate (uses only cash_balance history), cannot leverage the 22 engineered features
- This validates the need for multivariate ML models (XGBoost) in Week 7

### NEXT — Week 7+

XGBoost/LightGBM regression using all engineered features with the same walk-forward framework

---

## 12. Files in the Project

| File | Location | Purpose |
|---|---|---|
| `literature_notes_week1.md` | `reports/` | Per-paper notes for all 8 papers + synthesis |
| `BTP_Problem_Formulation.md` | `reports/` | 5-section formal problem statement (sent to advisor as PDF) |
| `data_notes_week2.md` | `reports/` | EDA findings from both datasets |
| `01_eda_ibm_dataset.ipynb` | `notebooks/` | EDA notebook for IBM + SAP datasets |
| `PROJECT_CONTEXT.md` | root | This file — full project memory |
| `README.md` | root | Project overview for GitHub |
| `.gitignore` | root | Ignores `venv/`, `data/raw/`, `conversation_context.md` |
| `WA_Fn-UseC_-Accounts-Receivable.csv` | `data/raw/` | IBM dataset (2,466 rows) |
| `dataset.csv` | `data/raw/` | SAP-style dataset (50,000 rows) |
| `build_daily_panel.py` | `src/` | Week 3: cleans SAP data, builds daily panel with rolling features |
| `fit_real_distributions.py` | `src/` | Week 4: fits lognormal, Poisson, gamma to real data |
| `generate_synthetic_data.py` | `src/` | Week 4: simulates 20 MSME businesses with full cash flow |
| `validate_synthetic.py` | `src/` | Week 4: KS-test validation report |
| `walk_forward.py` | `src/` | Week 5: Walk-forward validation engine |
| `models/baselines.py` | `src/models/` | Week 5: 4 baseline forecasters |
| `models/arima_model.py` | `src/models/` | Week 6: ARIMA forecaster with auto-order selection |
| `run_arima_eval.py` | `src/` | Week 6: Combined baselines + ARIMA evaluation |
| `daily_panel_u001.csv` | `data/processed/` | Real daily panel (510 rows × 17 cols) |
| `fitted_params.json` | `data/processed/` | Fitted distribution parameters |
| `synthetic_panels.csv` | `data/processed/` | Synthetic panels (7,300 rows × 22 cols) |
| `baseline_results.csv` | `reports/` | Raw walk-forward predictions (11,440 rows) |
| `arima_results.csv` | `reports/` | Raw predictions incl. ARIMA (14,300 rows) |
| `model_comparison.md` | `reports/` | Formatted model comparison table |

---

## 13. Communication with Prof. Vajpayee

### Established tone and standards

- "Proof, not claim" — every statement backed by a specific number or finding.
- Emails are weekly, short (3–4 substantive lines), structured as: what was done, key finding, what's next, optional hook.
- Inline numbers preferred over attachments (unless a polished PDF genuinely adds credibility, as with Week 1's problem formulation).
- Never promise something in an email that next week's actual work might not deliver.

### Email trail

- **Week 1 email:** Sent with `BTP_Problem_Formulation.pdf` attached. Led with literature positioning and the two research questions.
- **Week 2 email:** No attachment. Led with three specific numbers (38.7% early, 22.2% on-time, 39.1% late). Hook: asked if he has domain intuition on customer payment patterns in MSME credit.
- **Weeks 3–5 updates:** Shared data pipeline, synthetic generator calibration (KS-test verified), and baseline walk-forward results.
- **Week 6 email:** Shared ARIMA evaluation across 20 businesses (2,860 fits). Auto-ARIMA converged to (0,1,0) random walk with zero drift, failing to beat simple moving averages. Proves univariate limits and justifies multivariate ML.
- **Week 6 Hook Question (Offline Meeting Bridge):**
  *"Our ARIMA benchmark confirmed that cash balance cannot be forecasted from historical balances alone (it degenerated into a random walk with zero drift, mirroring the naive baseline). Given this, how should we formulate the ground-truth liquidity stress threshold for our multivariate models: as an expense-coverage multiple (e.g. 30 days of fixed expenses) or as a statistical drawdown from the business's historical operating buffer?"*

### Offline Meeting Strategy with Prof. Vajpayee
- **Core Message:** "Semester 1 culminates in an early warning ML model AND a live, visible web application by December. Semester 2 stays open-ended for prescriptive decision intelligence."
- **60-Second Verbal Elevator Pitch:**
  *"Sir, in the first 6 weeks, I finished the data pipeline, built a realistic simulation calibrated on real invoice patterns, and tested our baseline models and ARIMA across 20 businesses.*
  *For the rest of this semester, my goal is to deliver a complete, working system by December. I will train modern gradient boosting models, build a liquidity stress classifier, answer our two core research questions on early-warning lead time and data scarcity, and wrap it all inside a working web dashboard where someone can upload transactions and see the forecast and risk warnings live.*
  *This way, Semester 1 ends with a working software prototype and a solid research foundation. For Semester 2, we can sit down after the demo and decide whether to extend it towards decision optimization, invoice financing recommendations, or financial advisory."*

---

## 14. If Prof. Vajpayee Asks Harder Questions

**"How is this different from Udyox or similar apps?"**  
Udyox is an *operational recording* system — business types in data daily. This is a *predictive layer* — looks at what already happened and forecasts what's coming. No live data entry required.

**"How is this different from the News-to-Stock BTP he already advised?"**  
That project predicts *stock price direction* from *news sentiment* — public markets. This predicts *business cash shortfall* from *that business's own transaction history* — private, operational, entirely different domain.

**"Is this really two semesters of work, or is it too small?"**  
It is structured into two distinct, high-impact halves to avoid artificial elongation:
1. **Semester 1 (Predictive & Diagnostic Platform):** Delivers the complete forecasting benchmark, liquidity stress classifier, empirical studies (RQ1 lead time & RQ2 data scarcity), and a *live, visible web application* (FastAPI + interactive dashboard).
2. **Semester 2 (Prescriptive & Decision Intelligence):** Moves from passive prediction to active decision optimization — counterfactual simulation ("what if we offer a 2% early-payment discount?"), invoice discounting matching (TReDS), or agentic advisory workflows.

**"Where does the data come from? Does a business type it in live?"**  
No live data entry. We train and validate entirely on historical records. A demo would accept a CSV file upload of historical transactions — the system outputs forecast, risk score, and explanation.

---

## 15. Key Decisions Locked In

1. **Two-part payment model** (not single gamma) — required by real data showing 39% early payments.
2. **Per-customer-relationship parameters** in the synthetic generator — required by real data showing customer payment personality is structural, not noise.
3. **USD-only analysis** for the SAP dataset — CAD rows dropped for clean modeling.
4. **Open invoices excluded from delay calculations** — censored data; handled separately later (relevant to RQ1).
5. **Walk-forward validation** — not random train/test split (which would be data leakage).
6. **PR-AUC over accuracy** for the risk classifier — because stress events are a minority class.
7. **XGBoost/LightGBM as primary models** — justified by literature; LSTM is a stretch goal.

---

## 16. Open Questions / Things to Verify

- [ ] Exact sample size and best-model metric for Pizzi et al. (2025) — need full paper access.
- [ ] Medianovskyi et al. AUC — accessible material only says "CatBoost was most accurate," no number confirmed.
- [ ] Verify the 1.8M+ firm-year paper title/authors (real source of the 0.94 AUC) for eventual citation.
- [ ] Minimum operating buffer: needs a precise definition before stress labeling in Semester 2. Will it be business-specific (e.g., 30-day rolling average expense) or user-defined?
- [ ] IIITD library IEEE Xplore access — not yet resolved; worth checking for all future paywalled papers.
