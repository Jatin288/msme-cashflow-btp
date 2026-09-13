# MSME Cash-Flow BTP — Complete Project Context

> **Last updated:** September 5, 2026  
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

## 7. Two-Semester Technical Roadmap

### Semester 1 (Aug 15 – Dec 15, 2026): The Forecasting Core

| Phase | Weeks | Work | Status |
|---|---|---|---|
| Literature + problem formulation | 1–2 | 8 papers read, synthesis, problem formulation doc written | DONE |
| Data pipeline | 3–5 | Daily cash-flow panel from raw invoices, feature engineering | NEXT |
| Synthetic data + KS validation | 6–8 | Calibrated generator, two-part model, validation | Pending |
| Forecasting models (baselines to XGBoost) | 9–12 | Naive, ARIMA, XGBoost/LightGBM, walk-forward eval | Pending |
| Integration, evaluation, report | 13–16 | End-to-end demo (CSV upload), MAE/RMSE results | Pending |
| Buffer + submission | 17–18 | Exam period, final commit | Pending |

**Semester 1 deliverable:** A working forecasting pipeline that predicts cash balance 7/14/30/60 days ahead, backtested on real + synthetic data, with a clear "which model forecasts best and why" answer.

CORE (must finish): Forecasting pipeline + evaluation.  
STRETCH (if time allows): LSTM temporal model comparison.

### Semester 2 (Jan – May 2027): Risk, Early Warning, Explainability

| Phase | Month | Work | Status |
|---|---|---|---|
| Risk classification | 5 | Define stress label precisely; logistic regression + XGBoost classifier | Pending |
| **RQ1: Early-detection-horizon study** | 6 | Detection lead-time curve — main research contribution | Pending |
| **RQ2: Data-scarcity robustness** + SHAP | 7 | Robustness curve by history length; SHAP explanation layer | Pending |
| Decision support + demo + final report | 8 | Monte Carlo intervention simulation; integrate all; write thesis | Pending |

Semester 2 CORE: RQ1 + RQ2 + SHAP. These are what academically justify two semesters.  
STRETCH: Monte Carlo decision-support simulation, real-world MSME validation.

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

## 11. Current Project State (as of Sept 5, 2026)

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

### NEXT — Week 3

Goal: Build the daily cash-flow panel — the core data structure every downstream model reads from.

One row per business per day:
`business_id | date | revenue | expenses | receivables_outstanding | payables_due | inventory_spend | cash_balance`

Steps:
1. Clean SAP dataset properly (drop `area_business`, filter USD, handle open invoices, parse date columns)
2. Aggregate transaction records into the daily panel format
3. Compute rolling features (no leakage — all features must use only past information relative to prediction date)

---

## 12. Files in the Project

| File | Location | Purpose |
|---|---|---|
| `literature_notes_week1.md` | `reports/` | Per-paper notes for all 8 papers + synthesis |
| `BTP_Problem_Formulation.md` | `reports/` | 5-section formal problem statement (sent to advisor as PDF) |
| `data_notes_week2.md` | `reports/` | EDA findings from both datasets |
| `01_eda_ibm_dataset.ipynb` | `notebooks/` | EDA notebook for IBM + SAP datasets |
| `PROJECT_CONTEXT.md` | root | This file — full project memory |
| `.gitignore` | root | Ignores `venv/` and `data/raw/` |
| `WA_Fn-UseC_-Accounts-Receivable.csv` | `data/raw/` | IBM dataset (2,466 rows) |
| `dataset.csv` | `data/raw/` | SAP-style dataset (50,000 rows) |

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

### For future emails

- GitHub link: save for Week 3–4, once there is actual pipeline code in `src/` someone can read.
- Weekly plan file: do NOT attach — it has scaffolding notes written for internal use.
- Literature notes file: hold back unless he specifically asks; the problem formulation PDF is the distilled reader-facing version.

---

## 14. If Prof. Vajpayee Asks Harder Questions

**"How is this different from Udyox or similar apps?"**  
Udyox is an *operational recording* system — business types in data daily. This is a *predictive layer* — looks at what already happened and forecasts what's coming. No live data entry required.

**"How is this different from the News-to-Stock BTP he already advised?"**  
That project predicts *stock price direction* from *news sentiment* — public markets. This predicts *business cash shortfall* from *that business's own transaction history* — private, operational, entirely different domain.

**"Is this really two semesters of work?"**  
The forecasting pipeline alone is roughly one semester. What extends it is the two research studies: the early-detection-horizon curve (RQ1) and the robustness-under-scarcity curve (RQ2). Those are genuine experiments with citable findings, not extra features.

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
