# Cash-Flow Early Warning System for MSME Liquidity Risk

**B.Tech Project (BTP)** — IIIT Delhi  
**Student:** Jatin Pawar  
**Advisor:** Prof. Pankaj Vajpayee  

## Overview

Small businesses fail not because they are unprofitable, but because cash comes in late and goes out on time. This project builds a system that predicts a business's cash position weeks ahead, flags when a shortfall is likely, and explains why.

## Research Questions

- **RQ1 (Early-Detection Horizon):** How many days before a liquidity-stress event can the system reliably flag it, and at what precision?
- **RQ2 (Data-Scarcity Robustness):** How does prediction reliability degrade as available transaction history shrinks (12 months → 6 weeks)?

## Project Structure

```
src/
  build_daily_panel.py       → Cleans SAP data, builds daily cash-flow panel
  fit_real_distributions.py  → Fits statistical distributions to real data
  generate_synthetic_data.py → Simulates 20 MSME businesses with full cash flow
  validate_synthetic.py      → KS-test validation (synthetic vs real)
  walk_forward.py            → Walk-forward validation engine
  run_arima_eval.py          → Combined baselines + ARIMA evaluation
  models/
    baselines.py             → Baseline forecasters (Last Value, Moving Avg, Linear Trend)
    arima_model.py           → ARIMA forecaster with auto-order selection
data/raw/                    → Raw datasets (gitignored)
data/processed/              → Cleaned panels, fitted params, synthetic data
notebooks/                   → Exploratory analysis
reports/                     → Literature notes, data findings, model comparison
```

## Current Status

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | Literature review (8 papers) + problem formulation | ✅ |
| 2 | EDA on IBM & SAP datasets | ✅ |
| 3 | Daily cash-flow panel + feature engineering | ✅ |
| 4 | Synthetic data generation + KS validation | ✅ |
| 5 | Walk-forward validation + baseline models | ✅ |
| 6 | ARIMA / Auto-ARIMA | ✅ |
| 7 | XGBoost / LightGBM | 🔜 |

## Running the Pipeline

```bash
python -m venv venv
venv\Scripts\activate
pip install pandas numpy scipy

# Build real data panel
python src/build_daily_panel.py

# Fit distributions + generate synthetic data
python src/fit_real_distributions.py
python src/generate_synthetic_data.py

# Validate synthetic data
python src/validate_synthetic.py

# Run baseline evaluation
python src/walk_forward.py
```

## Baseline Results (Week 5)

| Model | t+7 MAE | t+30 MAE | t+60 MAE |
|---|---:|---:|---:|
| 7-Day Average | ₹3.48M | ₹5.09M | ₹7.34M |
| Last Value | ₹3.54M | ₹5.04M | ₹7.45M |
| 30-Day Average | ₹3.95M | ₹5.70M | ₹8.09M |
| Linear Trend (30d) | ₹4.16M | ₹8.19M | ₹13.95M |

Evaluated via expanding-window walk-forward validation on 20 synthetic businesses (11,440 predictions).

