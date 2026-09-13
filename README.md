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
src/                  → Pipeline code
data/raw/             → Raw datasets (gitignored)
data/processed/       → Cleaned daily panels
notebooks/            → Exploratory analysis
reports/              → Literature notes, problem formulation, data findings
```

## Current Status

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | Literature review (8 papers) + problem formulation | ✅ |
| 2 | EDA on IBM & SAP datasets | ✅ |
| 3 | Daily cash-flow panel + feature engineering | ✅ |
| 4–5 | Synthetic data generation + KS validation | 🔜 |

## Running the Pipeline

```bash
python -m venv venv
venv\Scripts\activate
pip install pandas numpy
python src/build_daily_panel.py
```

Output: `data/processed/daily_panel_u001.csv` — 510 rows × 17 features.
