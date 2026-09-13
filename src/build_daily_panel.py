"""
build_daily_panel.py — Week 3: Data Pipeline
Cleans the SAP dataset and builds a daily cash-flow panel for business U001.

Output: data/processed/daily_panel_u001.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "dataset.csv"
OUT_PATH = ROOT / "data" / "processed" / "daily_panel_u001.csv"

def main():
    # ── Step 1: Load ──
    print("Loading SAP dataset...")
    df = pd.read_csv(RAW_PATH)
    print(f"  Raw rows: {len(df):,}")

    # ── Step 2: Clean ──
    # Drop area_business (100% empty — confirmed in Week 2 EDA)
    df = df.drop(columns=["area_business"])

    # Filter to USD only (46,081 USD vs 3,919 CAD — decision locked in)
    df = df[df["invoice_currency"] == "USD"].copy()
    print(f"  After USD filter: {len(df):,}")

    # Keep only closed invoices (isOpen == 0)
    # Open invoices have no clear_date — censored data, handled separately later
    df = df[df["isOpen"] == 0].copy()
    print(f"  After removing open invoices: {len(df):,}")

    # Parse date columns
    df["posting_date"] = pd.to_datetime(df["posting_date"])
    df["clear_date"] = pd.to_datetime(df["clear_date"])
    df["due_in_date"] = pd.to_datetime(df["due_in_date"])

    # ── Step 3: Filter to U001 ──
    u001 = df[df["business_code"] == "U001"].copy()
    print(f"  U001 invoices: {len(u001):,}")
    print(f"  Date range: {u001['posting_date'].min().date()} to {u001['posting_date'].max().date()}")

    # ── Step 4: Build daily panel ──
    # Daily revenue booked (invoices issued, by posting_date)
    revenue_daily = u001.groupby("posting_date")["total_open_amount"].sum()
    revenue_daily.name = "revenue_booked"

    # Daily cash collected (invoices settled, by clear_date)
    collected_daily = u001.groupby("clear_date")["total_open_amount"].sum()
    collected_daily.name = "cash_collected"

    # Full daily calendar — no gaps allowed
    full_range = pd.date_range(
        start=u001["posting_date"].min(),
        end=u001["clear_date"].max(),
        freq="D",
    )
    panel = pd.DataFrame(index=full_range)
    panel.index.name = "date"
    panel["business_id"] = "U001"
    panel["revenue_booked"] = revenue_daily
    panel["cash_collected"] = collected_daily
    panel = panel.fillna(0)

    # ── Step 5: Receivables outstanding ──
    # On any given day: how much money is owed to this business?
    # Rises when invoices are issued, drops when payments come in
    panel["receivables_outstanding"] = (
        panel["revenue_booked"].cumsum() - panel["cash_collected"].cumsum()
    )

    # ── Step 6: Rolling features (no leakage — backward-looking only) ──

    # Rolling revenue averages
    panel["revenue_7d_avg"] = panel["revenue_booked"].rolling(7, min_periods=1).mean()
    panel["revenue_30d_avg"] = panel["revenue_booked"].rolling(30, min_periods=1).mean()

    # Rolling revenue volatility
    panel["revenue_7d_std"] = panel["revenue_booked"].rolling(7, min_periods=2).std()
    panel["revenue_30d_std"] = panel["revenue_booked"].rolling(30, min_periods=2).std()

    # Rolling collection averages
    panel["collections_7d_avg"] = panel["cash_collected"].rolling(7, min_periods=1).mean()
    panel["collections_30d_avg"] = panel["cash_collected"].rolling(30, min_periods=1).mean()

    # Collection efficiency: 30-day collections / 30-day revenue
    # Below 1.0 = collecting less than billing = cash getting stuck
    panel["collection_ratio_30d"] = (
        panel["cash_collected"].rolling(30, min_periods=1).sum()
        / panel["revenue_booked"].rolling(30, min_periods=1).sum()
    ).replace([np.inf, -np.inf], np.nan)

    # Receivables trend: change over last 7 / 30 days
    panel["receivables_change_7d"] = panel["receivables_outstanding"].diff(7)
    panel["receivables_change_30d"] = panel["receivables_outstanding"].diff(30)

    # Receivables-to-revenue ratio: outstanding vs recent billing
    # High = cash stuck in unpaid invoices
    panel["receivables_to_revenue_30d"] = (
        panel["receivables_outstanding"]
        / panel["revenue_booked"].rolling(30, min_periods=1).sum()
    ).replace([np.inf, -np.inf], np.nan)

    # Seasonality
    panel["day_of_week"] = panel.index.dayofweek
    panel["month"] = panel.index.month
    panel["is_weekend"] = panel["day_of_week"].isin([5, 6]).astype(int)

    print(f"  Features added: {len(panel.columns)} columns total")

    # ── Reconciliation check ──
    total_rev = panel["revenue_booked"].sum()
    total_col = panel["cash_collected"].sum()
    diff = total_rev - total_col
    print(f"\n  Reconciliation check:")
    print(f"    Total revenue booked:  {total_rev:,.2f}")
    print(f"    Total cash collected:  {total_col:,.2f}")
    print(f"    Difference:            {diff:,.2f}")
    assert abs(diff) < 0.01, f"RECONCILIATION FAILED: difference = {diff}"

    # ── Save ──
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_PATH)
    print(f"\n  Saved: {OUT_PATH}")
    print(f"  Panel shape: {panel.shape}")
    print(f"\nFirst 10 rows:")
    print(panel.head(10))

if __name__ == "__main__":
    main()
