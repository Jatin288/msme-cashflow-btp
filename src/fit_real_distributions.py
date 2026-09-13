"""
fit_real_distributions.py — Week 4: Learn from Real Data
Fits statistical distributions to the SAP dataset's patterns.
Saves learned parameters to data/processed/fitted_params.json

These parameters drive the synthetic data generator.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "dataset.csv"
OUT_PATH = ROOT / "data" / "processed" / "fitted_params.json"


def load_and_clean():
    """Reuses the same cleaning logic from build_daily_panel.py"""
    df = pd.read_csv(RAW_PATH)
    df = df.drop(columns=["area_business"])
    df = df[df["invoice_currency"] == "USD"].copy()
    df = df[df["isOpen"] == 0].copy()
    df["posting_date"] = pd.to_datetime(df["posting_date"])
    df["clear_date"] = pd.to_datetime(df["clear_date"])
    df["due_in_date"] = pd.to_datetime(df["due_in_date"].astype(str), format="%Y%m%d")
    # Filter to U001
    df = df[df["business_code"] == "U001"].copy()
    return df


def fit_invoice_amounts(df):
    """Fit lognormal to invoice amounts (right-skewed, confirmed in Week 2)"""
    amounts = df["total_open_amount"].values
    log_amounts = np.log(amounts[amounts > 0])
    mu = log_amounts.mean()
    sigma = log_amounts.std()
    print(f"  Invoice amounts (lognormal): mu={mu:.4f}, sigma={sigma:.4f}")
    print(f"    → median amount: ${np.exp(mu):,.0f}")
    print(f"    → 95th percentile: ${np.exp(mu + 1.645 * sigma):,.0f}")
    return {"distribution": "lognormal", "mu": float(mu), "sigma": float(sigma)}


def fit_daily_invoice_count(df):
    """Fit Poisson to daily invoice counts"""
    daily_counts = df.groupby("posting_date").size()
    # Fill in zero-count days
    full_range = pd.date_range(daily_counts.index.min(), daily_counts.index.max())
    daily_counts = daily_counts.reindex(full_range, fill_value=0)
    lam = daily_counts.mean()
    print(f"  Daily invoice count (Poisson): lambda={lam:.2f}")
    print(f"    → range: {daily_counts.min()} to {daily_counts.max()}")
    return {"distribution": "poisson", "lambda": float(lam)}


def fit_payment_delay(df):
    """
    Two-part payment delay model (key design decision from Week 2):
    1. Classify as early / on-time / late
    2. Fit magnitude within each category
    """
    df = df.copy()
    df["delay_days"] = (df["clear_date"] - df["due_in_date"]).dt.days

    # Part 1: proportions
    n_early = (df["delay_days"] < 0).sum()
    n_ontime = (df["delay_days"] == 0).sum()
    n_late = (df["delay_days"] > 0).sum()
    total = len(df)

    p_early = n_early / total
    p_ontime = n_ontime / total
    p_late = n_late / total

    print(f"  Payment delay proportions:")
    print(f"    Early:   {p_early:.1%} ({n_early:,})")
    print(f"    On-time: {p_ontime:.1%} ({n_ontime:,})")
    print(f"    Late:    {p_late:.1%} ({n_late:,})")

    # Part 2: magnitudes
    early_mag = np.abs(df.loc[df["delay_days"] < 0, "delay_days"].values).astype(float)
    late_mag = df.loc[df["delay_days"] > 0, "delay_days"].values.astype(float)

    # Fit gamma distribution (heavier tails than exponential — matches real data)
    early_shape, _, early_scale = stats.gamma.fit(early_mag, floc=0)
    late_shape, _, late_scale = stats.gamma.fit(late_mag, floc=0)

    print(f"  Early magnitude (gamma): shape={early_shape:.2f}, scale={early_scale:.2f}, mean={early_shape*early_scale:.1f} days")
    print(f"  Late magnitude (gamma):  shape={late_shape:.2f}, scale={late_scale:.2f}, mean={late_shape*late_scale:.1f} days")

    return {
        "proportions": {
            "early": float(p_early),
            "on_time": float(p_ontime),
            "late": float(p_late),
        },
        "early_magnitude": {
            "distribution": "gamma",
            "shape": float(early_shape),
            "scale": float(early_scale),
        },
        "late_magnitude": {
            "distribution": "gamma",
            "shape": float(late_shape),
            "scale": float(late_scale),
        },
    }


def fit_customer_personality(df):
    """
    Per-customer payment personality — confirmed real in Week 2 EDA.
    Some customers consistently pay early, others consistently late.
    """
    df = df.copy()
    df["delay_days"] = (df["clear_date"] - df["due_in_date"]).dt.days

    # Mean delay per customer (only customers with enough invoices)
    cust_stats = df.groupby("cust_number")["delay_days"].agg(["mean", "std", "count"])
    cust_stats = cust_stats[cust_stats["count"] >= 10]  # need enough data points

    personality_mu = cust_stats["mean"].mean()
    personality_sigma = cust_stats["mean"].std()
    within_customer_std = cust_stats["std"].mean()

    print(f"  Customer personalities ({len(cust_stats)} customers with 10+ invoices):")
    print(f"    Mean of customer means: {personality_mu:.1f} days")
    print(f"    Std of customer means:  {personality_sigma:.1f} days (between-customer variation)")
    print(f"    Avg within-customer std: {within_customer_std:.1f} days (invoice-to-invoice noise)")

    return {
        "n_customers_fitted": int(len(cust_stats)),
        "personality_mu": float(personality_mu),
        "personality_sigma": float(personality_sigma),
        "within_customer_std": float(within_customer_std),
    }


def main():
    print("=== Fitting distributions to real SAP data ===\n")
    df = load_and_clean()
    print(f"  Loaded {len(df):,} invoices for U001\n")

    params = {}
    params["invoice_amount"] = fit_invoice_amounts(df)
    print()
    params["daily_invoice_count"] = fit_daily_invoice_count(df)
    print()
    params["payment_delay"] = fit_payment_delay(df)
    print()
    params["customer_personality"] = fit_customer_personality(df)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\n  Saved fitted parameters to: {OUT_PATH}")


if __name__ == "__main__":
    main()
