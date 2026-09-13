"""
generate_synthetic_data.py — Week 4: Synthetic Business Simulator
Generates complete daily cash-flow panels for synthetic MSME businesses.
Uses parameters fitted from real data (fitted_params.json).

Each business gets:
- Revenue (inflows): calibrated from real invoice patterns
- Expenses (outflows): fixed costs + variable COGS with supplier delays
- Cash balance: emerges naturally from collections minus payments

Output: data/processed/synthetic_panels.csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "data" / "processed" / "fitted_params.json"
OUT_PATH = ROOT / "data" / "processed" / "synthetic_panels.csv"

# ── Configuration ──
N_BUSINESSES = 20
N_DAYS = 365
START_DATE = "2019-01-01"
SEED = 42


def load_params():
    with open(PARAMS_PATH) as f:
        return json.load(f)


def sample_delay(delay_params, customer_shift, rng):
    """
    Two-part payment delay model + customer personality shift.

    1. Draw category (early/on-time/late) from fitted proportions
    2. Draw magnitude from fitted gamma distribution
    3. Add customer-specific shift (some customers always pay late/early)
    """
    p = delay_params["proportions"]
    category = rng.choice(
        ["early", "on_time", "late"],
        p=[p["early"], p["on_time"], p["late"]],
    )
    if category == "early":
        base = -rng.gamma(
            delay_params["early_magnitude"]["shape"],
            delay_params["early_magnitude"]["scale"],
        )
    elif category == "on_time":
        base = 0.0
    else:
        base = rng.gamma(
            delay_params["late_magnitude"]["shape"],
            delay_params["late_magnitude"]["scale"],
        )

    return int(round(base + customer_shift))


def simulate_business(biz_id, params, rng):
    """
    Simulate one MSME business for N_DAYS days.

    Revenue side: Poisson(lambda) invoices/day, lognormal amounts,
                  collected after net-30 + delay (two-part model)
    Expense side: fixed monthly (rent + salary) + variable COGS with supplier delay
    Cash balance: initial buffer + cumulative(collections - payments)
    """
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")

    # ── Business-level parameters (sampled with variation) ──
    daily_lambda = rng.uniform(3, 30)                             # MSME scale: 3-30 invoices/day
    amt_mu = params["invoice_amount"]["mu"] + rng.normal(0, 0.3)  # per-business size shift
    amt_sigma = params["invoice_amount"]["sigma"]
    cogs_ratio = rng.uniform(0.55, 0.75)                          # 55-75% of revenue (tighter MSME margins)
    rent_pct = rng.uniform(0.08, 0.15)                            # 8-15% of monthly revenue
    salary_pct = rng.uniform(0.20, 0.35)                          # 20-35% of monthly revenue
    supplier_delay = int(rng.uniform(15, 45))                     # days to pay suppliers
    payment_terms = 30                                            # standard net-30

    # Assign customers with payment personalities
    n_cust = rng.integers(5, 25)
    cust_shifts = rng.normal(
        0, params["customer_personality"]["personality_sigma"], size=n_cust
    )

    # ── Weekly revenue variation (realistic: some bad weeks) ──
    n_weeks = (N_DAYS // 7) + 2
    weekly_multipliers = rng.lognormal(0, 0.12, size=n_weeks)  # gentle variation — matches real revenue std

    # ── Generate all invoices ──
    inv_dates, inv_amounts, inv_collect_dates = [], [], []
    for day_idx, date in enumerate(dates):
        week_idx = day_idx // 7
        adjusted_lambda = daily_lambda * weekly_multipliers[week_idx]
        n = rng.poisson(adjusted_lambda)
        for _ in range(n):
            amt = np.exp(rng.normal(amt_mu, amt_sigma))
            cust = rng.integers(0, n_cust)
            delay = sample_delay(params["payment_delay"], cust_shifts[cust], rng)
            collect = date + pd.Timedelta(days=payment_terms + delay)
            inv_dates.append(date)
            inv_amounts.append(amt)
            inv_collect_dates.append(collect)

    inv_df = pd.DataFrame({
        "date": inv_dates, "amount": inv_amounts, "collect_date": inv_collect_dates,
    })

    # Aggregate inflows by day
    revenue = inv_df.groupby("date")["amount"].sum()
    # Only count collections that fall within our simulation window
    collections = (
        inv_df[inv_df["collect_date"].between(dates[0], dates[-1])]
        .groupby("collect_date")["amount"].sum()
    )

    # ── Generate expenses ──
    # Size fixed costs relative to expected monthly revenue
    expected_daily_rev = daily_lambda * np.exp(amt_mu + amt_sigma**2 / 2)
    monthly_rent = expected_daily_rev * 30 * rent_pct
    monthly_salary = expected_daily_rev * 30 * salary_pct

    exp_incurred_records = []   # (date, amount)
    exp_paid_records = []       # (date, amount)

    for date in dates:
        day_rev = revenue.get(date, 0)

        # Variable: COGS — incurred daily, paid after supplier_delay
        cogs = day_rev * cogs_ratio
        if cogs > 0:
            exp_incurred_records.append((date, cogs))
            pay_date = date + pd.Timedelta(days=supplier_delay)
            if pay_date <= dates[-1]:
                exp_paid_records.append((pay_date, cogs))

        # Fixed: rent — incurred and paid on the 1st
        if date.day == 1:
            exp_incurred_records.append((date, monthly_rent))
            exp_paid_records.append((date, monthly_rent))

        # Fixed: salary — incurred and paid on 1st and 15th
        if date.day in (1, 15):
            exp_incurred_records.append((date, monthly_salary / 2))
            exp_paid_records.append((date, monthly_salary / 2))

    # Aggregate expenses by day
    exp_inc_df = pd.DataFrame(exp_incurred_records, columns=["date", "amount"])
    exp_pay_df = pd.DataFrame(exp_paid_records, columns=["date", "amount"])
    expenses_incurred = exp_inc_df.groupby("date")["amount"].sum()
    expenses_paid = exp_pay_df.groupby("date")["amount"].sum()

    # ── Build daily panel ──
    panel = pd.DataFrame(index=dates)
    panel.index.name = "date"
    panel["business_id"] = biz_id
    panel["revenue_booked"] = revenue
    panel["cash_collected"] = collections
    panel["expenses_incurred"] = expenses_incurred
    panel["expenses_paid"] = expenses_paid
    panel = panel.fillna(0)

    # Derived columns
    panel["receivables_outstanding"] = (
        panel["revenue_booked"].cumsum() - panel["cash_collected"].cumsum()
    )
    panel["payables_outstanding"] = (
        panel["expenses_incurred"].cumsum() - panel["expenses_paid"].cumsum()
    )

    # Cash balance: MSMEs typically have thin buffers (1-4 weeks)
    initial_cash = expected_daily_rev * rng.uniform(7, 30)
    panel["cash_balance"] = initial_cash + (
        panel["cash_collected"] - panel["expenses_paid"]
    ).cumsum()

    return panel


def add_rolling_features(panel):
    """Same rolling features as build_daily_panel.py — no leakage"""
    panel["revenue_7d_avg"] = panel["revenue_booked"].rolling(7, min_periods=1).mean()
    panel["revenue_30d_avg"] = panel["revenue_booked"].rolling(30, min_periods=1).mean()
    panel["revenue_7d_std"] = panel["revenue_booked"].rolling(7, min_periods=2).std()
    panel["revenue_30d_std"] = panel["revenue_booked"].rolling(30, min_periods=2).std()
    panel["collections_7d_avg"] = panel["cash_collected"].rolling(7, min_periods=1).mean()
    panel["collections_30d_avg"] = panel["cash_collected"].rolling(30, min_periods=1).mean()
    panel["collection_ratio_30d"] = (
        panel["cash_collected"].rolling(30, min_periods=1).sum()
        / panel["revenue_booked"].rolling(30, min_periods=1).sum()
    ).replace([np.inf, -np.inf], np.nan)
    panel["receivables_change_7d"] = panel["receivables_outstanding"].diff(7)
    panel["receivables_change_30d"] = panel["receivables_outstanding"].diff(30)
    panel["receivables_to_revenue_30d"] = (
        panel["receivables_outstanding"]
        / panel["revenue_booked"].rolling(30, min_periods=1).sum()
    ).replace([np.inf, -np.inf], np.nan)
    panel["day_of_week"] = panel.index.dayofweek
    panel["month"] = panel.index.month
    panel["is_weekend"] = panel["day_of_week"].isin([5, 6]).astype(int)
    return panel


def main():
    print(f"=== Generating {N_BUSINESSES} synthetic MSME businesses ===")
    print(f"    Period: {N_DAYS} days from {START_DATE}")
    print(f"    Seed: {SEED}\n")

    params = load_params()
    rng = np.random.default_rng(SEED)

    all_panels = []
    for i in range(N_BUSINESSES):
        biz_id = f"SYN_{i + 1:03d}"
        panel = simulate_business(biz_id, params, rng)
        panel = add_rolling_features(panel)
        all_panels.append(panel)

        # Print summary per business
        final_cash = panel["cash_balance"].iloc[-1]
        min_cash = panel["cash_balance"].min()
        stress_days = (panel["cash_balance"] < 0).sum()
        print(f"  {biz_id}: final_cash={final_cash:>14,.0f}"
              f"  min_cash={min_cash:>14,.0f}"
              f"  stress_days={stress_days:>3}")

    combined = pd.concat(all_panels)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH)

    # Summary
    n_stress = sum(1 for p in all_panels if (p["cash_balance"] < 0).any())
    print(f"\n  Total rows: {len(combined):,}")
    print(f"  Columns: {len(combined.columns)}")
    print(f"  Businesses with stress events (balance < 0): {n_stress}/{N_BUSINESSES}")
    print(f"  Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
