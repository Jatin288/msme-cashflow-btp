"""
walk_forward.py — Week 5: Walk-Forward Validation Engine
Evaluates forecasting models on synthetic cash-flow data using
expanding-window walk-forward validation.

Methodology:
    For each business:
        For each evaluation date t (starting day 90, stepping every 7 days):
            Training data = all rows up to t (expanding window)
            For each horizon h in [7, 14, 30, 60]:
                actual  = cash_balance at t + h
                predict = model.predict(train_data, h)
                Record (business, t, h, predicted, actual)

    Aggregate: per-model, per-horizon MAE, RMSE, sMAPE

Output:
    reports/baseline_results.csv   — per-prediction raw results
    reports/model_comparison.md    — formatted comparison table
    Console                        — summary table
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path for imports
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.baselines import ALL_BASELINES

# ── Configuration ──
DATA_PATH = ROOT / "data" / "processed" / "synthetic_panels.csv"
RESULTS_PATH = ROOT / "reports" / "baseline_results.csv"
COMPARISON_PATH = ROOT / "reports" / "model_comparison.md"

HORIZONS = [7, 14, 30, 60]
MIN_HISTORY = 90       # minimum days of history before first evaluation
STEP_SIZE = 7          # evaluate every 7 days (not every day — too slow)
TARGET_COL = "cash_balance"


def load_data():
    """Load synthetic panels and parse dates."""
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"  Loaded {len(df):,} rows, {df['business_id'].nunique()} businesses")
    return df


def smape(actual, predicted):
    """Symmetric Mean Absolute Percentage Error.
    
    Handles zero/negative values gracefully (unlike MAPE).
    Range: 0% (perfect) to 200% (worst).
    """
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    if denom == 0:
        return 0.0  # both are zero → perfect prediction
    return np.abs(actual - predicted) / denom * 100


def evaluate_business(biz_data, models):
    """Run walk-forward validation on a single business.
    
    Returns a list of result dicts, one per (model, eval_date, horizon).
    """
    biz_data = biz_data.sort_values("date").reset_index(drop=True)
    biz_id = biz_data["business_id"].iloc[0]
    n_days = len(biz_data)
    cash = biz_data[TARGET_COL]

    results = []

    # Evaluation dates: start from MIN_HISTORY, step every STEP_SIZE days
    eval_indices = range(MIN_HISTORY, n_days, STEP_SIZE)

    for t in eval_indices:
        # Training data: everything up to and including day t
        train_series = cash.iloc[:t + 1]

        for h in HORIZONS:
            target_idx = t + h
            if target_idx >= n_days:
                continue  # can't evaluate — target is beyond our data

            actual = cash.iloc[target_idx]

            for model in models:
                predicted = model.predict(train_series, h)

                results.append({
                    "business_id": biz_id,
                    "eval_date": biz_data["date"].iloc[t],
                    "horizon": h,
                    "model": model.name,
                    "predicted": predicted,
                    "actual": actual,
                    "error": predicted - actual,
                    "abs_error": abs(predicted - actual),
                    "smape": smape(actual, predicted),
                })

    return results


def compute_summary(results_df):
    """Aggregate results into per-model, per-horizon metrics."""
    summary = (
        results_df
        .groupby(["model", "horizon"])
        .agg(
            MAE=("abs_error", "mean"),
            RMSE=("abs_error", lambda x: np.sqrt(np.mean(x ** 2))),
            sMAPE=("smape", "mean"),
            n_predictions=("abs_error", "count"),
        )
        .round(2)
        .reset_index()
    )
    return summary


def format_comparison_table(summary):
    """Generate a markdown comparison table."""
    lines = [
        "# Baseline Model Comparison — Walk-Forward Validation",
        "",
        f"> **Data:** 20 synthetic MSME businesses × 365 days",
        f"> **Validation:** Expanding window, min {MIN_HISTORY} days history, step {STEP_SIZE} days",
        f"> **Target:** `{TARGET_COL}` (₹)",
        f"> **Horizons:** {', '.join(f't+{h}' for h in HORIZONS)}",
        "",
        "## Results",
        "",
        "| Model | Horizon | MAE (₹) | RMSE (₹) | sMAPE (%) | # Predictions |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | t+{row['horizon']} "
            f"| {row['MAE']:,.0f} | {row['RMSE']:,.0f} "
            f"| {row['sMAPE']:.1f} | {int(row['n_predictions'])} |"
        )

    # Add interpretation section
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **MAE** (Mean Absolute Error): Average prediction error in ₹. Lower = better.",
        "- **RMSE** (Root Mean Squared Error): Penalizes large errors more. RMSE ≥ MAE always.",
        "- **sMAPE** (Symmetric MAPE): Scale-independent error %. Handles zero/negative balances.",
        "- MAE should **increase with horizon** (harder to predict further ahead).",
        "- If 7-day avg beats Last Value → cash flow is **mean-reverting**.",
        "- If Last Value beats averages → cash flow has **momentum/trend**.",
        "- If Linear Trend beats others at long horizons → directional signal is strong.",
        "",
        "These numbers are the **benchmark** — ARIMA (Week 6) and XGBoost (Week 7) must beat them.",
    ])

    return "\n".join(lines)


def print_summary(summary):
    """Print formatted summary to console."""
    print("\n" + "=" * 80)
    print("BASELINE MODEL COMPARISON — WALK-FORWARD VALIDATION")
    print("=" * 80)

    for model_name in summary["model"].unique():
        print(f"\n  {model_name}:")
        model_rows = summary[summary["model"] == model_name]
        for _, row in model_rows.iterrows():
            print(f"    t+{row['horizon']:>2}:  MAE = {row['MAE']:>14,.0f}  "
                  f"RMSE = {row['RMSE']:>14,.0f}  "
                  f"sMAPE = {row['sMAPE']:>6.1f}%  "
                  f"(n={int(row['n_predictions'])})")

    # Find best model per horizon
    print(f"\n  {'-' * 60}")
    print("  Best model per horizon (by MAE):")
    for h in HORIZONS:
        h_rows = summary[summary["horizon"] == h]
        best = h_rows.loc[h_rows["MAE"].idxmin()]
        print(f"    t+{h:>2}: {best['model']} (MAE = {best['MAE']:,.0f})")

    print("=" * 80)


def main():
    print("=== Week 5: Walk-Forward Validation ===\n")

    # Load data
    print("Loading synthetic data...")
    df = load_data()

    # Get models
    models = ALL_BASELINES
    print(f"  Models: {', '.join(m.name for m in models)}")
    print(f"  Horizons: {HORIZONS}")
    print(f"  Min history: {MIN_HISTORY} days, Step: {STEP_SIZE} days\n")

    # Evaluate each business
    all_results = []
    businesses = df["business_id"].unique()

    for biz_id in businesses:
        biz_data = df[df["business_id"] == biz_id]
        results = evaluate_business(biz_data, models)
        all_results.extend(results)
        n_preds = len(results)
        print(f"  {biz_id}: {n_preds} predictions")

    results_df = pd.DataFrame(all_results)
    print(f"\n  Total predictions: {len(results_df):,}")

    # Compute summary
    summary = compute_summary(results_df)

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n  Raw results saved: {RESULTS_PATH}")

    # Save comparison table
    comparison_md = format_comparison_table(summary)
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        f.write(comparison_md)
    print(f"  Comparison table saved: {COMPARISON_PATH}")

    # Print summary to console
    print_summary(summary)


if __name__ == "__main__":
    main()
