"""
run_arima_eval.py — Week 6: ARIMA Evaluation via Walk-Forward Validation
Evaluates ARIMA against all baselines on synthetic cash-flow data.

Uses the same walk-forward methodology as Week 5, but adds ARIMA with
per-business auto-order selection.

ARIMA is computationally expensive (~20 seconds per business), so this
script prints progress and timing per business.

Output:
    reports/arima_results.csv      — per-prediction raw results (baselines + ARIMA)
    reports/model_comparison.md    — updated comparison table
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.baselines import ALL_BASELINES
from models.arima_model import ARIMAModel

# ── Configuration ──
DATA_PATH = ROOT / "data" / "processed" / "synthetic_panels.csv"
RESULTS_PATH = ROOT / "reports" / "arima_results.csv"
COMPARISON_PATH = ROOT / "reports" / "model_comparison.md"

HORIZONS = [7, 14, 30, 60]
MIN_HISTORY = 90
STEP_SIZE = 7
TARGET_COL = "cash_balance"


def smape(actual, predicted):
    """Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    if denom == 0:
        return 0.0
    return np.abs(actual - predicted) / denom * 100


def evaluate_business(biz_data, models):
    """Run walk-forward validation on a single business."""
    biz_data = biz_data.sort_values("date").reset_index(drop=True)
    biz_id = biz_data["business_id"].iloc[0]
    n_days = len(biz_data)
    cash = biz_data[TARGET_COL]

    results = []
    eval_indices = range(MIN_HISTORY, n_days, STEP_SIZE)

    for t in eval_indices:
        train_series = cash.iloc[:t + 1]

        for h in HORIZONS:
            target_idx = t + h
            if target_idx >= n_days:
                continue

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
        "# Model Comparison — Walk-Forward Validation (Baselines + ARIMA)",
        "",
        f"> **Data:** 20 synthetic MSME businesses x 365 days",
        f"> **Validation:** Expanding window, min {MIN_HISTORY} days history, step {STEP_SIZE} days",
        f"> **Target:** `{TARGET_COL}`",
        f"> **Horizons:** {', '.join(f't+{h}' for h in HORIZONS)}",
        "",
        "## Results",
        "",
        "| Model | Horizon | MAE | RMSE | sMAPE (%) | # Predictions |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | t+{row['horizon']} "
            f"| {row['MAE']:,.0f} | {row['RMSE']:,.0f} "
            f"| {row['sMAPE']:.1f} | {int(row['n_predictions'])} |"
        )

    # Best model per horizon
    lines.extend(["", "## Best Model Per Horizon (by MAE)", ""])
    for h in HORIZONS:
        h_rows = summary[summary["horizon"] == h]
        best = h_rows.loc[h_rows["MAE"].idxmin()]
        lines.append(f"- **t+{h}:** {best['model']} (MAE = {best['MAE']:,.0f})")

    # Improvement over baselines
    lines.extend([
        "",
        "## ARIMA vs Best Baseline",
        "",
        "| Horizon | Best Baseline | Baseline MAE | ARIMA MAE | Improvement |",
        "|---|---|---:|---:|---:|",
    ])

    for h in HORIZONS:
        h_rows = summary[summary["horizon"] == h]
        arima_row = h_rows[h_rows["model"] == "ARIMA"]
        baseline_rows = h_rows[h_rows["model"] != "ARIMA"]

        if len(arima_row) > 0 and len(baseline_rows) > 0:
            best_bl = baseline_rows.loc[baseline_rows["MAE"].idxmin()]
            arima_mae = arima_row["MAE"].values[0]
            bl_mae = best_bl["MAE"]
            improvement = (bl_mae - arima_mae) / bl_mae * 100
            sign = "+" if improvement > 0 else ""
            lines.append(
                f"| t+{h} | {best_bl['model']} | {bl_mae:,.0f} "
                f"| {arima_mae:,.0f} | {sign}{improvement:.1f}% |"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Positive improvement = ARIMA beats the baseline.",
        "- Negative improvement = baseline still wins (ARIMA overfits or struggles).",
        "- ARIMA should improve most at **longer horizons** where trend matters.",
        "- These results set the stage for **XGBoost (Week 7)** which can use all features.",
    ])

    return "\n".join(lines)


def print_summary(summary):
    """Print formatted summary to console."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON — BASELINES + ARIMA")
    print("=" * 80)

    for model_name in summary["model"].unique():
        print(f"\n  {model_name}:")
        model_rows = summary[summary["model"] == model_name]
        for _, row in model_rows.iterrows():
            print(f"    t+{row['horizon']:>2}:  MAE = {row['MAE']:>14,.0f}  "
                  f"RMSE = {row['RMSE']:>14,.0f}  "
                  f"sMAPE = {row['sMAPE']:>6.1f}%  "
                  f"(n={int(row['n_predictions'])})")

    print(f"\n  {'-' * 60}")
    print("  Best model per horizon (by MAE):")
    for h in HORIZONS:
        h_rows = summary[summary["horizon"] == h]
        best = h_rows.loc[h_rows["MAE"].idxmin()]
        print(f"    t+{h:>2}: {best['model']} (MAE = {best['MAE']:,.0f})")

    print("=" * 80)


def main():
    print("=== Week 6: ARIMA Walk-Forward Evaluation ===\n")

    # Load data
    print("Loading synthetic data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"  Loaded {len(df):,} rows, {df['business_id'].nunique()} businesses")

    # Setup models: baselines + ARIMA
    arima = ARIMAModel()
    models = ALL_BASELINES + [arima]
    print(f"  Models: {', '.join(m.name for m in models)}")
    print(f"  Horizons: {HORIZONS}")
    print(f"  Min history: {MIN_HISTORY} days, Step: {STEP_SIZE} days\n")

    # Evaluate each business
    all_results = []
    businesses = df["business_id"].unique()
    total_start = time.time()

    for i, biz_id in enumerate(businesses):
        biz_start = time.time()
        biz_data = df[df["business_id"] == biz_id]

        # Reset ARIMA for each new business (forces new auto_arima)
        arima.reset()

        results = evaluate_business(biz_data, models)
        all_results.extend(results)

        elapsed = time.time() - biz_start
        n_preds = len(results)
        arima_info = arima.summary()
        print(f"  {biz_id}: {n_preds} predictions ({elapsed:.1f}s) | ARIMA: {arima_info}")

    total_elapsed = time.time() - total_start
    results_df = pd.DataFrame(all_results)
    print(f"\n  Total predictions: {len(results_df):,}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

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

    # Print summary
    print_summary(summary)


if __name__ == "__main__":
    main()
