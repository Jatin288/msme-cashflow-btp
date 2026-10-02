# Model Comparison — Walk-Forward Validation (Baselines + ARIMA)

> **Data:** 20 synthetic MSME businesses x 365 days
> **Validation:** Expanding window, min 90 days history, step 7 days
> **Target:** `cash_balance`
> **Horizons:** t+7, t+14, t+30, t+60

## Results

| Model | Horizon | MAE | RMSE | sMAPE (%) | # Predictions |
|---|---|---:|---:|---:|---:|
| 30-Day Average | t+7 | 3,946,993 | 8,970,759 | 38.0 | 780 |
| 30-Day Average | t+14 | 4,549,320 | 10,300,609 | 40.9 | 760 |
| 30-Day Average | t+30 | 5,699,548 | 11,800,719 | 49.0 | 700 |
| 30-Day Average | t+60 | 8,093,900 | 14,272,168 | 59.8 | 620 |
| 7-Day Average | t+7 | 3,475,077 | 7,400,303 | 37.4 | 780 |
| 7-Day Average | t+14 | 4,033,738 | 9,523,489 | 40.7 | 760 |
| 7-Day Average | t+30 | 5,087,606 | 12,184,343 | 44.9 | 700 |
| 7-Day Average | t+60 | 7,335,527 | 14,335,610 | 55.3 | 620 |
| ARIMA | t+7 | 3,570,299 | 7,238,916 | 38.7 | 780 |
| ARIMA | t+14 | 4,105,152 | 9,372,962 | 43.0 | 760 |
| ARIMA | t+30 | 5,292,916 | 12,820,809 | 47.1 | 700 |
| ARIMA | t+60 | 7,799,985 | 15,058,654 | 58.4 | 620 |
| Last Value | t+7 | 3,539,352 | 7,258,741 | 37.5 | 780 |
| Last Value | t+14 | 4,012,213 | 9,368,313 | 40.5 | 760 |
| Last Value | t+30 | 5,038,316 | 12,700,721 | 43.1 | 700 |
| Last Value | t+60 | 7,446,628 | 14,819,207 | 55.7 | 620 |
| Linear Trend (30d) | t+7 | 4,159,438 | 8,549,483 | 42.0 | 780 |
| Linear Trend (30d) | t+14 | 5,216,204 | 12,988,947 | 46.9 | 760 |
| Linear Trend (30d) | t+30 | 8,186,788 | 22,753,916 | 57.9 | 700 |
| Linear Trend (30d) | t+60 | 13,954,079 | 38,190,887 | 73.5 | 620 |

## Best Model Per Horizon (by MAE)

- **t+7:** 7-Day Average (MAE = 3,475,077)
- **t+14:** Last Value (MAE = 4,012,213)
- **t+30:** Last Value (MAE = 5,038,316)
- **t+60:** 7-Day Average (MAE = 7,335,527)

## ARIMA vs Best Baseline

| Horizon | Best Baseline | Baseline MAE | ARIMA MAE | Improvement |
|---|---|---:|---:|---:|
| t+7 | 7-Day Average | 3,475,077 | 3,570,299 | -2.7% |
| t+14 | Last Value | 4,012,213 | 4,105,152 | -2.3% |
| t+30 | Last Value | 5,038,316 | 5,292,916 | -5.1% |
| t+60 | 7-Day Average | 7,335,527 | 7,799,985 | -6.3% |

## Interpretation

- Positive improvement = ARIMA beats the baseline.
- Negative improvement = baseline still wins (ARIMA overfits or struggles).
- ARIMA should improve most at **longer horizons** where trend matters.
- These results set the stage for **XGBoost (Week 7)** which can use all features.