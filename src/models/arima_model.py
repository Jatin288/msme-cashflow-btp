"""
arima_model.py — Week 6: ARIMA Forecasting Model
Wraps statsmodels ARIMA with auto-order selection via pmdarima.

Design:
    - Auto-ARIMA runs ONCE per business (on the first training window)
      to find optimal (p,d,q) order — avoids running auto_arima at every
      evaluation step (which would be extremely slow).
    - At each walk-forward step, ARIMA is re-fitted with the fixed order
      on the current training window, then forecasts h steps ahead.
    - Falls back to Last Value baseline if ARIMA fails (convergence issues,
      singular matrix, etc.) — robustness over perfection.

Interface matches baselines.py:
    fit(train_series)         — fit ARIMA on training data
    predict(train_series, h)  — forecast h steps ahead
"""

import warnings
import numpy as np

# Suppress convergence warnings during walk-forward (hundreds of fits)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ARIMAModel:
    """ARIMA forecaster with auto-order selection.
    
    On first call to fit(), uses pmdarima.auto_arima to find the best
    (p,d,q) order. Subsequent fits reuse that order for speed.
    """
    name = "ARIMA"

    def __init__(self):
        self.order = None           # (p, d, q) — set by auto_arima
        self.fitted_model = None
        self._order_determined = False
        self._fit_failures = 0
        self._fit_total = 0

    def _determine_order(self, train_series):
        """Run auto_arima once to find optimal (p,d,q) order."""
        import pmdarima as pm

        try:
            auto_model = pm.auto_arima(
                train_series.values,
                start_p=0, max_p=5,
                start_q=0, max_q=5,
                d=None,             # let auto_arima determine d
                max_d=2,
                seasonal=False,     # daily cash-flow — no obvious seasonality at ARIMA level
                stepwise=True,      # faster search
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
            )
            self.order = auto_model.order
        except Exception:
            # Default fallback if auto_arima fails
            self.order = (1, 1, 1)

        self._order_determined = True

    def fit(self, train_series):
        """Fit ARIMA on training data.
        
        First call: runs auto_arima to determine (p,d,q).
        Subsequent calls: reuses the order for speed.
        """
        from statsmodels.tsa.arima.model import ARIMA

        if not self._order_determined:
            self._determine_order(train_series)

        self._fit_total += 1

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(train_series.values, order=self.order)
                self.fitted_model = model.fit()
        except Exception:
            self.fitted_model = None
            self._fit_failures += 1

    def predict(self, train_series, horizon):
        """Forecast cash_balance at t+horizon.
        
        Re-fits ARIMA on the current training window, then forecasts.
        Falls back to last value if fitting fails.
        """
        self.fit(train_series)

        if self.fitted_model is not None:
            try:
                forecast = self.fitted_model.forecast(steps=horizon)
                return forecast[-1]  # value at t+h
            except Exception:
                pass

        # Fallback: last observed value (graceful degradation)
        return train_series.iloc[-1]

    def reset(self):
        """Reset for a new business (re-run auto_arima)."""
        self.order = None
        self.fitted_model = None
        self._order_determined = False
        self._fit_failures = 0
        self._fit_total = 0

    def summary(self):
        """Return a summary string of fitting performance."""
        if self._fit_total == 0:
            return "No fits attempted"
        fail_pct = self._fit_failures / self._fit_total * 100
        return (f"Order: {self.order}, "
                f"Fits: {self._fit_total}, "
                f"Failures: {self._fit_failures} ({fail_pct:.1f}%)")
