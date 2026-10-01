"""
baselines.py — Week 5: Baseline Forecasting Models
Three naive forecasters that set the performance floor for cash-flow prediction.

Each model implements:
    fit(train_series)         — optional, store any needed state
    predict(train_series, h)  — return a single float: predicted cash_balance at t+h

Models:
    1. LastValue:      predict(t+h) = cash_balance(t)
    2. MovingAverage7: predict(t+h) = mean(cash_balance[t-6 : t+1])
    3. MovingAverage30: predict(t+h) = mean(cash_balance[t-29 : t+1])
"""

import numpy as np


class LastValue:
    """Predict future cash balance = most recent observed value.
    
    Assumption: cash balance doesn't change. This is the simplest
    possible baseline — if cash is stable, nothing can beat this.
    """
    name = "Last Value"

    def fit(self, train_series):
        pass  # no fitting needed

    def predict(self, train_series, horizon):
        return train_series.iloc[-1]


class MovingAverage7:
    """Predict future cash balance = 7-day trailing average.
    
    Smooths out daily noise. Works well if cash balance
    fluctuates around a short-term mean.
    """
    name = "7-Day Average"

    def fit(self, train_series):
        pass

    def predict(self, train_series, horizon):
        window = train_series.iloc[-7:]
        return window.mean()


class MovingAverage30:
    """Predict future cash balance = 30-day trailing average.
    
    Captures longer-term trend. More stable but slower to adapt
    to recent changes. Works well if cash balance is mean-reverting.
    """
    name = "30-Day Average"

    def fit(self, train_series):
        pass

    def predict(self, train_series, horizon):
        window = train_series.iloc[-30:]
        return window.mean()


class LinearTrend:
    """Predict future cash balance by extrapolating the recent 30-day linear trend.
    
    Captures directional momentum — if cash has been declining,
    predicts continued decline. Important because naive averages
    can't capture directional risk (trending toward zero).
    """
    name = "Linear Trend (30d)"

    def fit(self, train_series):
        pass

    def predict(self, train_series, horizon):
        window = train_series.iloc[-30:]
        n = len(window)
        x = np.arange(n)
        y = window.values

        # Least-squares slope: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

        # Extrapolate from the last point
        last_val = y[-1]
        return last_val + slope * horizon


# Registry for easy iteration in walk_forward.py
ALL_BASELINES = [LastValue(), MovingAverage7(), MovingAverage30(), LinearTrend()]
