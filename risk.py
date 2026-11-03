"""
risk.py
-------
Portfolio risk: VaR, CVaR (Expected Shortfall), drawdowns and scenario analysis.

Losses are reported as POSITIVE numbers (e.g. VaR = 0.043 means a 4.3% loss).

VaR methods
  * Historical     – empirical percentile of portfolio returns
  * Parametric     – Normal(μ, σ):  VaR = −(μ + zσ),  CVaR = −(μ − σ·φ(z)/(1−α)),  z = Φ⁻¹(1−α)
  * Cornish-Fisher – Normal quantile adjusted for skewness and excess kurtosis (fat tails)

Multi-day horizons are built from overlapping h-day returns (historical) or by scaling
μ by h and σ by √h (parametric) – the usual square-root-of-time rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


# --------------------------------------------------------------------------- #
# Return series
# --------------------------------------------------------------------------- #
def portfolio_returns(returns: pd.DataFrame, weights: pd.Series | np.ndarray, method: str = "log") -> pd.Series:
    """
    Daily portfolio returns.
    With simple returns the weighted sum is exact. With log returns we convert to simple,
    weight, then convert back to log so the series stays in the same units as the inputs.
    """
    w = np.asarray(weights)
    if method == "log":
        simple = np.expm1(returns)
        return pd.Series(np.log1p(simple.values @ w), index=returns.index)
    return pd.Series(returns.values @ w, index=returns.index)


def horizon_returns(r: pd.Series, horizon: int, method: str = "log") -> pd.Series:
    """Overlapping h-day returns."""
    if horizon <= 1:
        return r
    if method == "log":
        return r.rolling(horizon).sum().dropna()
    return np.expm1(np.log1p(r).rolling(horizon).sum()).dropna()


def to_simple(r, method: str):
    """Convert log-return figures to simple returns (identity for simple)."""
    return np.expm1(r) if method == "log" else r


# --------------------------------------------------------------------------- #
# VaR / CVaR
# --------------------------------------------------------------------------- #
def historical_var(r: pd.Series, alpha: float) -> float:
    return float(-np.percentile(r, 100 * (1 - alpha)))


def historical_cvar(r: pd.Series, alpha: float) -> float:
    cutoff = np.percentile(r, 100 * (1 - alpha))
    return float(-r[r <= cutoff].mean())


def parametric_var(mu: float, sigma: float, alpha: float) -> float:
    return float(-(mu + norm.ppf(1 - alpha) * sigma))


def parametric_cvar(mu: float, sigma: float, alpha: float) -> float:
    z = norm.ppf(1 - alpha)
    return float(-(mu - sigma * norm.pdf(z) / (1 - alpha)))


def cornish_fisher_var(r: pd.Series, alpha: float) -> float:
    z = norm.ppf(1 - alpha)
    s, k = skew(r), kurtosis(r)  # k = excess kurtosis
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return float(-(r.mean() + z_cf * r.std(ddof=0)))


def risk_table(port_rets: dict[str, pd.Series], levels=(0.95, 0.99), horizon: int = 1,
               method: str = "log") -> pd.DataFrame:
    """VaR / CVaR for each portfolio, method and confidence level (as fractional losses)."""
    rows = []
    for name, r in port_rets.items():
        rh = horizon_returns(r, horizon, method)
        mu_h, sd_h = r.mean() * horizon, r.std(ddof=0) * np.sqrt(horizon)
        for a in levels:
            rows += [
                {"Portfolio": name, "Confidence": a, "Method": "Historical", "Measure": "VaR", "Loss": historical_var(rh, a)},
                {"Portfolio": name, "Confidence": a, "Method": "Historical", "Measure": "CVaR", "Loss": historical_cvar(rh, a)},
                {"Portfolio": name, "Confidence": a, "Method": "Parametric", "Measure": "VaR", "Loss": parametric_var(mu_h, sd_h, a)},
                {"Portfolio": name, "Confidence": a, "Method": "Parametric", "Measure": "CVaR", "Loss": parametric_cvar(mu_h, sd_h, a)},
                {"Portfolio": name, "Confidence": a, "Method": "Cornish-Fisher", "Measure": "VaR", "Loss": cornish_fisher_var(rh, a)},
            ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Drawdowns
# --------------------------------------------------------------------------- #
def wealth_index(r: pd.Series, method: str = "log", start: float = 1.0) -> pd.Series:
    growth = np.exp(r.cumsum()) if method == "log" else (1 + r).cumprod()
    return start * growth


def drawdown_series(r: pd.Series, method: str = "log") -> pd.Series:
    w = wealth_index(r, method)
    return w / w.cummax() - 1


def max_drawdown(r: pd.Series, method: str = "log") -> float:
    return float(drawdown_series(r, method).min())


# --------------------------------------------------------------------------- #
# Scenario analysis
# --------------------------------------------------------------------------- #
# Historical stress windows. The 2008 and COVID windows pre-date the bundled data
# (30/07/2021 onwards); they become available when you download a longer history
# with yfinance. META only listed in May 2012, so 2008 cannot be replayed for it –
# use the hypothetical stress test instead.
HISTORICAL_SCENARIOS = {
    "2008 GFC (Lehman → market low)": ("2008-09-12", "2009-03-09"),
    "COVID-19 crash": ("2020-02-19", "2020-03-23"),
    "2022 rate-hike tech sell-off": ("2022-01-03", "2022-12-30"),
    "2023 AI-led rally": ("2023-01-03", "2023-12-29"),
    "Apr-2025 tariff shock": ("2025-04-02", "2025-04-08"),
}

# Hypothetical shocks: the S&P 500 peak-to-trough fall in each crisis, applied uniformly
# to every stock by default (editable per stock in the dashboard).
#   GFC:   S&P 500 −56.8% (9 Oct 2007 → 9 Mar 2009)
#   COVID: S&P 500 −33.9% (19 Feb 2020 → 23 Mar 2020)
HYPOTHETICAL_SHOCKS = {
    "2008 GFC-style (−56.8%)": -0.568,
    "COVID-style (−33.9%)": -0.339,
    "Moderate correction (−10%)": -0.10,
}


def historical_scenarios(port_rets: dict[str, pd.Series], scenarios: dict = HISTORICAL_SCENARIOS,
                         method: str = "log") -> pd.DataFrame:
    rows = []
    for scen, (start, end) in scenarios.items():
        for name, r in port_rets.items():
            first, last = r.index.min(), r.index.max()
            covered = pd.Timestamp(start) >= first and pd.Timestamp(end) <= last
            window = r.loc[start:end]
            if not covered or window.empty:
                rows.append({"Scenario": scen, "Start": start, "End": end, "Portfolio": name,
                             "Available": False, "Cumulative": np.nan, "Worst day": np.nan, "Max drawdown": np.nan})
                continue
            cum = np.expm1(window.sum()) if method == "log" else (1 + window).prod() - 1
            rows.append({"Scenario": scen, "Start": start, "End": end, "Portfolio": name, "Available": True,
                         "Cumulative": float(cum), "Worst day": float(to_simple(window.min(), method)),
                         "Max drawdown": max_drawdown(window, method)})
    return pd.DataFrame(rows)


def betas(returns: pd.DataFrame, market: pd.Series | None = None) -> pd.Series:
    """Beta of each stock to a market proxy (default: equal-weight basket of the stocks)."""
    m = returns.mean(axis=1) if market is None else market
    return returns.apply(lambda s: s.cov(m) / m.var())


def stress_test(weights: pd.DataFrame, shocks: pd.Series, value: float = 1.0) -> pd.DataFrame:
    """Instant P&L from per-stock price shocks (simple returns)."""
    pnl = weights.T @ shocks.reindex(weights.index).fillna(0.0)
    return pd.DataFrame({"Return": pnl, "P&L": pnl * value})
