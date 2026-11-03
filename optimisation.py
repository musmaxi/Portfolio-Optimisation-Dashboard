"""
optimisation.py
---------------
Mean-variance optimisation: GMV, tangency (max-Sharpe), efficient frontier.

Two regimes:
  * Unconstrained (short selling allowed) – closed-form solutions, identical to the Excel model
        GMV:      w = Σ⁻¹1 / 1ᵀΣ⁻¹1
        Tangency: w = Σ⁻¹(μ − rf·1) / 1ᵀΣ⁻¹(μ − rf·1)
        Frontier: σ²(μp) = (A·μp² − 2B·μp + C) / D,
                  A = 1ᵀΣ⁻¹1, B = 1ᵀΣ⁻¹μ, C = μᵀΣ⁻¹μ, D = AC − B²
  * Constrained (long-only and/or a cap per stock) – numerical (SciPy SLSQP)

All inputs (μ, Σ, rf) must be in the SAME units – annual here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# --------------------------------------------------------------------------- #
# Portfolio statistics
# --------------------------------------------------------------------------- #
def portfolio_return(w, mu) -> float:
    return float(np.asarray(w) @ np.asarray(mu))


def portfolio_vol(w, cov) -> float:
    w = np.asarray(w)
    return float(np.sqrt(w @ np.asarray(cov) @ w))


def portfolio_stats(w, mu, cov, rf: float) -> dict:
    r, v = portfolio_return(w, mu), portfolio_vol(w, cov)
    return {"Return": r, "Volatility": v, "Sharpe": (r - rf) / v if v > 0 else np.nan}


def equal_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


# --------------------------------------------------------------------------- #
# Closed-form (short selling allowed)
# --------------------------------------------------------------------------- #
def frontier_constants(mu, cov) -> dict:
    mu, inv = np.asarray(mu), np.linalg.inv(np.asarray(cov))
    ones = np.ones(len(mu))
    A = ones @ inv @ ones
    B = ones @ inv @ mu
    C = mu @ inv @ mu
    return {"A": A, "B": B, "C": C, "D": A * C - B ** 2}


def gmv_closed_form(cov) -> np.ndarray:
    inv = np.linalg.inv(np.asarray(cov))
    ones = np.ones(len(inv))
    return inv @ ones / (ones @ inv @ ones)


def tangency_closed_form(mu, cov, rf: float) -> np.ndarray:
    mu, inv = np.asarray(mu), np.linalg.inv(np.asarray(cov))
    ones = np.ones(len(mu))
    gmv_ret = portfolio_return(gmv_closed_form(cov), mu)
    if gmv_ret <= rf:
        # If rf ≥ GMV return the formula returns the *minimum*-Sharpe point on
        # the lower branch – exactly the symptom from the original spreadsheet.
        raise ValueError(f"No tangency portfolio: GMV return ({gmv_ret:.2%}) ≤ rf ({rf:.2%}).")
    excess = mu - rf * ones
    return inv @ excess / (ones @ inv @ excess)


def frontier_vol_closed_form(target_returns, mu, cov) -> np.ndarray:
    k = frontier_constants(mu, cov)
    t = np.asarray(target_returns)
    return np.sqrt((k["A"] * t ** 2 - 2 * k["B"] * t + k["C"]) / k["D"])


# --------------------------------------------------------------------------- #
# Numerical (constrained)
# --------------------------------------------------------------------------- #
def _bounds(n, allow_short, max_weight):
    lo = None if allow_short else 0.0
    hi = max_weight if max_weight is not None else None
    return [(lo, hi)] * n


_SUM_TO_ONE = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def gmv_numerical(cov, allow_short=False, max_weight=None) -> np.ndarray:
    cov = np.asarray(cov); n = len(cov)
    res = minimize(lambda w: w @ cov @ w, equal_weights(n), method="SLSQP",
                   bounds=_bounds(n, allow_short, max_weight), constraints=[_SUM_TO_ONE],
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x


def tangency_numerical(mu, cov, rf, allow_short=False, max_weight=None) -> np.ndarray:
    mu, cov = np.asarray(mu), np.asarray(cov); n = len(mu)

    def neg_sharpe(w):
        v = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf) / v

    best = None
    rng = np.random.default_rng(0)
    starts = [equal_weights(n)] + [rng.dirichlet(np.ones(n)) for _ in range(5)]
    for x0 in starts:  # multiple starts guard against local optima
        res = minimize(neg_sharpe, x0, method="SLSQP", bounds=_bounds(n, allow_short, max_weight),
                       constraints=[_SUM_TO_ONE], options={"ftol": 1e-12, "maxiter": 1000})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise ValueError("Tangency optimisation did not converge.")
    return best.x


def min_var_for_target(mu, cov, target, allow_short=False, max_weight=None):
    mu, cov = np.asarray(mu), np.asarray(cov); n = len(mu)
    cons = [_SUM_TO_ONE, {"type": "eq", "fun": lambda w: w @ mu - target}]
    res = minimize(lambda w: w @ cov @ w, equal_weights(n), method="SLSQP",
                   bounds=_bounds(n, allow_short, max_weight), constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x if res.success else None


# --------------------------------------------------------------------------- #
# Public API used by the dashboard
# --------------------------------------------------------------------------- #
def optimise(mu: pd.Series, cov: pd.DataFrame, rf: float,
             allow_short: bool = True, max_weight: float | None = None) -> dict:
    """Return GMV, tangency and equal-weight portfolios (weights + stats)."""
    tickers = list(mu.index)
    closed_form = allow_short and max_weight is None
    if closed_form:
        w_gmv = gmv_closed_form(cov)
        w_tan = tangency_closed_form(mu, cov, rf)
    else:
        w_gmv = gmv_numerical(cov, allow_short, max_weight)
        w_tan = tangency_numerical(mu, cov, rf, allow_short, max_weight)
    w_eq = equal_weights(len(tickers))

    weights = pd.DataFrame({"GMV": w_gmv, "Tangency": w_tan, "Equal-weight": w_eq}, index=tickers)
    stats = pd.DataFrame({k: portfolio_stats(weights[k], mu, cov, rf) for k in weights}).T
    return {"weights": weights, "stats": stats, "method": "closed-form" if closed_form else "numerical"}


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, allow_short: bool = True,
                       max_weight: float | None = None, n_points: int = 60,
                       ret_max: float | None = None) -> pd.DataFrame:
    """Frontier from the GMV return up to ret_max (default: 1.5× the best stock, or best stock if long-only)."""
    mu_arr = np.asarray(mu)
    closed_form = allow_short and max_weight is None
    gmv = gmv_closed_form(cov) if closed_form else gmv_numerical(cov, allow_short, max_weight)
    r_gmv = portfolio_return(gmv, mu)

    if closed_form:
        hi = ret_max if ret_max is not None else max(1.5 * mu_arr.max(), r_gmv + 0.3)
        lo = r_gmv - (hi - r_gmv) * 0.6          # also draw part of the inefficient branch
        targets = np.linspace(lo, hi, n_points)
        return pd.DataFrame({"Return": targets,
                             "Volatility": frontier_vol_closed_form(targets, mu, cov),
                             "Efficient": targets >= r_gmv})

    hi = mu_arr.max() if ret_max is None else ret_max
    if max_weight is not None and not allow_short:  # highest return achievable under the cap
        res = minimize(lambda w: -(w @ mu_arr), equal_weights(len(mu_arr)), method="SLSQP",
                       bounds=_bounds(len(mu_arr), False, max_weight), constraints=[_SUM_TO_ONE])
        hi = min(hi, -res.fun)
    rows = []
    for t in np.linspace(r_gmv, hi - 1e-6, n_points):
        w = min_var_for_target(mu, cov, t, allow_short, max_weight)
        if w is not None:
            rows.append({"Return": t, "Volatility": portfolio_vol(w, cov), "Efficient": True})
    return pd.DataFrame(rows)


def random_portfolios(mu, cov, rf, n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Long-only Monte-Carlo portfolios for the feasible-set cloud."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(mu)), size=n)
    mu, cov = np.asarray(mu), np.asarray(cov)
    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    return pd.DataFrame({"Return": rets, "Volatility": vols, "Sharpe": (rets - rf) / vols})


def capital_market_line(rf: float, tangency_ret: float, tangency_vol: float, vol_max: float) -> pd.DataFrame:
    sr = (tangency_ret - rf) / tangency_vol
    vols = np.linspace(0, vol_max, 50)
    return pd.DataFrame({"Volatility": vols, "Return": rf + sr * vols})


if __name__ == "__main__":
    from data_pipeline import build_dataset, load_prices_csv
    md = build_dataset(load_prices_csv())
    out = optimise(md.mu, md.cov, rf=0.0438)
    print(out["weights"].round(4)); print(out["stats"].round(4))
