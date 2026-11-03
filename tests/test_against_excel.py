"""
Checks that the Python model reproduces the corrected Excel workbook
(log returns, population covariance, rf = 4.38%, short selling allowed).

Run:  pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_pipeline as dp  # noqa: E402
import optimisation as opt  # noqa: E402
import risk  # noqa: E402

RF = 0.0438


@pytest.fixture(scope="module")
def md():
    return dp.build_dataset(dp.load_prices_csv(), "log")


@pytest.fixture(scope="module")
def res(md):
    return opt.optimise(md.mu, md.cov, RF)


def test_data_shape(md):
    assert len(md.returns) == 1253
    assert md.tickers == dp.TICKERS


def test_gmv_matches_excel(res):
    w = res["weights"]["GMV"].values
    np.testing.assert_allclose(w, [0.4066, 0.4595, 0.1919, -0.0078, -0.0503], atol=1e-4)
    s = res["stats"].loc["GMV"]
    assert s.Return == pytest.approx(0.1278, abs=1e-4)
    assert s.Volatility == pytest.approx(0.2382, abs=1e-4)


def test_tangency_matches_excel(res):
    w = res["weights"]["Tangency"].values
    np.testing.assert_allclose(w, [1.2981, -0.7557, 1.2466, -0.7429, -0.0461], atol=1e-4)
    s = res["stats"].loc["Tangency"]
    assert s.Sharpe == pytest.approx(0.6735, abs=1e-4)


def test_tangency_above_gmv(res):
    st = res["stats"]
    assert st.loc["Tangency", "Sharpe"] > st.loc["GMV", "Sharpe"]
    assert st.loc["Tangency", "Return"] > st.loc["GMV", "Return"]


def test_closed_form_equals_numerical(md):
    w_cf = opt.gmv_closed_form(md.cov)
    w_num = opt.gmv_numerical(md.cov, allow_short=True)
    np.testing.assert_allclose(w_cf, w_num, atol=1e-4)
    t_cf = opt.tangency_closed_form(md.mu, md.cov, RF)
    t_num = opt.tangency_numerical(md.mu, md.cov, RF, allow_short=True)
    np.testing.assert_allclose(t_cf, t_num, atol=1e-3)


def test_long_only_weights_valid(md):
    out = opt.optimise(md.mu, md.cov, RF, allow_short=False, max_weight=0.4)
    w = out["weights"]
    assert (w >= -1e-8).all().all() and (w <= 0.4 + 1e-6).all().all()
    np.testing.assert_allclose(w.sum().values, 1.0, atol=1e-6)


def test_unit_mismatch_bug_reproduced(md):
    """The original spreadsheet bug: daily μ minus annual rf -> tangency ≈ GMV."""
    mu_daily = md.mu / 252
    cov_daily = md.cov / 252
    inv = np.linalg.inv(cov_daily)
    ex = mu_daily.values - RF
    w_bug = inv @ ex / inv.dot(ex).sum()
    np.testing.assert_allclose(w_bug, [0.3998, 0.4689, 0.1838, -0.0022, -0.0503], atol=1e-4)


def test_var_close_to_excel(md, res):
    # Excel weights log returns directly; Python aggregates in simple-return space,
    # so allow a small tolerance.
    r = risk.portfolio_returns(md.returns, res["weights"]["Tangency"], "log")
    assert risk.historical_var(r, 0.95) == pytest.approx(0.0429, abs=0.002)
    assert risk.historical_var(r, 0.99) == pytest.approx(0.0753, abs=0.004)
    assert risk.historical_cvar(r, 0.95) > risk.historical_var(r, 0.95)
