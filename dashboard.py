"""
dashboard.py
------------
Interactive portfolio-optimisation dashboard (Streamlit).

Run locally:   streamlit run dashboard.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_pipeline as dp
import optimisation as opt
import risk

st.set_page_config(page_title="Portfolio Optimisation Dashboard", layout="wide")

COLORS = {"GMV": "#00A86B", "Tangency": "#F2A900", "Equal-weight": "#2F6FDE", "Custom": "#C0392B"}
PLOT_TEMPLATE = "plotly_white"
pct = lambda x: f"{x:.2%}"


# --------------------------------------------------------------------------- #
# Cached data access
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_csv_prices() -> pd.DataFrame:
    return dp.load_prices_csv()


@st.cache_data(show_spinner="Downloading prices from Yahoo Finance…", ttl=3600)
def get_yf_prices(tickers: tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:
    return dp.download_prices(list(tickers), str(start), str(end))


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("Settings")

st.sidebar.subheader("Data")
source = st.sidebar.radio("Source", ["Bundled CSV (NASDAQ, 2021–2026)", "Live download (yfinance)"],
                          help="The bundled file is the same data as the Excel model.")
if source.startswith("Bundled"):
    prices_all = get_csv_prices()
else:
    tick_text = st.sidebar.text_input("Tickers (comma-separated)", ", ".join(dp.TICKERS))
    tickers = tuple(t.strip().upper() for t in tick_text.split(",") if t.strip())
    c1, c2 = st.sidebar.columns(2)
    yf_start = c1.date_input("From", dt.date(2019, 1, 1), min_value=dt.date(2000, 1, 1))
    yf_end = c2.date_input("To", dt.date.today())
    try:
        prices_all = get_yf_prices(tickers, yf_start, yf_end)
    except Exception as e:  # network / ticker errors
        st.sidebar.error(f"Download failed ({e}). Falling back to the bundled CSV.")
        prices_all = get_csv_prices()
    if prices_all.empty or prices_all.shape[1] < 2:
        st.sidebar.error("Need at least two tickers with overlapping history. Using bundled CSV.")
        prices_all = get_csv_prices()

d_min, d_max = prices_all.index.min().date(), prices_all.index.max().date()
date_range = st.sidebar.slider("Estimation window", min_value=d_min, max_value=d_max,
                               value=(d_min, d_max), format="DD/MM/YYYY")
prices = dp.filter_dates(prices_all, *date_range)

method = st.sidebar.radio("Return type", ["log", "simple"], horizontal=True,
                          help="Log matches the Excel model. Simple returns aggregate exactly across stocks.")

st.sidebar.subheader("Optimisation")
rf = st.sidebar.number_input("Risk-free rate (annual, %)", 0.0, 20.0, 4.38, 0.05) / 100
constraint = st.sidebar.radio("Weights", ["Short selling allowed", "Long-only"],
                              help="Short selling uses the closed-form solution; long-only uses SciPy SLSQP.")
allow_short = constraint == "Short selling allowed"
max_weight = None
if not allow_short:
    cap = st.sidebar.slider("Max weight per stock", 0.20, 1.00, 1.00, 0.05,
                            help="1.00 = no cap. Must be ≥ 1/N to be feasible.")
    max_weight = None if cap >= 1.0 else cap

st.sidebar.subheader("Risk")
levels = st.sidebar.multiselect("Confidence levels", [0.90, 0.95, 0.975, 0.99], [0.95, 0.99],
                                format_func=lambda x: f"{x:.1%}")
levels = sorted(levels) or [0.95]
horizon = st.sidebar.number_input("Horizon (trading days)", 1, 20, 1)
port_value = st.sidebar.number_input("Portfolio value ($)", 1_000, 1_000_000_000, 1_000_000, 10_000)

# --------------------------------------------------------------------------- #
# Core calculations
# --------------------------------------------------------------------------- #
if len(prices) < 60:
    st.error("Estimation window too short — choose at least ~3 months of data.")
    st.stop()

md = dp.build_dataset(prices, method)
n = len(md.tickers)
if max_weight is not None and max_weight * n < 1:
    st.error(f"Max weight {max_weight:.0%} is infeasible for {n} stocks (needs ≥ {1/n:.0%}).")
    st.stop()

try:
    res = opt.optimise(md.mu, md.cov, rf, allow_short, max_weight)
except ValueError as e:
    st.error(f"{e} Lower the risk-free rate or change the estimation window.")
    st.stop()

weights, stats = res["weights"].copy(), res["stats"].copy()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Portfolio Optimisation Dashboard")
st.caption(f"{', '.join(md.tickers)} · {len(md.returns):,} daily {method} returns · "
           f"{md.prices.index[0]:%d/%m/%Y} → {md.prices.index[-1]:%d/%m/%Y} · "
           f"rf {rf:.2%} · {constraint.lower()} ({res['method']})")

tabs = st.tabs(["Overview", "Efficient Frontier", "Sharpe Comparison", "VaR & CVaR", "Scenario Analysis", "Data"])

# ---------------------------- Overview ------------------------------------- #
with tabs[0]:
    c = st.columns(4)
    t, g, e = stats.loc["Tangency"], stats.loc["GMV"], stats.loc["Equal-weight"]
    c[0].metric("Tangency return", pct(t.Return), f"Sharpe {t.Sharpe:.2f}", delta_color="off")
    c[1].metric("Tangency volatility", pct(t.Volatility))
    c[2].metric("GMV volatility", pct(g.Volatility), f"Return {g.Return:.2%}", delta_color="off")
    c[3].metric("Sharpe uplift vs 1/N", f"{t.Sharpe - e.Sharpe:+.2f}")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Portfolio weights")
        wl = weights.reset_index(names="Ticker").melt("Ticker", var_name="Portfolio", value_name="Weight")
        fig = px.bar(wl, x="Ticker", y="Weight", color="Portfolio", barmode="group",
                     color_discrete_map=COLORS, template=PLOT_TEMPLATE)
        fig.update_layout(yaxis_tickformat=".0%", height=380, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Summary")
        st.dataframe(stats.style.format({"Return": pct, "Volatility": pct, "Sharpe": "{:.3f}"}), width="stretch")
        st.dataframe(weights.style.format(pct), width="stretch")
        if allow_short and (weights.abs().sum() > 1.5).any():
            st.info("Unconstrained tangency weights are leveraged long/short — "
                    "switch to **Long-only** in the sidebar to compare.")

# ---------------------------- Frontier ------------------------------------- #
with tabs[1]:
    st.subheader("Efficient frontier & capital market line")
    with st.expander("Build a custom portfolio"):
        cols = st.columns(n)
        raw = [cols[i].number_input(tk, -2.0 if allow_short else 0.0, 2.0, round(1 / n, 2), 0.05, key=f"cw_{tk}")
               for i, tk in enumerate(md.tickers)]
        total = sum(raw)
        if abs(total) < 1e-9:
            st.warning("Weights sum to zero — using equal weights.")
            w_custom = opt.equal_weights(n)
        else:
            w_custom = np.array(raw) / total
            st.caption(f"Weights are rescaled to sum to 100% (entered total {total:.2f}).")
    custom_stats = opt.portfolio_stats(w_custom, md.mu, md.cov, rf)

    show_cloud = st.checkbox("Show random long-only portfolios", True)
    frontier = opt.efficient_frontier(md.mu, md.cov, allow_short, max_weight)
    vol_max = max(frontier["Volatility"].max(), md.vol.max()) * 1.05

    fig = go.Figure()
    if show_cloud:
        cloud = opt.random_portfolios(md.mu, md.cov, rf)
        fig.add_trace(go.Scatter(x=cloud.Volatility, y=cloud.Return, mode="markers", name="Random portfolios",
                                 marker=dict(size=4, color=cloud.Sharpe, colorscale="Viridis", opacity=0.45,
                                             colorbar=dict(title="Sharpe", x=1.02)),
                                 hovertemplate="σ %{x:.2%}<br>μ %{y:.2%}<extra></extra>"))
    eff, ineff = frontier[frontier.Efficient], frontier[~frontier.Efficient]
    if not ineff.empty:
        fig.add_trace(go.Scatter(x=ineff.Volatility, y=ineff.Return, mode="lines", name="Inefficient branch",
                                 line=dict(color="#1F3864", dash="dot", width=1.5)))
    fig.add_trace(go.Scatter(x=eff.Volatility, y=eff.Return, mode="lines", name="Efficient frontier",
                             line=dict(color="#1F3864", width=3)))
    cml = opt.capital_market_line(rf, t.Return, t.Volatility, vol_max)
    fig.add_trace(go.Scatter(x=cml.Volatility, y=cml.Return, mode="lines", name="Capital market line",
                             line=dict(color="#C0392B", dash="dash")))
    fig.add_trace(go.Scatter(x=md.vol, y=md.mu, mode="markers+text", text=md.tickers, textposition="top center",
                             name="Stocks", marker=dict(size=10, color="#7F7F7F")))
    for name, sym in [("GMV", "diamond"), ("Tangency", "star"), ("Equal-weight", "square")]:
        s = stats.loc[name]
        fig.add_trace(go.Scatter(x=[s.Volatility], y=[s.Return], mode="markers", name=name,
                                 marker=dict(size=16, symbol=sym, color=COLORS[name], line=dict(width=1, color="black"))))
    fig.add_trace(go.Scatter(x=[custom_stats["Volatility"]], y=[custom_stats["Return"]], mode="markers", name="Custom",
                             marker=dict(size=14, symbol="x", color=COLORS["Custom"])))
    fig.update_layout(template=PLOT_TEMPLATE, height=600, xaxis_title="Annualised volatility",
                      yaxis_title="Annualised return", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      xaxis_range=[0, vol_max], legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Custom portfolio: return {custom_stats['Return']:.2%}, volatility {custom_stats['Volatility']:.2%}, "
               f"Sharpe {custom_stats['Sharpe']:.3f}")

    if allow_short and max_weight is None:
        with st.expander("Closed-form method"):
            k = opt.frontier_constants(md.mu, md.cov)
            st.latex(r"w_{GMV}=\frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}\qquad"
                     r"w_{TAN}=\frac{\Sigma^{-1}(\mu-r_f\mathbf{1})}{\mathbf{1}^\top\Sigma^{-1}(\mu-r_f\mathbf{1})}")
            st.latex(r"\sigma^2(\mu_p)=\frac{A\mu_p^2-2B\mu_p+C}{D}")
            st.write({key: round(v, 4) for key, v in k.items()})

# ---------------------------- Sharpe --------------------------------------- #
with tabs[2]:
    st.subheader("Sharpe ratio comparison")
    stock_stats = pd.DataFrame({"Return": md.mu, "Volatility": md.vol})
    stock_stats["Sharpe"] = (stock_stats.Return - rf) / stock_stats.Volatility
    comp = pd.concat([stats, pd.DataFrame([custom_stats], index=["Custom"]), stock_stats])
    comp["Type"] = ["Portfolio"] * 4 + ["Stock"] * n
    fig = px.bar(comp.reset_index(names="Name"), x="Name", y="Sharpe", color="Type", text_auto=".2f",
                 color_discrete_map={"Portfolio": "#1F3864", "Stock": "#9DB3D9"}, template=PLOT_TEMPLATE)
    fig.update_layout(height=420, xaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(comp.drop(columns="Type").style.format({"Return": pct, "Volatility": pct, "Sharpe": "{:.3f}"}),
                 width="stretch")

    st.markdown("**Rolling 1-year Sharpe ratio**")
    port_rets = {k: risk.portfolio_returns(md.returns, weights[k], method) for k in weights}
    roll = pd.DataFrame({k: (r.rolling(252).mean() * 252 - rf) / (r.rolling(252).std(ddof=0) * np.sqrt(252))
                         for k, r in port_rets.items()}).dropna()
    if roll.empty:
        st.caption("Need more than one year of data.")
    else:
        fig = px.line(roll, color_discrete_map=COLORS, template=PLOT_TEMPLATE)
        fig.update_layout(height=350, yaxis_title="Sharpe", xaxis_title=None, legend_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("In-sample weights applied throughout; shows how stable the risk-adjusted return was over time.")

# ---------------------------- VaR & CVaR ----------------------------------- #
with tabs[3]:
    st.subheader(f"Value at Risk & Expected Shortfall — {horizon}-day horizon")
    rt = risk.risk_table(port_rets, levels, horizon, method)
    rt["Label"] = rt.Method + " " + rt.Measure + " @ " + rt.Confidence.map(lambda x: f"{x:.1%}")
    pivot = rt.pivot_table(index="Label", columns="Portfolio", values="Loss", sort=False)[weights.columns]
    pivot.index.name = None

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Loss as % of portfolio value**")
        st.dataframe(pivot.style.format(pct).background_gradient(cmap="Reds", axis=None), width="stretch")
    with c2:
        st.markdown(f"**Loss in \\$ on \\${port_value:,.0f}**")
        st.dataframe((pivot * port_value).style.format("${:,.0f}"), width="stretch")

    sel = st.selectbox("Distribution for", list(weights.columns), index=1)
    r_sel = risk.horizon_returns(port_rets[sel], horizon, method)
    fig = px.histogram(r_sel, nbins=80, template=PLOT_TEMPLATE, color_discrete_sequence=[COLORS[sel]])
    for a, dash in zip(levels, ["dash", "dot", "dashdot", "longdash"]):
        v, cv = risk.historical_var(r_sel, a), risk.historical_cvar(r_sel, a)
        fig.add_vline(x=-v, line_dash=dash, line_color="#C0392B", annotation_text=f"VaR {a:.0%}")
        fig.add_vline(x=-cv, line_dash=dash, line_color="black", annotation_text=f"CVaR {a:.0%}",
                      annotation_position="bottom left")
    fig.update_layout(showlegend=False, height=400, xaxis_tickformat=".1%",
                      xaxis_title=f"{horizon}-day {method} return", yaxis_title="Frequency")
    st.plotly_chart(fig, width="stretch")
    st.caption("Historical CVaR above parametric CVaR indicates fatter left tails than a Normal distribution. "
               "Cornish-Fisher adjusts the Normal quantile for skewness and kurtosis.")

    st.markdown("**Drawdowns**")
    dd = pd.DataFrame({k: risk.drawdown_series(r, method) for k, r in port_rets.items()})
    fig = px.area(dd, color_discrete_map=COLORS, template=PLOT_TEMPLATE)
    fig.update_traces(stackgroup=None, fill="tozeroy")
    fig.update_layout(height=320, yaxis_tickformat=".0%", xaxis_title=None, legend_title=None, yaxis_title="Drawdown")
    st.plotly_chart(fig, width="stretch")
    st.caption(" · ".join(f"{k} max drawdown {dd[k].min():.1%}" for k in dd))

# ---------------------------- Scenarios ------------------------------------ #
with tabs[4]:
    st.subheader("Historical stress windows")
    hs = risk.historical_scenarios(port_rets, method=method)
    avail = hs[hs.Available]
    missing = hs.loc[~hs.Available, "Scenario"].unique()
    if not avail.empty:
        cum = avail.pivot_table(index="Scenario", columns="Portfolio", values="Cumulative", sort=False)[weights.columns]
        worst = avail.pivot_table(index="Scenario", columns="Portfolio", values="Worst day", sort=False)[weights.columns]
        mdd = avail.pivot_table(index="Scenario", columns="Portfolio", values="Max drawdown", sort=False)[weights.columns]
        for label, tbl in [("Cumulative return", cum), ("Worst day", worst), ("Max drawdown", mdd)]:
            tbl.index.name = None
            st.markdown(f"**{label}**")
            st.dataframe(tbl.style.format(pct).background_gradient(cmap="RdYlGn", axis=None, vmin=-0.4, vmax=0.4),
                         width="stretch")

        scen = st.selectbox("Plot scenario", list(cum.index))
        s, e_ = risk.HISTORICAL_SCENARIOS[scen]
        wi = pd.DataFrame({k: risk.wealth_index(r.loc[s:e_], method) for k, r in port_rets.items()})
        fig = px.line(wi * 100, color_discrete_map=COLORS, template=PLOT_TEMPLATE)
        fig.update_layout(height=360, yaxis_title="Value of $100", xaxis_title=None, legend_title=None)
        st.plotly_chart(fig, width="stretch")
    if len(missing):
        st.info("Not in the selected data window: **" + "**, **".join(missing) + "**. "
                "Switch to *Live download (yfinance)* with an earlier start date to replay them "
                "(META only listed in May 2012, so 2008 needs the hypothetical test below).")
    st.caption("Weights are estimated in-sample, so historical scenario results include look-ahead bias.")

    st.subheader("Hypothetical stress test")
    preset = st.selectbox("Shock preset (S&P 500 peak-to-trough)", list(risk.HYPOTHETICAL_SHOCKS))
    base = risk.HYPOTHETICAL_SHOCKS[preset]
    use_beta = st.checkbox("Scale shock by each stock's beta", True,
                           help="Beta is measured against an equal-weight basket of the selected stocks. "
                                "Untick to apply the same shock to every stock (then every fully invested "
                                "portfolio loses exactly the shock).")
    betas = risk.betas(md.returns)
    default = betas * base if use_beta else pd.Series(base, index=md.tickers)
    if use_beta:
        st.caption("Betas: " + " · ".join(f"{k} {v:.2f}" for k, v in betas.items()))
    shock_df = st.data_editor(pd.DataFrame({"Shock": default.clip(lower=-0.99)}), width="stretch",
                              column_config={"Shock": st.column_config.NumberColumn(format="percent",
                                                                                     min_value=-1.0, max_value=1.0)},
                              key=f"shock_{preset}_{use_beta}")
    st_res = risk.stress_test(weights.assign(Custom=w_custom), shock_df["Shock"], port_value)
    c1, c2 = st.columns([1, 1.3])
    money = lambda x: f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"
    c1.dataframe(st_res.style.format({"Return": pct, "P&L": money}), width="stretch")
    fig = px.bar(st_res.reset_index(names="Portfolio"), x="Portfolio", y="Return", color="Portfolio",
                 color_discrete_map=COLORS, text_auto=".1%", template=PLOT_TEMPLATE)
    fig.update_layout(showlegend=False, height=320, yaxis_tickformat=".0%")
    c2.plotly_chart(fig, width="stretch")
    st.caption("Leveraged long/short portfolios can lose more (or less) than the market shock — "
               "short positions gain when prices fall.")

# ---------------------------- Data ----------------------------------------- #
with tabs[5]:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("**Growth of $100**")
        fig = px.line(md.prices / md.prices.iloc[0] * 100, template=PLOT_TEMPLATE)
        fig.update_layout(height=380, xaxis_title=None, yaxis_title=None, legend_title=None)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Correlation matrix**")
        fig = px.imshow(md.corr, text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1)
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Annualised statistics**")
    st.dataframe(pd.DataFrame({"Return": md.mu, "Volatility": md.vol}).style.format(pct), width="stretch")
    st.markdown("**Annualised covariance matrix**")
    st.dataframe(md.cov.style.format("{:.4f}"), width="stretch")

    c1, c2 = st.columns(2)
    c1.download_button("Download weights (CSV)", weights.to_csv().encode(), "weights.csv", "text/csv")
    c2.download_button("Download prices (CSV)", md.prices.to_csv().encode(), "prices.csv", "text/csv")

st.divider()
st.caption("For educational purposes only — not investment advice. Mean-variance results are highly sensitive "
           "to the estimated expected returns.")
