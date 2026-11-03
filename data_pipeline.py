"""
data_pipeline.py
----------------
Loading prices, computing returns and annualised statistics.

Two data sources are supported:
  * "csv"      – the bundled NASDAQ closing prices (data/prices.csv, 30/07/2021–29/07/2026),
                 identical to the Excel workbook's Data sheet.
  * "yfinance" – a live download from Yahoo Finance (any tickers / date range).

Conventions (match the Excel model):
  * mean returns are annualised by x252, covariance by x252, volatility by sqrt(252)
  * covariance uses the population formula (ddof=0), matching STDEV.P in Excel
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
TRADING_DAYS = 252
DATA_PATH = Path(__file__).parent / "data" / "prices.csv"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_prices_csv(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the bundled closing-price file (Date index, one column per ticker)."""
    prices = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    return prices.astype(float)


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted daily closing prices from Yahoo Finance."""
    import yfinance as yf  # imported lazily so the app runs without it

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices[[t for t in tickers if t in prices.columns]]
    prices.index.name = "Date"
    return prices.dropna(how="any").sort_index()


def filter_dates(prices: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    if start is not None:
        prices = prices.loc[pd.Timestamp(start):]
    if end is not None:
        prices = prices.loc[: pd.Timestamp(end)]
    return prices


# --------------------------------------------------------------------------- #
# Returns & statistics
# --------------------------------------------------------------------------- #
def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Daily returns. method='log' -> ln(Pt/Pt-1); method='simple' -> Pt/Pt-1 - 1."""
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    elif method == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    return rets.dropna(how="any")


def annualised_stats(returns: pd.DataFrame, periods: int = TRADING_DAYS):
    """Annualised mean vector, covariance matrix, volatility and correlation."""
    mu = returns.mean() * periods
    cov = returns.cov(ddof=0) * periods
    vol = np.sqrt(np.diag(cov))
    vol = pd.Series(vol, index=returns.columns)
    corr = returns.corr()
    return mu, cov, vol, corr


@dataclass
class MarketData:
    prices: pd.DataFrame
    returns: pd.DataFrame
    method: str
    mu: pd.Series
    cov: pd.DataFrame
    vol: pd.Series
    corr: pd.DataFrame

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)


def build_dataset(prices: pd.DataFrame, method: str = "log",
                  periods: int = TRADING_DAYS) -> MarketData:
    """Turn a price table into everything the optimiser and risk modules need."""
    returns = compute_returns(prices, method)
    mu, cov, vol, corr = annualised_stats(returns, periods)
    return MarketData(prices, returns, method, mu, cov, vol, corr)


if __name__ == "__main__":
    md = build_dataset(load_prices_csv())
    print(f"{len(md.returns)} daily returns, {md.prices.index[0].date()} → {md.prices.index[-1].date()}")
    print(pd.DataFrame({"Ann. Return": md.mu, "Ann. Vol": md.vol}).round(4))
