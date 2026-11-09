"""Multi-asset return loader: yfinance first, deterministic synthetic fallback."""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("hrp.data")

DEFAULT_UNIVERSE: dict[str, str] = {
    "SPY": "US large cap equity",
    "QQQ": "US growth / tech equity",
    "IWM": "US small cap equity",
    "EFA": "Developed ex-US equity",
    "EEM": "Emerging markets equity",
    "TLT": "US Treasuries 20y+",
    "IEF": "US Treasuries 7-10y",
    "LQD": "US investment grade credit",
    "HYG": "US high yield credit",
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "Crude oil",
    "DBC": "Broad commodities",
    "VNQ": "US real estate",
    "BTC-USD": "Bitcoin",
}

# asset class used only by the synthetic generator so the fake data still clusters sensibly
_CLASS = {
    "SPY": "eq", "QQQ": "eq", "IWM": "eq", "EFA": "eq", "EEM": "eq", "VNQ": "eq",
    "TLT": "rate", "IEF": "rate", "LQD": "credit", "HYG": "credit",
    "GLD": "metal", "SLV": "metal", "USO": "energy", "DBC": "energy", "BTC-USD": "crypto",
}
_BASE_VOL = {  # annualised
    "SPY": .16, "QQQ": .21, "IWM": .22, "EFA": .17, "EEM": .20, "VNQ": .21,
    "TLT": .15, "IEF": .07, "LQD": .09, "HYG": .09, "GLD": .15, "SLV": .28,
    "USO": .38, "DBC": .17, "BTC-USD": .62,
}
# loading of each class on the shared "risk-on" factor
_RISK_ON = {"eq": 1.0, "credit": 0.55, "rate": -0.35, "metal": 0.15, "energy": 0.45, "crypto": 0.5, "other": 0.3}


@dataclass
class MarketData:
    returns: pd.DataFrame
    prices: pd.Series  # last price per ticker
    source: str  # "yfinance" | "synthetic"
    notes: list[str]


_cache: dict[tuple, tuple[float, MarketData]] = {}
CACHE_TTL = int(os.getenv("HRP_CACHE_TTL", "900"))


def _seed(tickers: list[str], n: int) -> int:
    return int(hashlib.sha256(("|".join(sorted(tickers)) + str(n)).encode()).hexdigest()[:8], 16)


def synthetic_market(tickers: list[str], n_days: int) -> MarketData:
    """Factor-model returns: risk-on factor + class factor + idiosyncratic noise. Deterministic."""
    rng = np.random.default_rng(_seed(tickers, n_days))
    classes = {t: _CLASS.get(t, "other") for t in tickers}
    uniq = sorted(set(classes.values()))
    f_mkt = rng.standard_normal(n_days)
    f_cls = {c: rng.standard_normal(n_days) for c in uniq}
    cols = {}
    for t in tickers:
        c = classes[t]
        vol = _BASE_VOL.get(t, 0.2) / np.sqrt(252)
        beta = _RISK_ON[c]
        x = 0.55 * beta * f_mkt + 0.6 * f_cls[c] + 0.45 * rng.standard_normal(n_days)
        x = x / x.std() * vol + 0.0003
        cols[t] = x
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    rets = pd.DataFrame(cols, index=idx)
    prices = pd.Series({t: 50.0 + (int(hashlib.md5(t.encode()).hexdigest()[:4], 16) % 45000) / 100 for t in tickers})
    return MarketData(rets, prices, "synthetic", ["Synthetic factor-model data (yfinance unavailable or disabled)."])


def _yfinance_market(tickers: list[str], n_days: int) -> MarketData:
    import yfinance as yf

    period_days = int(n_days * 1.5) + 10
    raw = yf.download(
        tickers, period=f"{period_days}d", interval="1d", auto_adjust=True, progress=False, threads=True
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")
    close = raw["Close"] if "Close" in raw else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    notes: list[str] = []
    close = close.dropna(axis=1, how="all")
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        notes.append(f"No data for: {', '.join(missing)} (dropped).")
    # crypto trades on weekends, ETFs do not -> align on dates where most assets have prices
    close = close.dropna(axis=0, thresh=max(2, int(0.8 * close.shape[1])))
    close = close.ffill().dropna(axis=1)
    rets = close.pct_change().dropna().tail(n_days)
    if rets.shape[1] < 2 or rets.shape[0] < rets.shape[1] + 5:
        raise RuntimeError("too little usable history from yfinance")
    return MarketData(rets, close.iloc[-1], "yfinance", notes)


def load_market(tickers: list[str], lookback_days: int = 504, source: str | None = None) -> MarketData:
    """source: 'auto' (yfinance, fall back to synthetic), 'yfinance' (strict), 'synthetic'."""
    source = (source or os.getenv("HRP_DATA_SOURCE", "auto")).lower()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    key = (tuple(sorted(tickers)), lookback_days, source)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    if source == "synthetic":
        md = synthetic_market(tickers, lookback_days)
    else:
        try:
            md = _yfinance_market(tickers, lookback_days)
        except Exception as exc:  # network, rate limit, bad tickers ...
            if source == "yfinance":
                raise
            log.warning("yfinance failed (%s); using synthetic data", exc)
            md = synthetic_market(tickers, lookback_days)
            md.notes.insert(0, f"yfinance failed: {exc}")
    _cache[key] = (time.time(), md)
    return md


def clear_cache() -> None:
    _cache.clear()
