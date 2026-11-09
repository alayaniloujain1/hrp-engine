"""Hierarchical Risk Parity (Lopez de Prado, 2016) in plain NumPy / SciPy.

Pipeline
--------
1. correlation distance      d_ij = sqrt(0.5 * (1 - rho_ij))
2. hierarchical clustering   scipy.cluster.hierarchy.linkage  (the dendrogram)
3. quasi-diagonalization     reorder assets so similar ones sit next to each other
4. recursive bisection       split the ordered list in half, allocate risk
                             inversely to each half's cluster variance
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

LinkageMethod = Literal["single", "complete", "average", "ward"]
CovMethod = Literal["sample", "ewma"]

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def estimate_cov(returns: pd.DataFrame, method: CovMethod = "sample", halflife: int = 63) -> pd.DataFrame:
    """Daily covariance matrix. `ewma` weights recent observations more."""
    if method == "sample":
        return returns.cov()
    if method == "ewma":
        n = len(returns)
        w = 0.5 ** (np.arange(n)[::-1] / halflife)
        w = w / w.sum()
        x = returns.values - np.average(returns.values, axis=0, weights=w)
        cov = (x * w[:, None]).T @ x / (1 - (w**2).sum())
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    raise ValueError(f"unknown cov method: {method}")


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    sd = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(sd, sd)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return dist


def cluster_linkage(corr: pd.DataFrame, method: LinkageMethod = "single") -> np.ndarray:
    """Linkage matrix (the dendrogram) built from the correlation distance."""
    return linkage(squareform(correlation_distance(corr), checks=False), method=method)


def quasi_diagonalize(link: np.ndarray) -> list[int]:
    """Leaf order of the dendrogram: the quasi-diagonal ordering of the assets."""
    n = link.shape[0] + 1
    stack = [2 * n - 2]
    order: list[int] = []
    while stack:
        node = stack.pop()
        if node < n:
            order.append(node)
        else:
            left, right = link[node - n, :2]
            stack.append(int(right))
            stack.append(int(left))
    return order


def cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    w = np.ones(cov.shape[0])
    clusters = [list(order)]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            h = len(c) // 2
            a, b = c[:h], c[h:]
            va, vb = cluster_variance(cov, a), cluster_variance(cov, b)
            alpha = 1.0 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1.0 - alpha
            nxt += [a, b]
        clusters = nxt
    return w / w.sum()


def cap_weights(w: np.ndarray, max_weight: float) -> np.ndarray:
    """Clip to `max_weight` and redistribute the excess pro rata to the uncapped names."""
    n = len(w)
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError(f"max_weight={max_weight} is infeasible for {n} assets (need >= {1 / n:.4f})")
    w = w.copy()
    for _ in range(n + 1):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        free = ~over & (w < max_weight - 1e-12)
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Risk analytics
# --------------------------------------------------------------------------- #
def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    """Annualised volatility."""
    return float(np.sqrt(w @ cov @ w * TRADING_DAYS))


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fraction of portfolio variance contributed by each asset (sums to 1)."""
    marginal = cov @ w
    rc = w * marginal
    return rc / rc.sum()


def diversification_ratio(w: np.ndarray, cov: np.ndarray) -> float:
    vols = np.sqrt(np.diag(cov))
    return float((w @ vols) / np.sqrt(w @ cov @ w))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclass
class HRPResult:
    weights: pd.Series
    order: list[str]
    linkage: np.ndarray
    corr: pd.DataFrame
    cov: pd.DataFrame
    dendrogram: dict
    stats: dict


def hrp_allocate(
    returns: pd.DataFrame,
    linkage_method: LinkageMethod = "single",
    cov_method: CovMethod = "sample",
    max_weight: float | None = None,
) -> HRPResult:
    if returns.shape[1] < 2:
        raise ValueError("HRP needs at least 2 assets")
    if returns.shape[0] < returns.shape[1] + 5:
        raise ValueError("not enough observations for the number of assets")

    cov = estimate_cov(returns, cov_method)
    corr = cov_to_corr(cov)
    link = cluster_linkage(corr, linkage_method)
    order_ix = quasi_diagonalize(link)
    w = recursive_bisection(cov.values, order_ix)
    if max_weight is not None:
        w = cap_weights(w, max_weight)

    tickers = list(returns.columns)
    weights = pd.Series(w, index=tickers, name="weight")
    order = [tickers[i] for i in order_ix]

    d = dendrogram(link, labels=tickers, no_plot=True)
    dendro = {
        "icoord": [list(map(float, r)) for r in d["icoord"]],
        "dcoord": [list(map(float, r)) for r in d["dcoord"]],
        "leaves": [str(x) for x in d["ivl"]],
    }

    c = cov.values
    n = len(tickers)
    ew = np.full(n, 1.0 / n)
    iv = 1.0 / np.diag(c)
    iv /= iv.sum()
    rc = risk_contributions(w, c)
    stats = {
        "vol_hrp": portfolio_vol(w, c),
        "vol_equal_weight": portfolio_vol(ew, c),
        "vol_inverse_vol": portfolio_vol(iv, c),
        "diversification_ratio": diversification_ratio(w, c),
        "effective_n": float(1.0 / (w**2).sum()),
        "risk_contribution": {t: float(r) for t, r in zip(tickers, rc)},
        "asset_vol": {t: float(np.sqrt(c[i, i] * TRADING_DAYS)) for i, t in enumerate(tickers)},
    }
    return HRPResult(weights, order, link, corr, cov, dendro, stats)
