import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["HRP_DATA_SOURCE"] = "synthetic"

from app import hrp, rebalance
from app.data import synthetic_market


@pytest.fixture(scope="module")
def returns():
    return synthetic_market(["SPY", "QQQ", "TLT", "IEF", "GLD", "USO", "HYG", "BTC-USD"], 400).returns


def test_weights_sum_to_one_and_positive(returns):
    r = hrp.hrp_allocate(returns)
    assert r.weights.sum() == pytest.approx(1.0)
    assert (r.weights > 0).all()


def test_deterministic(returns):
    a = hrp.hrp_allocate(returns).weights
    b = hrp.hrp_allocate(returns).weights
    pd.testing.assert_series_equal(a, b)


def test_quasi_diag_is_permutation(returns):
    corr = hrp.cov_to_corr(hrp.estimate_cov(returns))
    order = hrp.quasi_diagonalize(hrp.cluster_linkage(corr))
    assert sorted(order) == list(range(returns.shape[1]))


def test_two_uncorrelated_assets_are_inverse_variance():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"A": rng.normal(0, 0.01, 2000), "B": rng.normal(0, 0.02, 2000)})
    w = hrp.hrp_allocate(df).weights
    cov = df.cov().values
    expected = (1 / np.diag(cov)) / (1 / np.diag(cov)).sum()
    assert w.values == pytest.approx(expected, rel=1e-6)


def test_lower_vol_than_equal_weight(returns):
    s = hrp.hrp_allocate(returns).stats
    assert s["vol_hrp"] < s["vol_equal_weight"]


def test_max_weight_cap(returns):
    w = hrp.hrp_allocate(returns, max_weight=0.25).weights
    assert w.max() <= 0.25 + 1e-9 and w.sum() == pytest.approx(1.0)


def test_infeasible_cap(returns):
    with pytest.raises(ValueError):
        hrp.hrp_allocate(returns, max_weight=0.05)


def test_ewma_and_ward_run(returns):
    r = hrp.hrp_allocate(returns, linkage_method="ward", cov_method="ewma")
    assert r.weights.sum() == pytest.approx(1.0)


def test_drift_and_trades():
    target = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    d = rebalance.compute_drift({"A": 0.6, "B": 0.25, "C": 0.15}, target)
    assert d.loc["A", "drift_pp"] == pytest.approx(10.0)
    trades = rebalance.build_trades(d, pd.Series({"A": 100.0, "B": 50.0, "C": 10.0}), 100_000)
    assert trades[0]["side"] == "SELL" and trades[0]["ticker"] == "A"
    assert sum(t["trade_value"] for t in trades if t["side"] == "BUY") == pytest.approx(
        sum(t["trade_value"] for t in trades if t["side"] == "SELL"), rel=1e-6
    )
    assert rebalance.trades_to_csv(trades).splitlines()[0].startswith("ticker,side")


def test_api(tmp_path, monkeypatch):
    monkeypatch.setenv("HRP_STATE_FILE", str(tmp_path / "p.json"))
    import importlib
    import app.main as m
    importlib.reload(m)
    from fastapi.testclient import TestClient
    c = TestClient(m.app)
    assert c.get("/api/health").json()["status"] == "ok"
    assert c.get("/").status_code == 200
    r = c.post("/api/rebalance", json={"threshold_pct": 3}).json()
    assert r["triggered"] and r["trades"] and r["csv"].startswith("ticker,")
    # after executing, the portfolio is on target and nothing triggers
    c.post("/api/portfolio/apply", json={})
    r2 = c.post("/api/rebalance", json={"threshold_pct": 3}).json()
    assert not r2["triggered"] and r2["trades"] == []
    assert c.post("/api/rebalance/csv", json={"threshold_pct": 3}).status_code == 204


def test_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("HRP_STATE_FILE", str(tmp_path / "p.json"))
    import importlib
    import app.main as m
    importlib.reload(m)
    from fastapi.testclient import TestClient
    c = TestClient(m.app)
    assert c.post("/api/optimize", json={}).status_code == 401
    assert c.post("/api/optimize", json={}, headers={"X-API-Key": "secret"}).status_code == 200
