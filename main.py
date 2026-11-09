"""FastAPI service: HRP optimizer, drift check, trade basket, dashboard."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data as market
from .hrp import hrp_allocate
from .rebalance import build_trades, compute_drift, now_iso, send_webhook, trades_to_csv

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
STATE_FILE = Path(os.getenv("HRP_STATE_FILE", ROOT / "data" / "portfolio.json"))
_lock = threading.Lock()

app = FastAPI(title="HRP Allocation Engine", version="1.0.0")


# --------------------------------------------------------------------------- #
# Auth (optional): set API_KEY and send it as X-API-Key
# --------------------------------------------------------------------------- #
def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "invalid or missing X-API-Key")


# --------------------------------------------------------------------------- #
# Portfolio state (current holdings, as weights)
# --------------------------------------------------------------------------- #
DEFAULT_PORTFOLIO = {
    "value": 250_000.0,
    "weights": {  # deliberately equity-heavy so the first run shows meaningful drift
        "SPY": 0.28, "QQQ": 0.18, "IWM": 0.06, "EFA": 0.08, "EEM": 0.04,
        "TLT": 0.06, "IEF": 0.05, "LQD": 0.04, "HYG": 0.03,
        "GLD": 0.06, "SLV": 0.02, "USO": 0.02, "DBC": 0.03, "VNQ": 0.03, "BTC-USD": 0.02,
    },
    "updated": None,
}


def load_portfolio() -> dict:
    with _lock:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    return dict(DEFAULT_PORTFOLIO)


def save_portfolio(p: dict) -> dict:
    p = {**p, "updated": now_iso()}
    with _lock:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(p, indent=2))
    return p


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class OptimizeRequest(BaseModel):
    tickers: list[str] | None = Field(None, description="Defaults to the tickers in the stored portfolio")
    lookback_days: int = Field(504, ge=60, le=2520)
    linkage: Literal["single", "complete", "average", "ward"] = "single"
    cov_method: Literal["sample", "ewma"] = "sample"
    max_weight: float | None = Field(None, gt=0, le=1)
    engine: Literal["native", "riskfolio", "pypfopt"] = "native"
    source: Literal["auto", "yfinance", "synthetic"] | None = None


class RebalanceRequest(OptimizeRequest):
    threshold_pct: float = Field(3.0, ge=0, description="Trigger when any |drift| exceeds this many percentage points")
    portfolio_value: float | None = Field(None, gt=0)
    current_weights: dict[str, float] | None = Field(None, description="Defaults to the stored portfolio")
    min_trade_value: float = Field(50.0, ge=0)
    notify: bool = Field(False, description="POST the summary to the manager webhook if triggered")
    webhook_url: str | None = None


class PortfolioIn(BaseModel):
    value: float = Field(gt=0)
    weights: dict[str, float]


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
def _external_weights(engine: str, returns: pd.DataFrame, linkage: str) -> pd.Series:
    """Cross-check engines. Optional deps: riskfolio-lib, PyPortfolioOpt."""
    try:
        if engine == "riskfolio":
            import riskfolio as rp

            port = rp.HCPortfolio(returns=returns)
            w = port.optimization(
                model="HRP", codependence="pearson", rm="MV", rf=0, linkage=linkage, leaf_order=True
            )
            return w.iloc[:, 0].reindex(returns.columns)
        if engine == "pypfopt":
            from pypfopt import HRPOpt

            w = HRPOpt(returns).optimize(linkage_method=linkage)
            return pd.Series(dict(w)).reindex(returns.columns)
    except ImportError as exc:
        raise HTTPException(422, f"engine '{engine}' needs an optional package: {exc}. See requirements-optional.txt")
    raise HTTPException(422, f"unknown engine {engine}")


def run_optimization(req: OptimizeRequest, tickers: list[str]):
    try:
        md = market.load_market(tickers, req.lookback_days, req.source)
    except Exception as exc:
        raise HTTPException(502, f"market data failed: {exc}")
    try:
        res = hrp_allocate(md.returns, req.linkage, req.cov_method, req.max_weight)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if req.engine != "native":
        ext = _external_weights(req.engine, md.returns, req.linkage)
        res.weights = ext / ext.sum()
        res.stats["engine_note"] = f"weights from {req.engine}; dendrogram/risk stats from the native implementation"
    return md, res


def _tickers(req: OptimizeRequest) -> list[str]:
    if req.tickers:
        return req.tickers
    return list(load_portfolio()["weights"].keys())


def _opt_payload(md: market.MarketData, res) -> dict:
    return {
        "as_of": now_iso(),
        "data_source": md.source,
        "notes": md.notes,
        "n_obs": int(md.returns.shape[0]),
        "window": [str(md.returns.index[0].date()), str(md.returns.index[-1].date())],
        "tickers": list(res.weights.index),
        "weights": {k: float(v) for k, v in res.weights.items()},
        "order": res.order,
        "dendrogram": res.dendrogram,
        "correlation": {"labels": res.order, "matrix": res.corr.loc[res.order, res.order].round(4).values.tolist()},
        "stats": res.stats,
        "names": {t: market.DEFAULT_UNIVERSE.get(t, t) for t in res.weights.index},
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok", "auth": bool(os.getenv("API_KEY"))}


@app.get("/api/config", dependencies=[Depends(require_key)])
def config():
    p = load_portfolio()
    return {
        "universe": market.DEFAULT_UNIVERSE,
        "portfolio": p,
        "defaults": {"lookback_days": 504, "linkage": "single", "cov_method": "sample", "threshold_pct": 3.0},
    }


@app.post("/api/data/refresh", dependencies=[Depends(require_key)])
def refresh_data(req: OptimizeRequest):
    """Pull multi-asset returns (the n8n 'pull returns' step). Warms the cache."""
    market.clear_cache()
    try:
        md = market.load_market(_tickers(req), req.lookback_days, req.source)
    except Exception as exc:
        raise HTTPException(502, f"market data failed: {exc}")
    return {
        "source": md.source,
        "tickers": list(md.returns.columns),
        "n_obs": int(md.returns.shape[0]),
        "window": [str(md.returns.index[0].date()), str(md.returns.index[-1].date())],
        "notes": md.notes,
    }


@app.post("/api/optimize", dependencies=[Depends(require_key)])
def optimize(req: OptimizeRequest):
    md, res = run_optimization(req, _tickers(req))
    return _opt_payload(md, res)


def _rebalance(req: RebalanceRequest) -> dict:
    portfolio = load_portfolio()
    current = req.current_weights or portfolio["weights"]
    value = req.portfolio_value or portfolio["value"]
    tickers = req.tickers or sorted(set(current) | set(portfolio["weights"]))
    md, res = run_optimization(req, tickers)

    drift = compute_drift(current, res.weights)
    max_abs = float(drift["drift_pp"].abs().max())
    triggered = max_abs > req.threshold_pct
    trades = build_trades(drift, md.prices, value, req.min_trade_value) if triggered else []
    csv_text = trades_to_csv(trades) if triggered else ""

    out = {
        "triggered": triggered,
        "threshold_pct": req.threshold_pct,
        "max_drift_pp": round(max_abs, 3),
        "breaches": [t for t, d in drift["drift_pp"].items() if abs(d) > req.threshold_pct],
        "portfolio_value": value,
        "drift": [
            {"ticker": t, "current": float(r.current), "target": float(r.target), "drift_pp": float(r.drift_pp)}
            for t, r in drift.iterrows()
        ],
        "trades": trades,
        "trade_count": len(trades),
        "turnover_pct": round(sum(t["trade_value"] for t in trades) / 2 / value * 100, 2) if trades else 0.0,
        "csv": csv_text,
        "optimization": _opt_payload(md, res),
        "webhook": None,
    }
    if triggered and req.notify:
        url = req.webhook_url or os.getenv("MANAGER_WEBHOOK_URL")
        if not url:
            out["webhook"] = {"sent": False, "error": "no webhook_url given and MANAGER_WEBHOOK_URL is not set"}
        else:
            summary = {
                "event": "hrp.rebalance.required",
                "as_of": out["optimization"]["as_of"],
                "max_drift_pp": out["max_drift_pp"],
                "threshold_pct": req.threshold_pct,
                "breaches": out["breaches"],
                "turnover_pct": out["turnover_pct"],
                "portfolio_value": value,
                "trades": trades,
                "csv": csv_text,
            }
            out["webhook"] = send_webhook(url, summary)
    return out


@app.post("/api/rebalance", dependencies=[Depends(require_key)])
def rebalance(req: RebalanceRequest):
    return _rebalance(req)


@app.post("/api/rebalance/csv", dependencies=[Depends(require_key)], response_class=PlainTextResponse)
def rebalance_csv(req: RebalanceRequest):
    out = _rebalance(req)
    if not out["triggered"]:
        return PlainTextResponse("", status_code=204)
    return PlainTextResponse(
        out["csv"],
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="rebalance_basket.csv"'},
    )


@app.get("/api/portfolio", dependencies=[Depends(require_key)])
def get_portfolio():
    return load_portfolio()


@app.put("/api/portfolio", dependencies=[Depends(require_key)])
def put_portfolio(p: PortfolioIn):
    total = sum(p.weights.values())
    if total <= 0:
        raise HTTPException(422, "weights must sum to a positive number")
    weights = {k.strip().upper(): v / total for k, v in p.weights.items() if v > 0}
    return save_portfolio({"value": p.value, "weights": weights})


@app.post("/api/portfolio/apply", dependencies=[Depends(require_key)])
def apply_targets(req: OptimizeRequest):
    """Simulate execution: set the stored holdings equal to the fresh HRP targets."""
    md, res = run_optimization(req, _tickers(req))
    p = load_portfolio()
    return save_portfolio({"value": p["value"], "weights": {k: float(v) for k, v in res.weights.items()}})


@app.post("/api/portfolio/reset", dependencies=[Depends(require_key)])
def reset_portfolio():
    return save_portfolio({k: v for k, v in DEFAULT_PORTFOLIO.items() if k != "updated"})


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
