"""Drift detection, trade-basket generation, CSV export and manager webhook."""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
from datetime import datetime, timezone

import pandas as pd


def compute_drift(current: dict[str, float], target: pd.Series) -> pd.DataFrame:
    """Per-asset drift in percentage points (current - target). Weights are fractions."""
    cur = pd.Series(current, dtype=float).reindex(target.index).fillna(0.0)
    total = cur.sum()
    if total > 0:
        cur = cur / total  # tolerate holdings that don't add up to exactly 100 %
    df = pd.DataFrame({"current": cur, "target": target})
    df["drift_pp"] = (df["current"] - df["target"]) * 100.0
    return df


def _is_fractional(ticker: str) -> bool:
    return ticker.endswith("-USD")


def build_trades(
    drift: pd.DataFrame,
    prices: pd.Series,
    portfolio_value: float,
    min_trade_value: float = 50.0,
) -> list[dict]:
    """Trades that bring every asset back to its HRP target weight."""
    trades = []
    for t, row in drift.iterrows():
        delta_w = row["target"] - row["current"]
        value = delta_w * portfolio_value
        if abs(value) < min_trade_value:
            continue
        px = float(prices.get(t, float("nan")))
        qty = None
        if px == px and px > 0:
            qty = round(abs(value) / px, 6 if _is_fractional(t) else 0)
            if qty == 0:
                continue
        trades.append(
            {
                "ticker": t,
                "side": "BUY" if value > 0 else "SELL",
                "current_weight_pct": round(row["current"] * 100, 3),
                "target_weight_pct": round(row["target"] * 100, 3),
                "drift_pp": round(row["drift_pp"], 3),
                "trade_value": round(abs(value), 2),
                "price": round(px, 4) if px == px else None,
                "quantity": qty,
            }
        )
    # sells first so the basket funds itself, largest first within each side
    trades.sort(key=lambda r: (r["side"] != "SELL", -r["trade_value"]))
    return trades


CSV_FIELDS = [
    "ticker", "side", "quantity", "price", "trade_value",
    "current_weight_pct", "target_weight_pct", "drift_pp",
]


def trades_to_csv(trades: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(trades)
    return buf.getvalue()


def send_webhook(url: str, payload: dict, timeout: float = 10.0) -> dict:
    """POST the summary to the manager webhook. Signs the body if WEBHOOK_SECRET is set."""
    import httpx

    body = json.dumps(payload, separators=(",", ":"), default=str).encode()
    headers = {"Content-Type": "application/json"}
    secret = os.getenv("WEBHOOK_SECRET")
    if secret:
        headers["X-HRP-Signature"] = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    try:
        r = httpx.post(url, content=body, headers=headers, timeout=timeout)
        return {"sent": True, "status_code": r.status_code, "ok": r.is_success}
    except Exception as exc:
        return {"sent": False, "error": str(exc)}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
