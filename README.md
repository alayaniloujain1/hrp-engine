# HRP Allocation Engine

**Hierarchical Risk Parity & graph-clustered portfolio allocation**, packaged as a web app + API + n8n automation.

```
n8n weekly cron ─► pull multi-asset returns ─► FastAPI HRP dendrogram optimizer
                 ─► IF weight drift > 3% ─► trade-basket CSV  +  manager webhook
```

| Layer | What it does |
|---|---|
| `app/hrp.py` | Correlation distance → SciPy linkage (dendrogram) → quasi-diagonalization → recursive bisection. Optional weight cap, sample or EWMA covariance, single/average/complete/ward linkage. |
| `app/data.py` | yfinance returns (cached), with a deterministic synthetic fallback so the app runs offline. |
| `app/rebalance.py` | Drift per asset, trade basket (sells first), CSV, HMAC-signed webhook. |
| `app/main.py` | FastAPI: `/api/optimize`, `/api/rebalance`, `/api/rebalance/csv`, `/api/data/refresh`, `/api/portfolio*`. Docs at `/docs`. |
| `static/index.html` | Dashboard: dendrogram, quasi-diagonal correlation heatmap, holdings vs target, trade basket, CSV download, webhook button. |
| `n8n/hrp_weekly_rebalance.json` | Importable workflow (Mon 07:00 cron → refresh → optimize → IF → CSV + webhook). |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional
uvicorn app.main:app --reload   # http://localhost:8000
```

Or `docker compose up --build` (API on :8000, n8n on :5678).

Tests: `pip install -r requirements-dev.txt && pytest`

## Using it

1. Edit the universe, lookback, linkage and drift trigger in the sidebar, then **Run optimization**.
2. Type your real current weights into the *Current %* column and **Save holdings** (stored in `data/portfolio.json`).
3. If any asset drifts more than the trigger (default 3 percentage points), the trade basket appears. Download the CSV or send it to your webhook.
4. **Mark executed** simulates filling the basket: stored holdings become the HRP targets.

Drift is measured as `|current weight − HRP target|` in percentage points. When any asset breaches, *every* asset is traded back to target.

## n8n

Import `n8n/hrp_weekly_rebalance.json` (Workflows → Import from file). Then edit three things: the `X-API-Key` header value (match `API_KEY` in `.env`), the API host if you are not using the compose service name `hrp-api`, and the `Manager webhook` URL. The webhook body is `{event, as_of, max_drift_pp, breaches, turnover_pct, trades[], csv}`; if `WEBHOOK_SECRET` is set on the API, sign-check `X-HRP-Signature` (`sha256=` + HMAC of the body).

Alternatively set `MANAGER_WEBHOOK_URL` and let the API send it: call `/api/rebalance` with `"notify": true`.

## Riskfolio-Lib / PyPortfolioOpt

The default engine is the native implementation (no heavy dependencies). To cross-check against the libraries, `pip install -r requirements-optional.txt` and choose the engine in the sidebar or pass `"engine": "riskfolio"` / `"pypfopt"`. Weights then come from the library; the dendrogram and risk stats still come from the native code.

## Notes and limits

* Data source `auto` falls back to synthetic data if yfinance fails. The header pill shows which one you got. Use `yfinance only` for production.
* Whole-share quantities use the last close; crypto (`-USD`) is fractional. Review the basket before sending any orders: this tool does not place trades.
* Not investment advice. HRP uses historical covariance; results depend on the window and linkage.
