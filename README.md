# FX Triangular-Arbitrage Paper Trading

Automates the `FX_Strategy_Full.ipynb` notebook as a GitHub-hosted paper
trading book, following the same pattern as the regime-switching bot:
GitHub Actions runs the update script and GitHub Pages serves the dashboard.

## Strategy

- Trades a triangular FX spread built from `EURUSD=X`, `USDINR=X`, and
  `EURINR=X`.
- Computes the same detrended spread, z-score, rolling entry threshold,
  autocorrelation, trend, volatility, momentum, and carry-proxy features as
  the notebook.
- Uses a calibrated XGBoost classifier as the probability filter.
- Uses the notebook's probability hysteresis, spread-z entry logic, volatility
  targeting, trend overlay, turnover threshold, transaction cost, and slippage
  assumptions.
- Realizes today's P&L from yesterday's exposure, then makes today's decision
  for the next trading day to avoid lookahead.

## Retraining Cadence

The original notebook trained once on a 70/30 split. This automated version
uses the requested **weekly periodic retraining**:

- The model retrains on the first processed feature day of each ISO week.
- The training window is expanding and includes all completed feature rows
  before the current decision day.
- Missed scheduled runs are replayed in order on the next successful run.
- The model bundle is stored in `docs/data/model_bundle.joblib` so GitHub
  Actions can reuse it between daily runs.

## One-Time GitHub Setup

1. Create a new GitHub repo and push this folder to it.
2. Enable write permissions for Actions:
   Settings -> Actions -> General -> Workflow permissions -> Read and write.
3. Run the workflow manually once:
   Actions -> Daily FX paper trading update -> Run workflow.
4. Enable GitHub Pages:
   Settings -> Pages -> Deploy from a branch -> `main` / `/docs`.

The dashboard will be available at:

```text
https://<your-username>.github.io/<repo-name>/
```

## Local Run

```bash
pip install -r requirements.txt
python daily_update.py
```

The first run creates a bootstrap decision and saves state under `docs/data/`.
The next new feature day starts recording realized returns.

## Configuration

Tune all parameters in `config.py`:

- FX tickers and training start date
- minimum history required before training
- XGBoost parameters
- probability thresholds
- minimum position size
- hysteresis threshold
- transaction cost and slippage assumptions
- volatility target

## Files

- `config.py` - central parameters and paths
- `fx_lib.py` - data download, features, model training, prediction, signal logic
- `daily_update.py` - idempotent daily paper-trading runner
- `.github/workflows/daily.yml` - scheduled automation
- `docs/index.html` - static dashboard
- `docs/data/` - committed logs, state, and persisted model bundle

## Notes

This is paper trading only. It does not place real orders. Yahoo Finance FX
data can occasionally revise, lag, or fail; if a scheduled run fails, the next
successful run replays missed feature days in order.
