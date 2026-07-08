# -*- coding: utf-8 -*-
"""
Shared FX strategy logic used by daily_update.py.

This module reproduces the notebook's triangular-arbitrage feature set and
signal rules, but exposes them as daily, replayable functions for automation.
"""

import os

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    TARGET_VOL,
    TICKERS,
    TRAIN_START,
    XGB_PARAMS,
)


def download_fx_prices(end=None):
    """Download daily FX closes for EURUSD, USDINR, and EURINR."""
    ticker_list = list(TICKERS.values())
    data = yf.download(
        ticker_list,
        start=TRAIN_START,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    prices = pd.DataFrame(index=close.index)
    for name, ticker in TICKERS.items():
        prices[name] = close[ticker]

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.dropna().sort_index()


def compute_arbitrage(prices):
    """Compute the triangular implied EURUSD spread."""
    df = prices.copy()
    df["IMPLIED_EURUSD"] = df["EURINR"] / df["USDINR"]
    df["SPREAD"] = df["IMPLIED_EURUSD"] - df["EURUSD"]
    df["RET_EURUSD"] = df["EURUSD"].pct_change()
    return df.dropna()


def _rolling_trend(values):
    return np.polyfit(range(len(values)), values, 1)[0]


def create_features(df):
    """Reproduce the notebook feature engineering and ML target."""
    out = df.copy()

    out["spread_trend_slow"] = out["SPREAD"].rolling(120).mean()
    out["spread_detrended"] = out["SPREAD"] - out["spread_trend_slow"]

    out["spread_mean"] = out["spread_detrended"].rolling(20).mean()
    out["spread_std"] = out["spread_detrended"].rolling(20).std()
    out["spread_z"] = (out["spread_detrended"] - out["spread_mean"]) / out["spread_std"]

    out["z_abs"] = out["spread_z"].abs()
    out["z_entry_dynamic"] = out["z_abs"].rolling(120).quantile(0.15)

    out["spread_z_60"] = (
        (out["spread_detrended"] - out["spread_detrended"].rolling(60).mean())
        / out["spread_detrended"].rolling(60).std()
    )
    out["spread_change"] = out["spread_detrended"].diff()
    out["spread_vol"] = out["spread_detrended"].rolling(20).std()
    out["eurusd_vol"] = out["RET_EURUSD"].rolling(20).std()
    out["mom_5"] = out["EURUSD"].pct_change(5)
    out["rel_vol"] = out["spread_vol"] / out["eurusd_vol"]
    out["spread_autocorr"] = out["spread_detrended"].rolling(20).apply(
        lambda x: pd.Series(x).autocorr(),
        raw=False,
    )
    out["spread_trend"] = out["spread_detrended"].rolling(10).apply(_rolling_trend, raw=True)
    out["carry_proxy"] = out["EURINR"].pct_change(20) - out["USDINR"].pct_change(20)

    out["arb_pnl"] = -out["SPREAD"].shift(1) * out["SPREAD"].diff()
    out["target"] = (out["arb_pnl"] > 0).astype(int)
    out["spread_ret"] = -out["SPREAD"].diff() / out["EURUSD"]

    return out.replace([np.inf, -np.inf], np.nan).dropna()


def apply_dynamic_position_sizing(df, target_vol=TARGET_VOL):
    """Add the notebook's trend and volatility capital-weight overlay."""
    out = df.copy()
    out["sma_12m"] = out["EURUSD"].rolling(252).mean()
    out["trend_signal"] = np.where(out["EURUSD"] > out["sma_12m"], 1.0, 0.5)
    out["realized_vol"] = out["EURUSD"].pct_change().rolling(252).std() * np.sqrt(252)
    out["vol_target"] = (target_vol / out["realized_vol"]).clip(0.25, 2.5)
    out["capital_weight"] = out["trend_signal"] * out["vol_target"]
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def prepare_feature_frame(end=None):
    """Download prices and return the full automated strategy feature frame."""
    prices = download_fx_prices(end=end)
    arbitrage = compute_arbitrage(prices)
    features = create_features(arbitrage)
    return apply_dynamic_position_sizing(features)


def _calibrated_classifier(base_model):
    try:
        return CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_model, method="isotonic", cv=3)


def train_model(train_df):
    """Fit the scaler and calibrated XGBoost model on the provided history."""
    X = train_df[FEATURE_COLUMNS]
    y = train_df["target"]
    if y.nunique() < 2:
        raise ValueError("Need both positive and negative target classes to train the FX model.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base = XGBClassifier(**XGB_PARAMS)
    model = _calibrated_classifier(base)
    model.fit(X_scaled, y)
    return {"model": model, "scaler": scaler, "feature_columns": FEATURE_COLUMNS}


def predict_probability(row, bundle):
    """Predict the probability that the arbitrage setup is favorable."""
    X = pd.DataFrame([row[bundle["feature_columns"]].values], columns=bundle["feature_columns"])
    X_scaled = bundle["scaler"].transform(X)
    return float(bundle["model"].predict_proba(X_scaled)[0, 1])


def save_model(bundle, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path=MODEL_PATH):
    return joblib.load(path) if os.path.exists(path) else None


def calculate_decision(
    row,
    prob,
    prev_exposure,
    prev_prob_signal,
    prob_thresh,
    prob_reset_thresh,
    min_size,
    hysteresis,
):
    """Convert today's features and probability into tomorrow's exposure."""
    if prob > prob_thresh:
        prob_signal = 1
    elif prob < prob_reset_thresh:
        prob_signal = 0
    else:
        prob_signal = int(prev_prob_signal or 0)

    signal = 0
    if prob_signal == 1 and row["spread_z"] > row["z_entry_dynamic"]:
        signal = 1
    elif prob_signal == 1 and row["spread_z"] < -row["z_entry_dynamic"]:
        signal = -1

    safe_entry = max(float(row["z_entry_dynamic"]), 1e-8)
    z_mult = float(np.clip(row["z_abs"] / safe_entry, 1.0, 2.0))
    size = float(np.clip(min_size + (1.0 - min_size) * prob * z_mult, 0.0, 1.5))
    target_exposure = float(signal * size * row["capital_weight"])

    prior = 0.0 if prev_exposure is None else float(prev_exposure)
    if prev_exposure is None:
        exposure = target_exposure
    elif np.sign(target_exposure) != np.sign(prior) or abs(target_exposure - prior) > hysteresis:
        exposure = target_exposure
    else:
        exposure = prior

    turnover = abs(exposure - prior)
    return {
        "prob_signal": prob_signal,
        "signal": signal,
        "target_exposure": target_exposure,
        "exposure": float(exposure),
        "turnover": float(turnover),
    }
