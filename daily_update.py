# -*- coding: utf-8 -*-
"""
Run once per trading day after the latest Yahoo Finance daily FX close is
available. Safe to re-run: it no-ops when there is no new feature day and
replays missed feature days in order.

The XGBoost probability model retrains weekly on an expanding window using
only completed rows before the current decision day. Today's decision is the
exposure that earns the next feature day's spread return.
"""

import json
import os

import pandas as pd

from config import (
    EXPOSURE_HYSTERESIS,
    MIN_HISTORY_DAYS,
    MIN_SIZE,
    MODEL_PATH,
    PROB_RESET_THRESH,
    PROB_THRESH,
    SLIP_BPS,
    STATE_PATH,
    TC_BPS,
    TRADE_LOG_PATH,
)
from fx_lib import (
    calculate_decision,
    load_model,
    predict_probability,
    prepare_feature_frame,
    save_model,
    train_model,
)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "prev_exposure": None,
        "prev_prob_signal": 0,
        "pending_cost": 0.0,
        "equity": 1.0,
        "last_decision_date": None,
        "last_train_week": None,
        "last_train_date": None,
    }


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def iso_week_key(ts):
    iso = pd.Timestamp(ts).isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"


def should_retrain(state, decision_date, model_bundle):
    if model_bundle is None:
        return True
    return state.get("last_train_week") != iso_week_key(decision_date)


def main():
    state = load_state()

    print("Downloading latest FX prices...")
    features = prepare_feature_frame()
    valid_dates = features.index

    last_decision_date = state.get("last_decision_date")
    if last_decision_date is None:
        dates_to_process = valid_dates[-1:]
    else:
        last_ts = pd.Timestamp(last_decision_date)
        dates_to_process = valid_dates[valid_dates > last_ts]

    if len(dates_to_process) == 0:
        print(f"No new feature day since {last_decision_date}. Nothing to do.")
        return

    model_bundle = load_model(MODEL_PATH)
    prev_exposure = state.get("prev_exposure")
    prev_prob_signal = int(state.get("prev_prob_signal") or 0)
    pending_cost = float(state.get("pending_cost") or 0.0)
    equity = float(state.get("equity") or 1.0)

    for decision_date in dates_to_process:
        history = features.loc[features.index < decision_date]
        if len(history) < MIN_HISTORY_DAYS:
            print(
                f"{decision_date.date()} only {len(history)} training rows available "
                f"(< MIN_HISTORY_DAYS={MIN_HISTORY_DAYS}); skipping decision."
            )
            continue

        retrained = False
        if should_retrain(state, decision_date, model_bundle):
            model_bundle = train_model(history)
            model_bundle["trained_on"] = decision_date.date().isoformat()
            model_bundle["trained_through"] = history.index[-1].date().isoformat()
            model_bundle["trained_week"] = iso_week_key(decision_date)
            model_bundle["n_train_rows"] = int(len(history))
            save_model(model_bundle, MODEL_PATH)
            state["last_train_week"] = model_bundle["trained_week"]
            state["last_train_date"] = model_bundle["trained_on"]
            retrained = True
            print(
                f"{decision_date.date()} retrained model on {len(history)} rows "
                f"through {history.index[-1].date()}."
            )

        row = features.loc[decision_date]
        realized_return = None
        if prev_exposure is not None:
            realized_return = float(prev_exposure * row["spread_ret"] - pending_cost)
            equity *= 1.0 + realized_return
            print(
                f"{decision_date.date()} realized return={realized_return:.4%} "
                f"equity={equity:.4f}"
            )
        else:
            print(f"{decision_date.date()} bootstrap day, no prior exposure yet.")

        prob = predict_probability(row, model_bundle)
        decision = calculate_decision(
            row=row,
            prob=prob,
            prev_exposure=prev_exposure,
            prev_prob_signal=prev_prob_signal,
            prob_thresh=PROB_THRESH,
            prob_reset_thresh=PROB_RESET_THRESH,
            min_size=MIN_SIZE,
            hysteresis=EXPOSURE_HYSTERESIS,
        )
        cost = decision["turnover"] * (TC_BPS + SLIP_BPS)

        append_csv(
            TRADE_LOG_PATH,
            {
                "date": decision_date.date().isoformat(),
                "realized_return": realized_return,
                "equity": equity,
                "prob": prob,
                "prob_signal": decision["prob_signal"],
                "signal": decision["signal"],
                "target_exposure": decision["target_exposure"],
                "exposure": decision["exposure"],
                "turnover": decision["turnover"],
                "cost_next_day": cost,
                "spread": row["SPREAD"],
                "spread_z": row["spread_z"],
                "z_entry_dynamic": row["z_entry_dynamic"],
                "capital_weight": row["capital_weight"],
                "spread_ret": row["spread_ret"],
                "eurusd": row["EURUSD"],
                "usdinr": row["USDINR"],
                "eurinr": row["EURINR"],
                "implied_eurusd": row["IMPLIED_EURUSD"],
                "retrained": retrained,
                "trained_week": state.get("last_train_week"),
                "trained_through": model_bundle.get("trained_through"),
                "n_train_rows": model_bundle.get("n_train_rows"),
            },
        )

        prev_exposure = decision["exposure"]
        prev_prob_signal = decision["prob_signal"]
        pending_cost = cost

    state.update(
        {
            "prev_exposure": prev_exposure,
            "prev_prob_signal": prev_prob_signal,
            "pending_cost": pending_cost,
            "equity": equity,
            "last_decision_date": dates_to_process[-1].date().isoformat(),
        }
    )
    save_state(state)
    print("State saved. Last decision date:", state["last_decision_date"])


if __name__ == "__main__":
    main()
