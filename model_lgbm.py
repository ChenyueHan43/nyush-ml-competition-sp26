"""
LightGBM model for CSI500 stock selection.

Improvements over the XGBoost baseline:
- 28 features vs 14 (MACD, Bollinger, acceleration, skew/kurtosis, etc.)
- LightGBM (faster training, better generalisation on tabular data)
- Ensemble: train on 3 overlapping time windows, average predictions
- Walk-forward backtest with reported rank IC per window
- Portfolio construction: score-weighted with 10% cap and min-30 constraint

Usage
-----
  python model_lgbm.py                        # predict from latest data
  python model_lgbm.py --as-of 20260503       # submission 1
  python model_lgbm.py --as-of 20260510       # submission 2
  python model_lgbm.py --as-of 20260503 --out submissions/week1.csv
  python model_lgbm.py --backtest             # walk-forward IC report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features_enhanced import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, prediction_frame,
)

DATA_DIR = Path(__file__).parent / "data"
VAL_DAYS = 10
EMBARGO_DAYS = 5
MIN_STOCKS = 30
MAX_WEIGHT = 0.10
DEFAULT_TOP_K = 50


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)


def train_lgbm(train_df: pd.DataFrame, val_df: pd.DataFrame) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


# ---------------------------------------------------------------------------
# Ensemble: train on multiple historical windows, average scores
# ---------------------------------------------------------------------------

def train_ensemble(panel: pd.DataFrame, as_of_ts: pd.Timestamp) -> list:
    """Train 3 LightGBM models on rolling windows, return list of models."""
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])

    train_pool = training_frame(panel, max_date=train_cutoff)
    all_dates = np.sort(train_pool["date"].unique())

    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 40:
        raise RuntimeError("Not enough history. Try --start 20220101 when downloading.")

    # Window boundaries: use last 1y, last 2y, and all-history
    models = []
    window_configs = [
        (None, None, "full history"),
        (all_dates[max(0, len(all_dates) - 500)], None, "~2 years"),
        (all_dates[max(0, len(all_dates) - 250)], None, "~1 year"),
    ]

    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])

    for min_d, _, label in window_configs:
        sub = train_pool.copy()
        if min_d is not None:
            sub = sub[sub["date"] >= pd.Timestamp(min_d)]
        train_df = sub[sub["date"] <= train_end]
        val_df = sub[sub["date"] >= val_start]
        if len(train_df) < 500 or len(val_df) < 20:
            continue
        model = train_lgbm(train_df, val_df)
        val_pred = model.predict(val_df[FEATURE_COLUMNS])
        ic = rank_ic(
            val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy()
        )
        print(f"   [{label}] val rank IC: {ic:.4f}  (train rows: {len(train_df):,})")
        models.append(model)

    return models


def ensemble_predict(models: list, df: pd.DataFrame) -> np.ndarray:
    preds = np.stack([m.predict(df[FEATURE_COLUMNS]) for m in models], axis=1)
    return preds.mean(axis=1)


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K) -> pd.Series:
    """Top-K names, rank-weighted with iterative 10% cap redistribution."""
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")
    chosen = scores.sort_values(ascending=False).head(top_k).copy()

    # Rank-based initial weights
    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    for _ in range(100):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()

    w = w / w.sum()  # re-normalise for floating point safety
    assert abs(w.sum() - 1.0) < 1e-5
    assert (w <= MAX_WEIGHT + 1e-9).all()
    assert (w > 0).sum() >= MIN_STOCKS
    return w


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def walk_forward_backtest(panel: pd.DataFrame, n_windows: int = 8):
    """Rolling backtest: report rank IC over multiple prediction windows."""
    print("\n=== Walk-forward backtest ===")
    trading_dates = np.sort(panel["date"].unique())
    results = []

    step = 10  # predict every 10 trading days
    start_idx = 300  # need at least 300 days of history
    indices = range(start_idx, len(trading_dates) - FORWARD_HORIZON - VAL_DAYS, step)
    indices = list(indices)[-n_windows:]  # last n windows

    for idx in indices:
        as_of_ts = pd.Timestamp(trading_dates[idx])
        cutoff_idx = max(0, idx - FORWARD_HORIZON)
        train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])

        train_pool = training_frame(panel, max_date=train_cutoff)
        all_dates = np.sort(train_pool["date"].unique())
        if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 40:
            continue

        val_start = pd.Timestamp(all_dates[-VAL_DAYS])
        train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
        train_df = train_pool[train_pool["date"] <= train_end]
        val_df = train_pool[train_pool["date"] >= val_start]

        model = train_lgbm(train_df, val_df)
        pred_df = prediction_frame(panel, as_of=as_of_ts)
        if pred_df.empty:
            continue

        # Evaluate on next FORWARD_HORIZON days (out-of-sample)
        future_dates = trading_dates[idx + 1: idx + 1 + FORWARD_HORIZON]
        future = panel[panel["date"].isin(future_dates)].dropna(
            subset=FEATURE_COLUMNS + [TARGET_COLUMN]
        )
        if future.empty:
            continue
        future_pred = model.predict(future[FEATURE_COLUMNS])
        ic = rank_ic(
            future[TARGET_COLUMN].to_numpy(),
            future_pred,
            future["date"].to_numpy(),
        )
        results.append({"as_of": as_of_ts.date(), "oos_rank_ic": ic})
        print(f"  as_of {as_of_ts.date()}: OOS rank IC = {ic:.4f}")

    if results:
        ics = [r["oos_rank_ic"] for r in results]
        print(f"\nMean OOS rank IC: {np.mean(ics):.4f}  (std: {np.std(ics):.4f})")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--as-of", default=None, help="YYYYMMDD prediction date")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--backtest", action="store_true", help="run walk-forward backtest")
    args = p.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"dates {prices['date'].min().date()} to {prices['date'].max().date()}")

    print(">> Building enhanced features (28 features)...")
    panel = build_features(prices)

    if args.backtest:
        walk_forward_backtest(panel)
        return

    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    print(f">> Prediction date: {as_of_ts.date()}")

    print(">> Training ensemble (3 LightGBM models)...")
    models = train_ensemble(panel, as_of_ts)
    if not models:
        raise RuntimeError("No models trained. Check data.")

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=as_of_ts)
    if pred_df.empty:
        raise RuntimeError(f"No data for as_of={as_of_ts.date()}")
    print(f"   scoring {len(pred_df)} stocks")

    scores_arr = ensemble_predict(models, pred_df)
    scores = pd.Series(scores_arr, index=pred_df["stock_code"].values)

    weights = build_portfolio(scores, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)
    print(f">> Wrote {len(out)} stocks to {out_path}")
    print(f"   weights: min={out['weight'].min():.4f} "
          f"max={out['weight'].max():.4f} sum={out['weight'].sum():.6f}")


if __name__ == "__main__":
    main()
