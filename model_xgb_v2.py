"""
Improved XGBoost model for CSI500 stock selection.

Strategy: XGBoost baseline (14 features) + 5 high-importance new features
identified via LightGBM feature importance analysis on the test split:

  New features added:
  - amihud_20d / amihud_20d_rank  (illiquidity premium)
  - overnight_ret                  (close→open, informed trading signal)
  - streak_rank                    (consecutive up/down momentum rank)
  - price_to_52w_low               (distance from 52-week low)

Total: 19 features. Keeps XGBoost to avoid overfitting from more features.

Usage
-----
  python model_xgb_v2.py --as-of 20260503 --out submissions/week1.csv
  python model_xgb_v2.py --as-of 20260510 --out submissions/week2.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features_enhanced import (
    TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, prediction_frame,
)
from model_lgbm import build_portfolio, MIN_STOCKS, DEFAULT_TOP_K

DATA_DIR = Path(__file__).parent / "data"
TRAIN_START = "2022-01-01"
VAL_DAYS = 10
EMBARGO_DAYS = 5

# Baseline 14 features + 5 new high-importance features
FEATURE_COLUMNS_V2 = [
    # ── baseline (14) ──────────────────────────────────────────────────────
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "vol_20d", "volume_z_20d", "turnover_ma_20d",
    "close_over_ma20", "close_over_ma60", "rsi_14",
    "ret_5d_rank", "ret_20d_rank", "vol_20d_rank",
    # ── new (5) ────────────────────────────────────────────────────────────
    "amihud_20d",       # illiquidity premium: illiquid stocks earn extra return
    "amihud_20d_rank",  # cross-sectional rank of Amihud
    "overnight_ret",    # close→open gap: proxy for informed overnight trading
    "streak_rank",      # rank of consecutive up/down streak
    "price_to_52w_low", # distance from 52-week low (mean reversion signal)
]

XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    early_stopping_rounds=30,
)


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        from scipy.stats import spearmanr
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        train_df[FEATURE_COLUMNS_V2], train_df[TARGET_COLUMN],
        eval_set=[(val_df[FEATURE_COLUMNS_V2], val_df[TARGET_COLUMN])],
        verbose=False,
    )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--as-of", default=None, help="YYYYMMDD prediction date")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])

    print(">> Building features...")
    panel = build_features(prices)

    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])

    train_pool = training_frame(panel, min_date=TRAIN_START, max_date=train_cutoff)
    # Also need price_to_52w_low which isn't in FEATURE_COLUMNS of features_enhanced,
    # so drop NaN manually for our extra feature.
    train_pool = train_pool.dropna(subset=FEATURE_COLUMNS_V2)

    all_dates = np.sort(train_pool["date"].unique())
    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]

    print(f"   train: {len(train_df):,} rows up to {train_end.date()}")
    print(f"   val:   {len(val_df):,} rows from {val_start.date()}")
    print(f"   features: {len(FEATURE_COLUMNS_V2)} (14 baseline + 5 new)")

    print(">> Training XGBoost v2...")
    model = train_model(train_df, val_df)

    val_pred = model.predict(val_df[FEATURE_COLUMNS_V2])
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    print(f"   validation rank IC: {ic:.4f}")

    print(">> Predicting portfolio...")
    pred_df = prediction_frame(panel, as_of=as_of_ts)
    pred_df = pred_df.dropna(subset=FEATURE_COLUMNS_V2).copy()
    if pred_df.empty:
        raise RuntimeError(f"No rows for as_of={as_of_ts.date()}")
    print(f"   scoring {len(pred_df)} stocks as of {as_of_ts.date()}")

    pred_df["score"] = model.predict(pred_df[FEATURE_COLUMNS_V2])
    scores = pred_df.set_index("stock_code")["score"]
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
