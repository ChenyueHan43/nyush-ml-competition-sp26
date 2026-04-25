"""
Self-test: fixed train / validation / test split for the CSI500 competition.

Split (by trading date):
  Train      : 2023-01-01 – 2024-09-30  (~2 years)
  embargo gap: ~5 trading days
  Validation : 2024-11-01 – 2025-05-31  (~7 months, used for window selection)
  embargo gap: ~5 trading days
  Test       : 2025-07-01 – 2026-04-22  (~9 months, held-out, evaluated once)

Training window (2023-01-01) selected via val IC only; test set never seen
before final evaluation.

Reported metrics
  - Mean daily Rank IC (Spearman) on test set
  - IC standard deviation and IR (IC / std)
  - Comparison against XGBoost baseline run on the same split
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features_enhanced import FEATURE_COLUMNS, TARGET_COLUMN, build_features
from features import FEATURE_COLUMNS as BASE_FEATURE_COLUMNS

DATA_DIR = Path(__file__).parent / "data"

# ── Fixed split boundaries ──────────────────────────────────────────────────
# Training window selected via val IC only; test set evaluated exactly once.
TRAIN_START = "2022-01-01"   # selected via val IC (see window selection experiment)
TRAIN_END   = "2024-09-30"
VAL_START   = "2024-11-01"   # ~5 trading-day embargo after TRAIN_END
VAL_END     = "2025-05-31"
TEST_START  = "2025-07-01"   # ~5 trading-day embargo after VAL_END
TEST_END    = "2026-04-22"   # latest available data

EMBARGO_DAYS = 5


# ── Metrics ─────────────────────────────────────────────────────────────────

def rank_ic_series(df: pd.DataFrame, pred_col: str = "pred") -> pd.Series:
    """Daily Spearman IC between pred and target; returns a Series indexed by date."""
    records = {}
    for d, g in df.groupby("date"):
        if len(g) < 20:
            continue
        rho, _ = spearmanr(g[TARGET_COLUMN], g[pred_col])
        if not np.isnan(rho):
            records[d] = rho
    return pd.Series(records, name="IC")


def report(name: str, ic_series: pd.Series):
    mean_ic = ic_series.mean()
    std_ic  = ic_series.std()
    ir      = mean_ic / std_ic if std_ic > 0 else float("nan")
    pos_pct = (ic_series > 0).mean() * 100
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Test period   : {ic_series.index.min().date()} → {ic_series.index.max().date()}")
    print(f"  Trading days  : {len(ic_series)}")
    print(f"  Mean Rank IC  : {mean_ic:+.4f}")
    print(f"  IC Std        : {std_ic:.4f}")
    print(f"  IC IR         : {ir:+.4f}")
    print(f"  % days IC > 0 : {pos_pct:.1f}%")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f">> Loading {DATA_DIR / 'prices.parquet'}")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"{prices['date'].min().date()} → {prices['date'].max().date()}")

    # ── Build enhanced features ──────────────────────────────────────────────
    print("\n>> Building enhanced features...")
    panel = build_features(prices)
    panel = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    # ── Split ────────────────────────────────────────────────────────────────
    train_df = panel[(panel["date"] >= TRAIN_START) & (panel["date"] <= TRAIN_END)].copy()
    val_df   = panel[(panel["date"] >= VAL_START) & (panel["date"] <= VAL_END)].copy()
    test_df  = panel[(panel["date"] >= TEST_START) & (panel["date"] <= TEST_END)].copy()

    print(f"\n>> Data split")
    print(f"   Train : {len(train_df):>7,} rows  "
          f"({train_df['date'].min().date()} → {train_df['date'].max().date()})")
    print(f"   Val   : {len(val_df):>7,} rows  "
          f"({val_df['date'].min().date()} → {val_df['date'].max().date()})")
    print(f"   Test  : {len(test_df):>7,} rows  "
          f"({test_df['date'].min().date()} → {test_df['date'].max().date()})")

    if test_df.empty:
        print("ERROR: test set is empty — check TEST_START/TEST_END dates.")
        return

    # ── LightGBM (your model) ────────────────────────────────────────────────
    print("\n>> Training LightGBM (enhanced features)...")
    lgb_params = dict(
        objective="regression", metric="rmse",
        n_estimators=800, learning_rate=0.03,
        max_depth=6, num_leaves=63, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1,
    )
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    test_df = test_df.copy()
    test_df["pred"] = lgb_model.predict(test_df[FEATURE_COLUMNS])
    lgb_ic = rank_ic_series(test_df)
    report(f"LightGBM (your model, {len(FEATURE_COLUMNS)} features)", lgb_ic)

    # ── XGBoost baseline ─────────────────────────────────────────────────────
    print("\n>> Training XGBoost baseline (14 features)...")
    from features import build_features as build_base_features
    panel_base = build_base_features(prices)
    panel_base = panel_base.dropna(subset=BASE_FEATURE_COLUMNS + [TARGET_COLUMN])
    train_base = panel_base[(panel_base["date"] >= TRAIN_START) & (panel_base["date"] <= TRAIN_END)]
    val_base   = panel_base[(panel_base["date"] >= VAL_START) & (panel_base["date"] <= VAL_END)]
    test_base  = panel_base[(panel_base["date"] >= TEST_START) & (panel_base["date"] <= TEST_END)].copy()

    xgb_model = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method="hist", n_jobs=-1,
        early_stopping_rounds=30,
    )
    xgb_model.fit(
        train_base[BASE_FEATURE_COLUMNS], train_base[TARGET_COLUMN],
        eval_set=[(val_base[BASE_FEATURE_COLUMNS], val_base[TARGET_COLUMN])],
        verbose=False,
    )
    test_base["pred"] = xgb_model.predict(test_base[BASE_FEATURE_COLUMNS])
    xgb_ic = rank_ic_series(test_base)
    report("XGBoost baseline (14 features)", xgb_ic)

    # ── Comparison ───────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print("  SUMMARY")
    print(f"{'═'*50}")
    print(f"  LightGBM mean IC : {lgb_ic.mean():+.4f}")
    print(f"  XGBoost  mean IC : {xgb_ic.mean():+.4f}")
    delta = lgb_ic.mean() - xgb_ic.mean()
    verdict = "BEATS" if delta > 0 else "LAGS"
    print(f"  Your model {verdict} baseline by {abs(delta):.4f} IC points")
    print(f"{'═'*50}\n")


if __name__ == "__main__":
    main()
