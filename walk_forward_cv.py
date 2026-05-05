"""
Walk-Forward Cross-Validation for CSI500 stock selection.

Design
------
Expanding training window, fixed-length validation window (3 months each).
5-day embargo between train end and val start to prevent leakage.

Folds (approximate):
  fold 1: train 2022-01 → 2022-12 | val 2023-02 → 2023-04
  fold 2: train 2022-01 → 2023-03 | val 2023-05 → 2023-07
  fold 3: train 2022-01 → 2023-06 | val 2023-08 → 2023-10
  fold 4: train 2022-01 → 2023-09 | val 2023-11 → 2024-01
  fold 5: train 2022-01 → 2023-12 | val 2024-02 → 2024-04
  fold 6: train 2022-01 → 2024-03 | val 2024-05 → 2024-07
  fold 7: train 2022-01 → 2024-06 | val 2024-08 → 2024-10

Compares 3 feature sets:
  - Baseline  : 14 features, 5d target (original)
  - Full       : 56 features, 3d target
  - Pruned-30  : top-30 by gain, 3d target

Usage
-----
  python walk_forward_cv.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features import FEATURE_COLUMNS as BASE_FEATURES, TARGET_COLUMN, build_features as build_base
from features_enhanced import FEATURE_COLUMNS as ENH_FEATURES, build_features

DATA_DIR = Path(__file__).parent / "data"

TRAIN_START  = "2022-01-01"
CV_END       = "2024-10-31"   # keep 2025+ as held-out test
EMBARGO_DAYS = 5              # trading days gap between train end and val start
VAL_MONTHS   = 3              # validation window length in months
STEP_MONTHS  = 3              # how far to slide the val window each fold

TOP30 = [
    "price_pressure_rank", "volume_z_20d", "streak_rank", "price_pressure",
    "turnover_z_60d", "streak", "reversal_5d_rank", "macd", "close_over_ma60",
    "ret_5d_rank", "bb_position", "close_over_ma5", "ret_5d", "reversal_1d_rank",
    "overnight_ret_rank", "ret_1d", "price_to_52w_high", "overnight_ret",
    "price_to_52w_high_rank", "close_over_ma20_rank", "ret_3d", "ret_1d_rank",
    "rsi_14_rank", "ret_accel_5d", "intraday_ret", "max_ret_20d",
    "turnover_z_60d_rank", "volume_ratio_5_20", "ma5_over_ma20", "vol_20d",
]


def train_xgb(train_df, val_df, feats, target):
    m = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method="hist", n_jobs=-1, early_stopping_rounds=30,
    )
    m.fit(train_df[feats], train_df[target],
          eval_set=[(val_df[feats], val_df[target])], verbose=False)
    return m


def rank_ic(df: pd.DataFrame, pred_col: str = "pred", target_col: str = TARGET_COLUMN) -> float:
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < 20:
            continue
        rho, _ = spearmanr(g[target_col], g[pred_col])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def make_folds(trading_dates: np.ndarray) -> list[dict]:
    """Generate walk-forward fold boundaries."""
    dates = pd.DatetimeIndex(trading_dates)
    folds = []

    # First val window starts ~3 months after TRAIN_START
    val_start = pd.Timestamp("2023-02-01")

    while True:
        val_end = val_start + pd.DateOffset(months=VAL_MONTHS) - pd.Timedelta(days=1)
        if val_end > pd.Timestamp(CV_END):
            break

        # embargo: skip EMBARGO_DAYS before val_start
        before_val = dates[dates < val_start]
        if len(before_val) < EMBARGO_DAYS + 50:
            val_start += pd.DateOffset(months=STEP_MONTHS)
            continue
        train_end = before_val[-EMBARGO_DAYS - 1]

        # Need enough training data
        train_dates = dates[(dates >= pd.Timestamp(TRAIN_START)) & (dates <= train_end)]
        val_dates   = dates[(dates >= val_start) & (dates <= val_end)]
        if len(train_dates) < 200 or len(val_dates) < 20:
            val_start += pd.DateOffset(months=STEP_MONTHS)
            continue

        folds.append({
            "train_end": train_end,
            "val_start": val_start,
            "val_end":   val_end,
            "n_train":   len(train_dates),
            "n_val":     len(val_dates),
        })
        val_start += pd.DateOffset(months=STEP_MONTHS)

    return folds


def run_cv(panel, feats, target_col, folds, label):
    """Run walk-forward CV for one feature/target config. Returns per-fold IC."""
    fold_ics = []
    for k, fold in enumerate(folds):
        train_df = panel[
            (panel["date"] >= TRAIN_START) &
            (panel["date"] <= fold["train_end"])
        ].dropna(subset=feats + [target_col])

        val_df = panel[
            (panel["date"] >= fold["val_start"]) &
            (panel["date"] <= fold["val_end"])
        ].dropna(subset=feats + [target_col])

        if len(train_df) < 500 or len(val_df) < 100:
            continue

        # Use last 10 trading days of train as inner val for early stopping
        inner_dates = np.sort(train_df["date"].unique())
        if len(inner_dates) < 20:
            continue
        inner_val_start = pd.Timestamp(inner_dates[-10])
        inner_train = train_df[train_df["date"] < inner_val_start]
        inner_val   = train_df[train_df["date"] >= inner_val_start]
        if len(inner_train) < 100 or len(inner_val) < 20:
            continue

        model = train_xgb(inner_train, inner_val, feats, target_col)

        val_df = val_df.copy()
        val_df["pred"] = model.predict(val_df[feats])
        ic = rank_ic(val_df, pred_col="pred", target_col=target_col)
        fold_ics.append(ic)

        print(f"  [{label}] fold {k+1}: "
              f"train→{fold['train_end'].date()}  "
              f"val {fold['val_start'].date()}→{fold['val_end'].date()}  "
              f"IC={ic:+.4f}")

    return fold_ics


def main():
    print(">> Loading data...")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    print(">> Building features...")
    panel_enh  = build_features(prices, industry_neutral=False)
    panel_base = build_base(prices)

    # Merge target_3d into base panel if needed
    trading_dates = np.sort(panel_enh["date"].unique())

    # Generate folds
    folds = make_folds(trading_dates)
    print(f"\n>> Generated {len(folds)} walk-forward folds:")
    for k, f in enumerate(folds):
        print(f"   fold {k+1}: train 2022-01-01→{f['train_end'].date()}  "
              f"val {f['val_start'].date()}→{f['val_end'].date()}  "
              f"(train={f['n_train']}d, val={f['n_val']}d)")

    configs = [
        ("Baseline  (14 feat, 5d)", panel_base, BASE_FEATURES, TARGET_COLUMN),
        ("Full      (56 feat, 3d)", panel_enh,  ENH_FEATURES,  "target_3d"),
        ("Pruned-30 (30 feat, 3d)", panel_enh,  TOP30,         "target_3d"),
    ]

    results = {}
    for label, panel, feats, target in configs:
        print(f"\n{'─'*60}")
        print(f">> {label}")
        print(f"{'─'*60}")
        ics = run_cv(panel, feats, target, folds, label)
        results[label] = ics

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Walk-Forward CV Summary — Mean Rank IC across folds")
    print(f"{'═'*60}")
    print(f"  {'Config':<30} {'Mean IC':>8} {'Std IC':>8} {'IR':>8} {'Folds':>6}")
    print(f"  {'─'*58}")
    best_label, best_ic = None, -np.inf
    for label, ics in results.items():
        if not ics:
            continue
        arr = np.array(ics)
        mean_ic = arr.mean()
        std_ic  = arr.std()
        ir      = mean_ic / std_ic if std_ic > 0 else float("nan")
        marker  = ""
        if mean_ic > best_ic:
            best_ic = mean_ic; best_label = label; marker = " ← best"
        print(f"  {label:<30} {mean_ic:>+8.4f} {std_ic:>8.4f} {ir:>+8.4f} "
              f"{len(ics):>6}{marker}")
    print(f"{'═'*60}")
    print(f"\n  Best config: {best_label}")
    print(f"  Recommendation: use this feature set for final submission.")


if __name__ == "__main__":
    main()
