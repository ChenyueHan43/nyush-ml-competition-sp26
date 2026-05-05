"""
Feature selection: remove low-importance features from the 56-feature set.

Strategy
--------
1. Train XGBoost (default params) on the fixed train split with all 56 features.
2. Extract per-feature 'gain' importance (average information gain per split —
   the most reliable XGBoost importance metric).
3. Print sorted importances so we can see the long tail.
4. Define two pruned feature sets:
     - "keep_positive" : drop features with gain == 0 (never used)
     - "keep_top50pct" : drop bottom 50% of features by gain
5. Re-train and compare portfolio backtest on the test set.

Usage
-----
  python feature_selection.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features_enhanced import FEATURE_COLUMNS, TARGET_COLUMN, build_features
from score_submission import score_window

DATA_DIR = Path(__file__).parent / "data"

# ── Fixed split (same as self_test.py) ──────────────────────────────────────
TRAIN_START = "2022-01-01"
TRAIN_END   = "2024-09-30"
VAL_START   = "2024-11-01"
VAL_END     = "2025-05-31"
TEST_START  = "2025-07-01"
TEST_END    = "2026-04-30"

# ── Portfolio params (current best) ─────────────────────────────────────────
FORWARD_HORIZON = 3    # 3-day target (best in previous experiments)
TOP_K           = 30
MAX_WEIGHT      = 0.10
MIN_STOCKS      = 30
TARGET_COL      = "target_3d"   # train on 3-day target


def train_xgb(train_df, val_df, feature_cols, target_col=TARGET_COL):
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method="hist", n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(
        train_df[feature_cols], train_df[target_col],
        eval_set=[(val_df[feature_cols], val_df[target_col])],
        verbose=False,
    )
    return model


def rank_ic_mean(df: pd.DataFrame, pred_col: str = "pred") -> float:
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < 20:
            continue
        rho, _ = spearmanr(g[TARGET_COLUMN], g[pred_col])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def build_score_weighted(scores: pd.Series, top_k: int) -> pd.Series:
    chosen = scores.sort_values(ascending=False).head(top_k).copy()
    chosen = chosen - chosen.min() + 1e-6
    if chosen.sum() == 0:
        chosen = pd.Series(1.0, index=chosen.index)
    w = chosen / chosen.sum()
    for _ in range(100):
        over = w > MAX_WEIGHT - 1e-9
        if not over.any():
            break
        excess = (w[over] - (MAX_WEIGHT - 1e-9)).sum()
        w[over] = MAX_WEIGHT - 1e-9
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    w = w / w.sum()
    w = w.clip(upper=MAX_WEIGHT)
    return w / w.sum()


def portfolio_backtest(panel, prices, index_df, model, feature_cols, label):
    trading_dates = np.sort(panel["date"].unique())
    test_mask = (
        (trading_dates >= np.datetime64(TEST_START)) &
        (trading_dates <= np.datetime64(TEST_END))
    )
    test_dates = trading_dates[test_mask]

    results = []
    i = 0
    while i + FORWARD_HORIZON < len(test_dates):
        as_of_ts = pd.Timestamp(test_dates[i])
        pred_df = panel[panel["date"] == as_of_ts].dropna(subset=feature_cols).copy()
        if len(pred_df) < MIN_STOCKS:
            i += FORWARD_HORIZON
            continue

        pred_df["score"] = model.predict(pred_df[feature_cols])
        scores = pred_df.set_index("stock_code")["score"]
        weights = build_score_weighted(scores, TOP_K)

        start = pd.Timestamp(test_dates[i + 1])
        end   = pd.Timestamp(test_dates[min(i + FORWARD_HORIZON, len(test_dates) - 1)])
        try:
            r = score_window(weights, prices, index_df, start, end)
        except RuntimeError:
            i += FORWARD_HORIZON
            continue

        results.append(r)
        i += FORWARD_HORIZON

    if not results:
        print(f"  [{label}] no results")
        return None

    df = pd.DataFrame(results)
    cum_excess  = (1 + df["portfolio_return"]).prod() - (1 + df["benchmark_return"]).prod()
    win_rate    = (df["excess_return"] > 0).mean() * 100
    mean_excess = df["excess_return"].mean() * 100
    print(f"  {label:<40} cum={cum_excess*100:>+7.2f}%  "
          f"win={win_rate:>4.1f}%  mean/win={mean_excess:>+6.3f}%  "
          f"n={len(df)}")
    return {"cum_excess": cum_excess, "win_rate": win_rate, "mean_excess": mean_excess}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(">> Loading data...")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    index_df["date"] = pd.to_datetime(index_df["date"])

    print(">> Building features (from cache if available)...")
    panel = build_features(prices, industry_neutral=False)

    # Make sure 3d target is available (fallback to 5d)
    if TARGET_COL not in panel.columns:
        print(f"  [warn] {TARGET_COL} not found, using {TARGET_COLUMN}")
        target_col_use = TARGET_COLUMN
    else:
        target_col_use = TARGET_COL

    # Drop rows with any feature or target NaN
    all_cols = FEATURE_COLUMNS + [target_col_use, TARGET_COLUMN]
    available_cols = [c for c in all_cols if c in panel.columns]
    panel = panel.dropna(subset=available_cols)

    train_df = panel[(panel["date"] >= TRAIN_START) & (panel["date"] <= TRAIN_END)].copy()
    val_df   = panel[(panel["date"] >= VAL_START)   & (panel["date"] <= VAL_END)].copy()
    test_df  = panel[(panel["date"] >= TEST_START)  & (panel["date"] <= TEST_END)].copy()

    print(f"   train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # ── Step 1: Train with ALL 56 features ───────────────────────────────────
    print(f"\n>> Training XGBoost with ALL {len(FEATURE_COLUMNS)} features...")
    model_full = train_xgb(train_df, val_df, FEATURE_COLUMNS, target_col_use)

    test_df = test_df.copy()
    test_df["pred"] = model_full.predict(test_df[FEATURE_COLUMNS])
    ic_full = rank_ic_mean(test_df)
    print(f"   Test Rank IC (all features): {ic_full:+.4f}")

    # ── Step 2: Feature importance analysis ──────────────────────────────────
    booster = model_full.get_booster()
    importance_gain = booster.get_score(importance_type="gain")
    importance_weight = booster.get_score(importance_type="weight")

    # Build sorted DataFrame
    imp_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
    })
    imp_df["gain"]   = imp_df["feature"].map(importance_gain).fillna(0)
    imp_df["weight"] = imp_df["feature"].map(importance_weight).fillna(0)
    imp_df = imp_df.sort_values("gain", ascending=False).reset_index(drop=True)

    print(f"\n{'═'*65}")
    print(f"  Feature Importance (gain) — sorted descending")
    print(f"{'═'*65}")
    print(f"  {'#':>3}  {'feature':<30} {'gain':>10} {'weight':>8} {'%gain':>7}")
    print(f"  {'-'*63}")
    total_gain = imp_df["gain"].sum()
    for i, row in imp_df.iterrows():
        pct = row["gain"] / total_gain * 100 if total_gain > 0 else 0
        marker = " ← zero" if row["gain"] == 0 else ""
        print(f"  {i+1:>3}. {row['feature']:<30} {row['gain']:>10.2f} "
              f"{int(row['weight']):>8} {pct:>6.2f}%{marker}")
    print(f"{'═'*65}")

    # Summary
    zero_gain = (imp_df["gain"] == 0).sum()
    zero_weight = (imp_df["weight"] == 0).sum()
    print(f"\n  Features with gain == 0 (never contributed): {zero_gain}")
    print(f"  Features with weight == 0 (never used):     {zero_weight}")

    # ── Step 3: Define pruned feature sets ───────────────────────────────────
    # Set A: remove zero-gain features
    features_nonzero = imp_df[imp_df["gain"] > 0]["feature"].tolist()
    # Set B: keep only top 50% by gain (remove bottom half)
    n_keep = max(MIN_STOCKS, len(FEATURE_COLUMNS) // 2)
    features_top50 = imp_df.head(n_keep)["feature"].tolist()
    # Set C: keep only features with gain > 5% of max (aggressive pruning)
    threshold_5pct = imp_df["gain"].max() * 0.05
    features_top5pct = imp_df[imp_df["gain"] >= threshold_5pct]["feature"].tolist()

    print(f"\n  Feature set sizes:")
    print(f"    Full           : {len(FEATURE_COLUMNS)}")
    print(f"    Non-zero gain  : {len(features_nonzero)}")
    print(f"    Top 50%        : {len(features_top50)}")
    print(f"    Gain ≥ 5% max  : {len(features_top5pct)}")

    print(f"\n  Dropped (zero gain):")
    zero_feats = imp_df[imp_df["gain"] == 0]["feature"].tolist()
    for f in zero_feats:
        print(f"    - {f}")

    print(f"\n  Dropped (bottom 50%):")
    dropped_50 = imp_df.tail(len(FEATURE_COLUMNS) - n_keep)["feature"].tolist()
    for f in dropped_50:
        print(f"    - {f}")

    # ── Step 4: Retrain and compare backtest ─────────────────────────────────
    print(f"\n>> Retraining with pruned feature sets...")

    configs = [
        ("Full (56 feat)",           FEATURE_COLUMNS),
        (f"Non-zero gain ({len(features_nonzero)} feat)", features_nonzero),
        (f"Top 50% ({len(features_top50)} feat)",         features_top50),
        (f"Gain≥5%max ({len(features_top5pct)} feat)",    features_top5pct),
    ]

    print(f"\n>> Portfolio backtest comparison (test set: {TEST_START} → {TEST_END})")
    print(f"   strategy: score-weighted top-{TOP_K}, 3-day target")
    print(f"{'─'*75}")

    for label, feat_cols in configs:
        # Re-train on this feature subset
        model_sub = train_xgb(train_df, val_df, feat_cols, target_col_use)

        # IC on test
        test_sub = test_df[["date", "stock_code", TARGET_COLUMN, target_col_use] + feat_cols].copy()
        test_sub["pred"] = model_sub.predict(test_sub[feat_cols])
        ic = rank_ic_mean(test_sub)

        # Portfolio backtest
        result = portfolio_backtest(panel, prices, index_df, model_sub, feat_cols, label)
        if result:
            print(f"    → Test Rank IC: {ic:+.4f}")

    # ── Step 5: Show optimal feature list for the best pruned set ─────────────
    print(f"\n>> Best pruned feature list (non-zero gain, {len(features_nonzero)} features):")
    print("PRUNED_FEATURE_COLUMNS = [")
    for f in features_nonzero:
        print(f'    "{f}",')
    print("]")


if __name__ == "__main__":
    main()
