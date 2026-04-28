"""
Grid search over portfolio construction parameters.

Tests combinations of:
  - top_k: how many stocks to select (30, 40, 50, 60, 80)
  - weighting: rank-weighted (baseline), equal-weight, score-weighted

Tests two models:
  - XGBoost baseline (14 features, no industry neutralization)
  - LightGBM (52 features, industry neutralization)
Reports cumulative excess return and win rate for each combination.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from features import FEATURE_COLUMNS as BASE_FEATURE_COLUMNS, TARGET_COLUMN, build_features as build_base_features, training_frame, prediction_frame
from features_enhanced import FEATURE_COLUMNS, build_features
from score_submission import score_window

DATA_DIR = __import__("pathlib").Path(__file__).parent / "data"

TRAIN_START = "2022-01-01"
TRAIN_END   = "2024-09-30"
VAL_START   = "2024-11-01"
VAL_END     = "2025-05-31"
TEST_START  = "2025-07-01"
TEST_END    = "2026-04-22"

FORWARD_HORIZON = 5
VAL_DAYS = 10
EMBARGO_DAYS = 5
MIN_STOCKS = 30
MAX_WEIGHT = 0.10


# ── Portfolio construction ────────────────────────────────────────────────────

def build_rank_weighted(scores: pd.Series, top_k: int) -> pd.Series:
    """Baseline: rank-weighted with 10% cap."""
    chosen = scores.sort_values(ascending=False).head(top_k)
    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index, dtype=float)
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
    return w / w.sum()


def build_equal_weight(scores: pd.Series, top_k: int) -> pd.Series:
    """Equal weight across top-k stocks, capped at 10%."""
    chosen = scores.sort_values(ascending=False).head(top_k)
    w = pd.Series(1.0 / top_k, index=chosen.index)
    # cap at 10% (only matters if top_k < 10)
    w = w.clip(upper=MAX_WEIGHT)
    return w / w.sum()


def build_score_weighted(scores: pd.Series, top_k: int) -> pd.Series:
    """Weight proportional to predicted score (shifted to be positive), capped at 10%."""
    chosen = scores.sort_values(ascending=False).head(top_k).copy()
    # shift so minimum score = 0, then normalize
    chosen = chosen - chosen.min()
    if chosen.sum() == 0:
        chosen = pd.Series(1.0, index=chosen.index)
    w = chosen / chosen.sum()
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
    return w / w.sum()


WEIGHT_METHODS = {
    "rank-weighted": build_rank_weighted,
    "equal-weight":  build_equal_weight,
    "score-weighted": build_score_weighted,
}

TOP_K_VALUES = [30, 40, 50, 60, 80]


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(panel, prices, index_df, model, top_k, weight_fn, step_days=5, feature_cols=None):
    if feature_cols is None:
        feature_cols = BASE_FEATURE_COLUMNS
    trading_dates = np.sort(panel["date"].unique())
    test_mask = (trading_dates >= np.datetime64(TEST_START)) & \
                (trading_dates <= np.datetime64(TEST_END))
    test_dates = trading_dates[test_mask]

    results = []
    i = 0
    while i + FORWARD_HORIZON < len(test_dates):
        as_of_ts = pd.Timestamp(test_dates[i])
        pred_df = panel[panel["date"] == as_of_ts].dropna(subset=feature_cols).copy()
        if len(pred_df) < MIN_STOCKS:
            i += step_days
            continue

        pred_df["score"] = model.predict(pred_df[feature_cols])
        scores = pred_df.set_index("stock_code")["score"]

        try:
            weights = weight_fn(scores, top_k)
        except Exception:
            i += step_days
            continue

        start = pd.Timestamp(test_dates[i + 1])
        end   = pd.Timestamp(test_dates[min(i + FORWARD_HORIZON, len(test_dates) - 1)])

        try:
            r = score_window(weights, prices, index_df, start, end)
        except RuntimeError:
            i += step_days
            continue

        results.append(r)
        i += step_days

    if not results:
        return None
    df = pd.DataFrame(results)
    return {
        "cum_excess":   (1 + df["portfolio_return"]).prod() - (1 + df["benchmark_return"]).prod(),
        "mean_excess":  df["excess_return"].mean() * 100,
        "win_rate":     (df["excess_return"] > 0).mean() * 100,
        "n_windows":    len(df),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def print_grid(rows: list, title: str):
    df = pd.DataFrame(rows).sort_values("cum_excess", ascending=False)
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'═'*64}")
    print(f"  {'weighting':<16} {'top_k':>5} {'cum_excess':>10} {'win_rate':>9} {'mean/win':>9}")
    print(f"  {'-'*60}")
    for i, (_, row) in enumerate(df.iterrows()):
        marker = " ← best" if i == 0 else ""
        print(f"  {row['weighting']:<16} {int(row['top_k']):>5} "
              f"{row['cum_excess']*100:>+9.2f}% "
              f"{row['win_rate']:>8.1f}% "
              f"{row['mean_excess']:>+8.3f}%{marker}")
    print(f"{'═'*64}")


def main():
    print(">> Loading data...")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    index_df["date"] = pd.to_datetime(index_df["date"])

    # ── Model 1: XGBoost baseline (no neutralization) ─────────────────────────
    print("\n>> [Model 1] Building baseline features (no industry neutral)...")
    panel_base = build_base_features(prices)
    train_base = panel_base[(panel_base["date"] >= TRAIN_START) & (panel_base["date"] <= TRAIN_END)].dropna(subset=BASE_FEATURE_COLUMNS + [TARGET_COLUMN])
    val_base   = panel_base[(panel_base["date"] >= VAL_START)   & (panel_base["date"] <= VAL_END)].dropna(subset=BASE_FEATURE_COLUMNS + [TARGET_COLUMN])

    print(">> Training XGBoost baseline...")
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

    rows_xgb = []
    print(f"\n  Running {len(TOP_K_VALUES)} × {len(WEIGHT_METHODS)} combinations (XGBoost baseline)...")
    for weight_name, weight_fn in WEIGHT_METHODS.items():
        for top_k in TOP_K_VALUES:
            r = run_backtest(panel_base, prices, index_df, xgb_model, top_k, weight_fn, feature_cols=BASE_FEATURE_COLUMNS)
            if r is None:
                continue
            rows_xgb.append({"weighting": weight_name, "top_k": top_k, **r})
            print(f"  {weight_name:<16} top-{top_k:<3} | "
                  f"excess {r['cum_excess']*100:>+6.2f}%  "
                  f"win {r['win_rate']:>4.1f}%  "
                  f"mean/window {r['mean_excess']:>+.3f}%")

    # ── Model 2: LightGBM + industry neutralization ───────────────────────────
    print("\n>> [Model 2] Building enhanced features (industry neutral)...")
    panel_lgb = build_features(prices, industry_neutral=True)
    train_lgb = panel_lgb[(panel_lgb["date"] >= TRAIN_START) & (panel_lgb["date"] <= TRAIN_END)].dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    val_lgb   = panel_lgb[(panel_lgb["date"] >= VAL_START)   & (panel_lgb["date"] <= VAL_END)].dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    print(">> Training LightGBM (industry neutral)...")
    lgb_model = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        n_estimators=800, learning_rate=0.03,
        max_depth=6, num_leaves=63, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1,
    )
    lgb_model.fit(
        train_lgb[FEATURE_COLUMNS], train_lgb[TARGET_COLUMN],
        eval_set=[(val_lgb[FEATURE_COLUMNS], val_lgb[TARGET_COLUMN])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    rows_lgb = []
    print(f"\n  Running {len(TOP_K_VALUES)} × {len(WEIGHT_METHODS)} combinations (LightGBM neutral)...")
    for weight_name, weight_fn in WEIGHT_METHODS.items():
        for top_k in TOP_K_VALUES:
            r = run_backtest(panel_lgb, prices, index_df, lgb_model, top_k, weight_fn, feature_cols=FEATURE_COLUMNS)
            if r is None:
                continue
            rows_lgb.append({"weighting": weight_name, "top_k": top_k, **r})
            print(f"  {weight_name:<16} top-{top_k:<3} | "
                  f"excess {r['cum_excess']*100:>+6.2f}%  "
                  f"win {r['win_rate']:>4.1f}%  "
                  f"mean/window {r['mean_excess']:>+.3f}%")

    # ── Model 3: XGBoost baseline with 3-day target ───────────────────────────
    print("\n>> [Model 3] XGBoost baseline trained on 3-day target...")
    train_3d = panel_base[(panel_base["date"] >= TRAIN_START) & (panel_base["date"] <= TRAIN_END)].dropna(subset=BASE_FEATURE_COLUMNS + ["target_3d"])
    val_3d   = panel_base[(panel_base["date"] >= VAL_START)   & (panel_base["date"] <= VAL_END)].dropna(subset=BASE_FEATURE_COLUMNS + ["target_3d"])

    xgb_3d = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method="hist", n_jobs=-1,
        early_stopping_rounds=30,
    )
    xgb_3d.fit(
        train_3d[BASE_FEATURE_COLUMNS], train_3d["target_3d"],
        eval_set=[(val_3d[BASE_FEATURE_COLUMNS], val_3d["target_3d"])],
        verbose=False,
    )

    rows_3d = []
    print(f"\n  Running {len(TOP_K_VALUES)} × {len(WEIGHT_METHODS)} combinations (XGBoost 3d target)...")
    for weight_name, weight_fn in WEIGHT_METHODS.items():
        for top_k in TOP_K_VALUES:
            r = run_backtest(panel_base, prices, index_df, xgb_3d, top_k, weight_fn, feature_cols=BASE_FEATURE_COLUMNS)
            if r is None:
                continue
            rows_3d.append({"weighting": weight_name, "top_k": top_k, **r})
            print(f"  {weight_name:<16} top-{top_k:<3} | "
                  f"excess {r['cum_excess']*100:>+6.2f}%  "
                  f"win {r['win_rate']:>4.1f}%  "
                  f"mean/window {r['mean_excess']:>+.3f}%")

    print_grid(rows_xgb, "XGBoost baseline — 5d target (original)")
    print_grid(rows_lgb,  "LightGBM — industry neutral, 5d target")
    print_grid(rows_3d,   "XGBoost baseline — 3d target")


if __name__ == "__main__":
    main()
