"""
Self-test: fixed train / validation / test split for the CSI500 competition.

Split (by trading date):
  Train      : 2022-01-01 – 2024-09-30  (~3 years)
  embargo gap: ~5 trading days
  Validation : 2024-11-01 – 2025-05-31  (~7 months, used for window selection)
  embargo gap: ~5 trading days
  Test       : 2025-07-01 – 2026-04-22  (~9 months, held-out, evaluated once)

Training window (2022-01-01) selected via val IC only; test set never seen
before final evaluation.

Reported metrics
  - Mean daily Rank IC (Spearman) on test set
  - IC standard deviation and IR (IC / std)
  - Comparison against XGBoost baseline run on the same split
  - Walk-forward portfolio backtest: actual excess return vs CSI500
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
from model_xgb_v2 import FEATURE_COLUMNS_V2
from model_lgbm import build_portfolio, MIN_STOCKS, MAX_WEIGHT, FORWARD_HORIZON
from score_submission import score_window

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


# ── Portfolio backtest ───────────────────────────────────────────────────────

def _recency_weights(dates: pd.Series, half_life: int) -> np.ndarray:
    """Exponential time-decay sample weights. More recent rows get higher weight.
    Half-life is in trading days. Weights are normalised so their mean = 1.
    """
    idx = np.arange(len(dates))          # positional index (0 = oldest)
    w   = np.exp(np.log(2) / half_life * (idx - idx.max()))
    return w / w.mean()


def make_lgb_train_fn(lgb_params, val_days=10, embargo_days=5,
                      lookback_days=None, target_col=None,
                      recency_halflife=None):
    """Return a train_fn(panel, as_of_ts, feature_cols) → model for walk-forward retraining.

    lookback_days    : rolling training window in trading days (None = full history).
    target_col       : which column to use as training label (default TARGET_COLUMN).
    recency_halflife : if set, apply exponential time-decay sample weights with this
                       half-life in trading days (e.g. 126 ≈ 6 months).
    """
    _target = target_col or TARGET_COLUMN

    def train_fn(panel, as_of_ts, feature_cols):
        trading_dates = np.sort(panel["date"].unique())
        as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
        cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
        train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])

        train_pool = panel[panel["date"] <= train_cutoff].dropna(
            subset=feature_cols + [_target]
        )
        if lookback_days is not None:
            all_pool_dates = np.sort(train_pool["date"].unique())
            if len(all_pool_dates) > lookback_days:
                window_start = pd.Timestamp(all_pool_dates[-lookback_days])
                train_pool = train_pool[train_pool["date"] >= window_start]

        all_dates = np.sort(train_pool["date"].unique())
        if len(all_dates) < val_days + embargo_days + 40:
            return None

        val_start = pd.Timestamp(all_dates[-val_days])
        train_end = pd.Timestamp(all_dates[-(val_days + embargo_days + 1)])
        train_df = train_pool[train_pool["date"] <= train_end].reset_index(drop=True)
        val_df   = train_pool[train_pool["date"] >= val_start]

        sw = _recency_weights(train_df["date"], recency_halflife) \
             if recency_halflife else None

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            train_df[feature_cols], train_df[_target],
            sample_weight=sw,
            eval_set=[(val_df[feature_cols], val_df[_target])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        return model
    return train_fn


def select_lookback_on_val(panel, val_df, feature_cols, lgb_params,
                            lookback_options=(250, 500, None),
                            target_col=None):
    """Select training lookback window using validation IC only. Test data never touched.

    For each candidate window, trains on data up to TRAIN_END (with that window),
    evaluates Rank IC on the held-out val set, and returns the best lookback.
    target_col: which label column to train on (default TARGET_COLUMN).
    """
    _target = target_col or TARGET_COLUMN
    print("\n>> Selecting lookback window on val set (no test data used)...")
    trading_dates = np.sort(panel["date"].unique())
    train_end_ts  = pd.Timestamp(TRAIN_END)

    best_ic      = -np.inf
    best_lookback = None

    for lookback in lookback_options:
        train_pool = panel[panel["date"] <= train_end_ts].dropna(
            subset=feature_cols + [_target]
        )
        if lookback is not None:
            pool_dates = np.sort(train_pool["date"].unique())
            if len(pool_dates) > lookback:
                window_start = pd.Timestamp(pool_dates[-lookback])
                train_pool = train_pool[train_pool["date"] >= window_start]

        pool_dates = np.sort(train_pool["date"].unique())
        if len(pool_dates) < 20:
            continue

        es_start  = pd.Timestamp(pool_dates[-10])
        train_fit = train_pool[train_pool["date"] < es_start]
        es_val    = train_pool[train_pool["date"] >= es_start]

        if len(train_fit) < 500 or len(es_val) < 20:
            continue

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            train_fit[feature_cols], train_fit[_target],
            eval_set=[(es_val[feature_cols], es_val[_target])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        val_preds = model.predict(val_df[feature_cols])
        ic_series = rank_ic_series(val_df.assign(pred=val_preds))
        mean_ic   = ic_series.mean()

        label = f"{lookback}d" if lookback is not None else "full"
        print(f"   lookback={label:>6}: val Rank IC = {mean_ic:+.4f}")

        if mean_ic > best_ic:
            best_ic      = mean_ic
            best_lookback = lookback

    chosen = f"{best_lookback}d" if best_lookback is not None else "full"
    print(f"   → Selected: {chosen} (val IC = {best_ic:+.4f})")
    return best_lookback


def portfolio_backtest(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    model,
    feature_cols: list,
    step_days: int = 5,
    label: str = "model",
    train_fn=None,
    retrain_freq: int = 20,
):
    """
    Walk-forward backtest over the test period.

    Every `step_days` trading days, generate portfolio weights using `model`,
    then score against actual prices over the next FORWARD_HORIZON trading days.
    If `train_fn` is provided, retrains the model every `retrain_freq` trading days
    using all data available up to that point.
    """
    trading_dates = np.sort(panel["date"].unique())
    test_mask = (trading_dates >= np.datetime64(TEST_START)) & \
                (trading_dates <= np.datetime64(TEST_END))
    test_dates = trading_dates[test_mask]

    if len(test_dates) < FORWARD_HORIZON + 1:
        print("  [portfolio backtest] not enough test dates, skipping.")
        return

    current_model = model
    steps_since_retrain = retrain_freq  # force retrain on first step if train_fn given

    results = []
    i = 0
    while i + FORWARD_HORIZON < len(test_dates):
        as_of_ts = pd.Timestamp(test_dates[i])

        # Retrain if scheduled
        if train_fn is not None and steps_since_retrain >= retrain_freq:
            new_model = train_fn(panel, as_of_ts, feature_cols)
            if new_model is not None:
                current_model = new_model
                print(f"  [retrain] {as_of_ts.date()}")
            steps_since_retrain = 0

        pred_df = panel[panel["date"] == as_of_ts].dropna(subset=feature_cols).copy()
        if len(pred_df) < MIN_STOCKS:
            i += step_days
            steps_since_retrain += step_days
            continue

        pred_df["score"] = current_model.predict(pred_df[feature_cols])
        scores = pred_df.set_index("stock_code")["score"]
        weights = build_portfolio(scores)

        # Evaluation window: [t+1, t+FORWARD_HORIZON] trading days
        start = pd.Timestamp(test_dates[i + 1])
        end   = pd.Timestamp(test_dates[min(i + FORWARD_HORIZON, len(test_dates) - 1)])

        try:
            r = score_window(weights, prices, index_df, start, end)
        except RuntimeError as e:
            print(f"  [warn] {as_of_ts.date()}: {e}")
            i += step_days
            continue

        r["as_of"] = as_of_ts.date()
        results.append(r)
        i += step_days
        steps_since_retrain += step_days

    if not results:
        print("  [portfolio backtest] no results produced.")
        return

    df = pd.DataFrame(results).set_index("as_of")
    # Cumulative compounded returns
    cum_port  = (1 + df["portfolio_return"]).prod() - 1
    cum_bench = (1 + df["benchmark_return"]).prod() - 1
    cum_excess = cum_port - cum_bench
    win_rate  = (df["excess_return"] > 0).mean() * 100

    print(f"\n{'─'*50}")
    print(f"  Portfolio backtest — {label}")
    print(f"{'─'*50}")
    print(f"  Test period  : {df.index.min()} → {df.index.max()}")
    print(f"  # windows    : {len(df)}")
    print(f"\n  Per-window results:")
    print(f"  {'date':<12} {'port':>8} {'bench':>8} {'excess':>8}")
    for date, row in df.iterrows():
        flag = "+" if row["excess_return"] > 0 else "-"
        print(f"  {str(date):<12} "
              f"{row['portfolio_return']*100:>+7.2f}%  "
              f"{row['benchmark_return']*100:>+7.2f}%  "
              f"{row['excess_return']*100:>+7.2f}% {flag}")
    print(f"\n  Cumulative portfolio : {cum_port*100:>+.2f}%")
    print(f"  Cumulative benchmark : {cum_bench*100:>+.2f}%")
    print(f"  Cumulative excess    : {cum_excess*100:>+.2f}%")
    print(f"  Window win rate      : {win_rate:.1f}%")
    print(f"  Mean excess / window : {df['excess_return'].mean()*100:>+.3f}%")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f">> Loading {DATA_DIR / 'prices.parquet'}")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"{prices['date'].min().date()} → {prices['date'].max().date()}")

    # ── Build enhanced features ──────────────────────────────────────────────
    print("\n>> Building enhanced features...")
    panel = build_features(prices, industry_neutral=True)
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

    # ── XGBoost v2 (19 features) ─────────────────────────────────────────────
    print("\n>> Training XGBoost v2 (19 features = baseline + 5 new)...")
    panel_v2 = panel.dropna(subset=FEATURE_COLUMNS_V2 + [TARGET_COLUMN])
    train_v2 = panel_v2[(panel_v2["date"] >= TRAIN_START) & (panel_v2["date"] <= TRAIN_END)]
    val_v2   = panel_v2[(panel_v2["date"] >= VAL_START)   & (panel_v2["date"] <= VAL_END)]
    test_v2  = panel_v2[(panel_v2["date"] >= TEST_START)  & (panel_v2["date"] <= TEST_END)].copy()

    xgb_v2 = xgb.XGBRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method="hist", n_jobs=-1,
        early_stopping_rounds=30,
    )
    xgb_v2.fit(
        train_v2[FEATURE_COLUMNS_V2], train_v2[TARGET_COLUMN],
        eval_set=[(val_v2[FEATURE_COLUMNS_V2], val_v2[TARGET_COLUMN])],
        verbose=False,
    )
    test_v2["pred"] = xgb_v2.predict(test_v2[FEATURE_COLUMNS_V2])
    v2_ic = rank_ic_series(test_v2)
    report(f"XGBoost v2 (19 features)", v2_ic)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print("  SUMMARY — Rank IC")
    print(f"{'═'*50}")
    print(f"  XGBoost baseline (14) : {xgb_ic.mean():+.4f}")
    print(f"  XGBoost v2      (19)  : {v2_ic.mean():+.4f}")
    print(f"  LightGBM        (52)  : {lgb_ic.mean():+.4f}")
    print(f"{'═'*50}\n")

    # ── Portfolio backtest ────────────────────────────────────────────────────
    print("\n>> Loading price / index data for portfolio backtest...")
    prices_raw = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices_raw["date"] = pd.to_datetime(prices_raw["date"])
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    index_df["date"] = pd.to_datetime(index_df["date"])

    lgb_params = dict(
        objective="regression", metric="rmse",
        n_estimators=800, learning_rate=0.03,
        max_depth=6, num_leaves=63, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1,
    )

    # Select lookback window on val set only — test data never used for this choice
    best_lookback = select_lookback_on_val(panel, val_df, FEATURE_COLUMNS, lgb_params)
    lgb_train_fn  = make_lgb_train_fn(lgb_params, lookback_days=best_lookback)

    portfolio_backtest(
        panel_base, prices_raw, index_df, xgb_model, BASE_FEATURE_COLUMNS,
        step_days=5, label="XGBoost baseline (14 features)",
    )
    portfolio_backtest(
        panel_v2, prices_raw, index_df, xgb_v2, FEATURE_COLUMNS_V2,
        step_days=5, label="XGBoost v2 (19 features)",
    )
    portfolio_backtest(
        panel, prices_raw, index_df, lgb_model, FEATURE_COLUMNS,
        step_days=5, label="LightGBM walk-forward (52 features, val-selected lookback)",
        train_fn=lgb_train_fn, retrain_freq=20,
    )


if __name__ == "__main__":
    main()
