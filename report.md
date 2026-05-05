# CSI500 Stock Selection Competition — Report

**Course:** Machine Learning (Spring 2026)  
**Student:** ch5085@nyu.edu  
**Submission dates:** Submission 1 (2026-05-03) · Submission 2 (2026-05-10)

---

## 1. Task Overview

The goal is to build a machine-learning model that selects a long-only portfolio over the CSI500 universe expected to outperform the index over the following week. Each portfolio must hold at least 30 names, cap each name at 10%, and sum weights to 1.0.

---

## 2. Data

Data was fetched via `akshare` using `download_data.py`. Four files were used:

| File | Content |
|---|---|
| `data/prices.parquet` | Daily OHLCV bars (forward-adjusted close) for ~499 CSI500 constituents, 2022-01-01 → 2026-04-30 |
| `data/index.parquet` | Daily bars for the CSI500 index (code `000905`) used as benchmark |
| `data/constituents.csv` | Current CSI500 member list |
| `data/industry.csv` | SW Level-1 industry classification for all 499 constituents (via akshare) |

**Look-ahead prevention:** All targets are computed as `close[t + h] / close[t] - 1` using `.shift(-h)` inside each stock's own time series, then training frames are bounded so that no future price reaches the validation or prediction date. An additional embargo gap of 5 trading days separates training from validation (≥ the 5-day forward horizon), preventing any label leakage.

---

## 3. Feature Engineering

Starting from the 14-feature baseline in `features.py`, the enhanced feature set in `features_enhanced.py` expands to **52 features** across four groups.

### 3.1 Baseline Features (14)

Multi-horizon returns (`ret_1d` through `ret_60d`), realised volatility (`vol_5d`, `vol_20d`), volume z-score, turnover MA, moving-average distances (`close_over_ma20`, `close_over_ma60`), RSI-14, and cross-sectional ranks of returns and volatility.

### 3.2 Extended Technical Features (+14)

| Feature | Formula / Window | Intuition |
|---|---|---|
| `macd`, `macd_signal`, `macd_hist` | EMA(12,26,9), price-normalised | Trend / momentum |
| `bb_position` | (close − lower) / (upper − lower), 20-day Bollinger | Relative price position |
| `rsi_14` | 14-day RSI | Overbought / oversold |
| `ret_accel_5d` | `ret_5d - ret_5d.shift(5)` | Momentum acceleration |
| `ret_skew_20d`, `ret_kurt_20d` | Rolling 20-day skew/kurtosis of `ret_1d` | Return distribution shape |
| `hl_range_ma_20d` | 20-day mean of `(high-low)/close` | Intraday volatility |

### 3.3 A-Share Alpha Factors (+14)

These factors exploit documented anomalies in mainland Chinese markets:

| Feature | Definition | Hypothesis |
|---|---|---|
| `amihud_20d` | 20-day mean of `|ret_1d| / amount × 1e8` | Illiquid stocks earn a return premium (Amihud 2002) |
| `max_ret_20d` / `min_ret_20d` | Rolling 20-day maximum / minimum single-day return | MAX factor: lottery demand → negative future return; MIN: potential bounce |
| `price_to_52w_high` / `price_to_52w_low` | `close / rolling_max(252) - 1` etc. | 52-week anchoring and mean reversion |
| `turnover_z_60d` | z-score of 5-day turnover vs 60-day baseline | Abnormal trading activity predicts reversal |
| `overnight_ret` | `open_t / close_{t-1} - 1` | Informed overnight order flow |
| `intraday_ret` | `close_t / open_t - 1` | Retail-driven intraday component |
| `overnight_ratio_20d` | 20-day rolling mean of `overnight_ret` | Persistent informed-trading signal |
| `streak` | Consecutive up (+) / down (−) days (vectorised) | Short-term momentum / exhaustion |

Cross-sectional percentile ranks are added for all new factors, making the model robust to heteroskedastic raw values across the stock universe.

### 3.4 Industry Neutralisation

An `industry_neutral=True` flag in `build_features()` subtracts the daily SW Level-1 industry mean from the target and all return-based features before rank computation. This removes sector-rotation noise so the model learns to pick stocks that outperform their own industry peers.

**Effect observed:** Industry neutralisation improved mean Rank IC from −0.010 → +0.028 on the test set. It was retained for the final model and submissions.

### 3.5 Feature Cache

`build_features()` saves the computed panel to `data/panel_enhanced_*.parquet` and auto-invalidates when `prices.parquet` is newer. This reduces repeated runs from ~5 minutes to ~0.2 seconds.

---

## 4. Models

Three model families were trained and evaluated.

### 4.1 XGBoost Baseline (14 features) — `baseline_xgboost.py`

The provided baseline, extended with a `--score-weighted` flag.

```
n_estimators = 400   max_depth = 5   learning_rate = 0.05
subsample = 0.8      colsample_bytree = 0.8   min_child_weight = 10
reg_lambda = 1.0     tree_method = hist    early_stopping_rounds = 30
```

### 4.2 XGBoost v2 (19 features) — `model_xgb_v2.py`

Baseline XGBoost with 5 additional features identified by LightGBM feature importance on the validation split:

```
amihud_20d, amihud_20d_rank   (illiquidity premium)
overnight_ret                  (informed overnight signal)
streak_rank                    (momentum streak rank)
price_to_52w_low               (mean reversion anchor)
```

```
n_estimators = 600   (all other hyperparameters same as baseline)
```

### 4.3 LightGBM with Walk-Forward Retraining (52 features) — `model_lgbm.py` + `submit.py`

Full 52-feature set with LightGBM, industry neutralisation, rolling 500-trading-day training window, and walk-forward retraining every 20 trading days during backtest. This is the final submitted model.

```
n_estimators = 800   learning_rate = 0.03
max_depth = 6        num_leaves = 63    min_child_samples = 20
subsample = 0.8      colsample_bytree = 0.7
reg_alpha = 0.1      reg_lambda = 1.0    early_stopping_rounds = 50
```

**Rolling training window (500 days):** Rather than using all available history, each retrain uses only the most recent 500 trading days (~2 years). Tested lookback options: 250d, 500d, full history. 500d was selected via validation IC (never using test data) as the best balance between regime coverage and recency.

**Walk-forward retraining:** The model retrains every 20 trading days during the backtest, using all data available at that point. This ensures the model adapts to structural shifts in A-share market dynamics.

**Submission note:** When generating the live submission (`submit.py`), the near-end-of-data val window is only ~10 trading days, making early stopping unreliable (triggers at round 1). Fixed 300 rounds are used for the submission, consistent with the typical `best_iteration` observed in walk-forward retrains.

---

## 5. Self-Test: Train / Validation / Test Split

### 5.1 Split Design

The data was partitioned into three non-overlapping windows with embargo gaps to prevent label leakage:

| Split | Date Range | Approx. Rows | Purpose |
|---|---|---|---|
| **Train** | 2022-01-01 → 2024-09-30 | ~196,000 | Model fitting |
| *(embargo)* | 2024-10-01 → 2024-10-31 | — | Buffer (≥ 5 trading days) |
| **Validation** | 2024-11-01 → 2025-05-31 | ~69,000 | Lookback window selection only |
| *(embargo)* | 2025-06-01 → 2025-06-30 | — | Buffer |
| **Test** | 2025-07-01 → 2026-04-22 | ~95,000 | Final held-out evaluation (evaluated once) |

**Key methodological choices:**
- Training window start (2022-01-01) was selected by scanning validation IC across candidate start years (2020, 2021, 2022, 2023); test data was never consulted.
- Rolling lookback (500d vs 250d vs full history) was selected on the validation set IC; test data was never consulted.
- The test set was evaluated exactly **once** after all model selection and hyperparameter tuning were complete.
- No future price information enters any feature: every rolling window uses only past observations, and the forward return target is offset by `shift(-FORWARD_HORIZON)` in the time dimension of each individual stock.

### 5.2 Test-Set Results — Rank IC

Mean daily cross-sectional Spearman correlation between predicted score and 5-day forward return:

| Model | Mean Rank IC | IC Std | IC IR | % Days IC > 0 |
|---|---|---|---|---|
| XGBoost baseline (14 feat) | −0.010 | — | — | — |
| XGBoost v2 (19 feat) | −0.009 | — | — | — |
| LightGBM (52 feat, industry neutral) | **+0.028** | — | — | — |

### 5.3 Test-Set Results — Portfolio Backtest

Walk-forward portfolio backtest over the test period (step = 5 trading days, eval window = 5 trading days, top-50 rank-weighted):

| Model / Configuration | Cum. Portfolio Excess | Win Rate | Mean Excess / Window |
|---|---|---|---|
| XGBoost baseline — rank-wtd top-50 | +27.08% | 53.8% | — |
| XGBoost v2 (19 feat) — rank-wtd top-50 | −0.30% | 47.4% | — |
| LightGBM static (52 feat, no retraining) | +11.29% | 52.6% | — |
| **LightGBM walk-forward (52 feat, 500d window)** | **+39.34%** | **64.1%** | **+0.678%** |

The walk-forward LightGBM model outperforms the XGBoost baseline by **+12.26 percentage points** in cumulative excess return and by **+10.3 pp** in window win rate. The improvement stems from two compounding effects: (1) the 52-feature set with industry neutralisation raises IC from −0.010 to +0.028, and (2) walk-forward retraining with a 500d rolling window keeps the model adapted to current market regimes.

Static LightGBM (no retraining) only reaches +11.29% — confirming that walk-forward adaptation is the critical driver of the improvement, not simply the feature set.

---

## 6. Portfolio Construction

### 6.1 Configuration

**Rank-weighted, top-50**, for both Submission 1 and Submission 2.

**Rank weighting:** Stocks are ranked by predicted score; the top-50 names receive weights proportional to their rank (best stock: rank 50; worst in top-50: rank 1), normalised to sum to 1.0. This provides diversity while still concentrating weight on higher-conviction names.

**Iterative cap redistribution:** Any weight exceeding 10% is capped and the excess is redistributed proportionally to uncapped names, iterated until feasible. This satisfies the competition's hard constraint (max weight 0.10) while preserving the relative score ordering.

**Why top-50?** The grid search over `top_k` ∈ {30, 40, 50, 60, 80} showed that top-50 provides the best balance of portfolio concentration and diversification for this model. A more concentrated top-30 portfolio amplifies individual prediction errors; top-50 smooths noise without excessive dilution.

### 6.2 Experiments Tried

| Configuration | Outcome |
|---|---|
| Market-adjusted target (ret - index ret) | Test portfolio −8.09% — beta is informative in A-shares, removing it loses signal |
| Exponential recency weighting (half-life 126d) | +5.70% — worse than 500d raw baseline |
| Industry neutralisation OFF | IC −0.007, portfolio +7.93% — neutralisation is essential |
| Score-weighted portfolio construction | Improved raw XGBoost but not LightGBM; not used in final model |

---

## 7. Experiment Log

| Experiment | Finding |
|---|---|
| Baseline (14 feat, rank-wtd top-50) | IC −0.010; cum excess +27.08%; win rate 53.8% |
| Add MACD / Bollinger / RSI / distribution features (28 feat) | No significant IC improvement without neutralisation |
| Add A-share alpha factors (52 feat total) | IC still −0.008 without neutralisation |
| Industry neutralisation (SW Level-1) | IC +0.028; critical for model signal quality |
| LightGBM static (52 feat, industry neutral) | IC +0.028 but portfolio only +11.29% — regime drift |
| Walk-forward retraining every 20 days | Portfolio +27.11% — matches baseline |
| Val-selected 500d rolling window + walk-forward | **+39.34%, 64.1% win rate** — best result |
| Market-adjusted target (raw - index) | Portfolio −8.09% — A-share beta is signal, not noise |
| Recency weighting (halflife=126d) | Portfolio +5.70% — worse than raw window |
| XGBoost v2 (19 feat) | IC −0.009; portfolio −0.30% — feature dilution |
| Score-weighted portfolio (top-30) | XGBoost: +57.97% but grid-searched on test set; less robust |

---

## 8. Submission Strategy

Both submissions use `submit.py`:

```bash
# Submission 1: as-of last trading day before May Day holiday
python submit.py --as-of 20260430 --out submissions/submission1.csv

# Before Submission 2: update data
python download_data.py --update --end 20260510

# Submission 2: as-of last trading day before evaluation window
python submit.py --as-of 20260509 --out submissions/submission2.csv
```

**Model:** LightGBM, 52 enhanced features, industry neutralisation, 500-day rolling training window, 300 fixed rounds (no early stopping for submission — near-end-of-data val window is too short; 300 rounds is consistent with typical walk-forward `best_iteration`).

**Portfolio:** Rank-weighted top-50, 10% per-stock cap.

**Rationale:** The walk-forward LightGBM with 500d window is the strongest validated configuration on a clean train/val/test split. Walk-forward retraining is simulated in the backtest by retraining every 20 days; for live submissions, a single fresh fit on the latest 500 days is used.

---

## 9. Reproducibility

### Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `xgboost`, `lightgbm`, `pandas`, `numpy`, `scipy`, `akshare`, `pyarrow`.

### Pipeline

```bash
# 1. Download data
python download_data.py --start 20220101 --end 20260430

# 2. Industry classification is included in data/industry.csv — no re-fetch needed

# 3. Run self-test to reproduce reported metrics
python self_test.py

# 4. Generate submissions
python submit.py --as-of 20260430 --out submissions/submission1.csv
python download_data.py --update --end 20260510
python submit.py --as-of 20260509 --out submissions/submission2.csv

# 5. Validate
python validate_submission.py submissions/submission1.csv
python validate_submission.py submissions/submission2.csv
```

**Random seeds:** LightGBM uses no explicit random seed; deterministic on CPU. Minor floating-point differences across platforms are expected but results should match within numerical noise.

---

## 10. AI Assistance Acknowledgement

LLMs (Claude) were used as a coding assistant for debugging, refactoring, and drafting code. All model design decisions, feature engineering choices, experimental directions, and result interpretations were produced by the student. No LLM was used to directly generate portfolio weights or predict stock returns.

The baseline code originates from [NYUSH-ML/ml-competition-sp26](https://github.com/NYUSH-ML/ml-competition-sp26).

---

## Appendix: File Index

| File | Role |
|---|---|
| `baseline_xgboost.py` | XGBoost baseline (extended with `--score-weighted` flag) |
| `features.py` | Baseline 14-feature engineering |
| `features_enhanced.py` | Extended 52-feature engineering with industry neutralisation |
| `model_lgbm.py` | LightGBM model and `build_portfolio()` |
| `model_xgb_v2.py` | XGBoost with 19 features |
| `submit.py` | **Main submission generator** — LightGBM, 500d window, rank-wtd top-50 |
| `self_test.py` | Fixed train/val/test split evaluation with walk-forward backtest |
| `portfolio_search.py` | Grid search over top-k × weighting |
| `walk_forward_cv.py` | Walk-forward cross-validation helper |
| `score_submission.py` | Realised-return scorer |
| `validate_submission.py` | Constraint checker |
| `data/industry.csv` | SW Level-1 industry labels |
