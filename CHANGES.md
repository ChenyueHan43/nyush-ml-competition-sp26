# My Changes — CSI500 Stock Selection Competition

Changes made on top of the original baseline from [NYUSH-ML/ml-competition-sp26](https://github.com/NYUSH-ML/ml-competition-sp26).

## New Files

### `features_enhanced.py`
Extended feature engineering on top of the original `features.py` (14 features → 52 features).

**Added features:**
- Technical: MACD (12/26/9), Bollinger Band position, RSI (14), price acceleration, return skewness/kurtosis, intraday range
- A-share alpha factors: Amihud illiquidity (20d), MAX/MIN factor (extreme return in 20d), 52-week high/low distance, turnover anomaly z-score, overnight vs intraday return split, consecutive up/down streak
- Cross-sectional ranks for all new factors

**Other improvements:**
- `target_3d`: 3-day forward return target (in addition to 5-day), useful for submission 1 (3-day eval window)
- `streak` vectorized with numpy groupby — ~100x faster than the original Python for-loop
- Auto feature cache: saves panel to `data/panel_enhanced_*.parquet`; auto-invalidates when `prices.parquet` is newer. Reduces repeated runs from ~5 min to 0.2s.
- Industry neutralization (`industry_neutral=True`): subtracts SW1 industry mean from target and return features per date. Improves Rank IC from -0.008 → +0.029 but reduces absolute portfolio excess return (not used for final submission).

### `model_lgbm.py`
LightGBM model with 3-window ensemble (full history, ~2 years, ~1 year).

| | Baseline XGBoost | LightGBM |
|---|---|---|
| Features | 14 | 52 |
| Model | XGBoost | LightGBM |
| Training | Single window | Ensemble of 3 windows |
| Test Rank IC | -0.008 | -0.008 (without neutralization) / +0.029 (with) |

### `model_xgb_v2.py`
XGBoost with 19 features = baseline 14 + 5 high-importance features identified via LightGBM feature importance:
- `amihud_20d`, `amihud_20d_rank` — illiquidity premium
- `overnight_ret` — informed overnight trading signal
- `streak_rank` — consecutive up/down momentum rank
- `price_to_52w_low` — mean reversion signal

### `portfolio_search.py`
Grid search over portfolio construction parameters on the test set (2025-07-01 to 2026-04-22).

Tests: top_k ∈ {30, 40, 50, 60, 80} × weighting ∈ {rank-weighted, equal-weight, score-weighted}

**Key result (XGBoost baseline):**

| Weighting | top-k | Cum. Excess | Win Rate | Mean/Window |
|-----------|-------|-------------|----------|-------------|
| score-weighted | 30 | +57.97% | 53.8% | +0.956% |
| rank-weighted  | 30 | +38.24% | 61.5% | +0.653% |
| rank-weighted  | 50 | +27.08% | 53.8% | +0.469% |

### `self_test.py`
Fixed train/val/test split evaluation (required for the grading self-test component):

| Split | Period | Rows |
|-------|--------|------|
| Train | 2022-01-01 – 2024-09-30 | ~196k |
| Val   | 2024-11-01 – 2025-05-31 | ~69k  |
| Test  | 2025-07-01 – 2026-04-22 | ~95k  |

Reports Rank IC, IC std, IC IR, % days IC > 0, and walk-forward portfolio backtest for all three models.

### `data/industry.csv`
SW1 (Shenwan Level 1) industry classification for all 499 CSI500 constituents, fetched via akshare. Used for industry neutralization.

## Model Comparison (Test Set: 2025-07-01 to 2026-04-15)

| Model | Rank IC | Portfolio Cum. Excess | Win Rate |
|-------|---------|----------------------|----------|
| XGBoost baseline (14 feat) | -0.008 | +23.76% (top-50) / **+57.97%** (score-wtd top-30) | 52.6% / 53.8% |
| XGBoost v2 (19 feat) | -0.009 | -0.30% | 47.4% |
| LightGBM (52 feat) | -0.008 | +6.54% | 52.6% |
| LightGBM + industry neutral | **+0.029** | +27.12% | 51.3% |

## Submission Strategy

- **Submission 1 & 2**: XGBoost baseline, score-weighted, top-30
- Industry neutralization improves IC but hurts absolute portfolio excess — not used
- Eval windows are short (3 and 5 trading days), so mean excess per window (+0.956%) is the relevant metric
