"""
Enhanced feature engineering for CSI500 stock selection.

Extends the baseline with:
- More return windows (3d, 120d)
- MACD and signal line
- Bollinger Band position
- Price acceleration (momentum of momentum)
- Return skewness and kurtosis
- Volume OBV trend
- High-low range features
- Cross-sectional ranks for all key features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    # --- returns ---
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d",
    # --- volatility ---
    "vol_5d", "vol_20d", "vol_60d",
    # --- volume ---
    "volume_z_20d", "turnover_ma_20d", "volume_ratio_5_20",
    # --- trend / MA ---
    "close_over_ma5", "close_over_ma20", "close_over_ma60",
    "ma5_over_ma20", "ma20_over_ma60",
    # --- MACD ---
    "macd", "macd_signal", "macd_hist",
    # --- Bollinger ---
    "bb_position",
    # --- RSI ---
    "rsi_14",
    # --- price acceleration ---
    "ret_accel_5d",
    # --- distribution ---
    "ret_skew_20d", "ret_kurt_20d",
    # --- intraday range ---
    "hl_range_ma_20d",
    # --- cross-sectional ranks ---
    "ret_5d_rank", "ret_20d_rank", "vol_20d_rank",
    "ret_1d_rank", "ret_60d_rank", "vol_5d_rank",
    "close_over_ma20_rank", "rsi_14_rank",
]

TARGET_COLUMN = "target_5d"
FORWARD_HORIZON = 5


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    volume = df["volume"].astype(float)

    # --- Returns ---
    df["ret_1d"] = close.pct_change(1)
    df["ret_3d"] = close.pct_change(3)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_60d"] = close.pct_change(60)
    df["ret_120d"] = close.pct_change(120)

    # --- Volatility ---
    df["vol_5d"] = df["ret_1d"].rolling(5).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["vol_60d"] = df["ret_1d"].rolling(60).std()

    # --- Volume features ---
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (volume - vol_mean_20) / vol_std_20
    if "turnover" in df.columns:
        df["turnover_ma_20d"] = df["turnover"].astype(float).rolling(20).mean()
    else:
        df["turnover_ma_20d"] = np.nan
    vol_mean_5 = volume.rolling(5).mean()
    df["volume_ratio_5_20"] = vol_mean_5 / vol_mean_20.replace(0, np.nan) - 1.0

    # --- Moving averages & distance ---
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    df["close_over_ma5"] = close / ma5.replace(0, np.nan) - 1.0
    df["close_over_ma20"] = close / ma20.replace(0, np.nan) - 1.0
    df["close_over_ma60"] = close / ma60.replace(0, np.nan) - 1.0
    df["ma5_over_ma20"] = ma5 / ma20.replace(0, np.nan) - 1.0
    df["ma20_over_ma60"] = ma20 / ma60.replace(0, np.nan) - 1.0

    # --- MACD (12/26/9 EMA) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_line / close.replace(0, np.nan)           # normalised by price
    df["macd_signal"] = signal_line / close.replace(0, np.nan)
    df["macd_hist"] = (macd_line - signal_line) / close.replace(0, np.nan)

    # --- Bollinger Band position ---
    bb_std = close.rolling(20).std()
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = (close - bb_lower) / bb_width  # 0=lower band, 1=upper band

    # --- RSI (14-period) ---
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + up / down)

    # --- Price acceleration (momentum of momentum) ---
    df["ret_accel_5d"] = df["ret_5d"] - df["ret_5d"].shift(5)

    # --- Return distribution (rolling skew & kurtosis) ---
    df["ret_skew_20d"] = df["ret_1d"].rolling(20).skew()
    df["ret_kurt_20d"] = df["ret_1d"].rolling(20).kurt()

    # --- Intraday range (high-low / close) normalised ---
    df["hl_range_ma_20d"] = ((high - low) / close.replace(0, np.nan)).rolling(20).mean()

    # --- Target ---
    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close.replace(0, np.nan) - 1.0

    return df


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    rank_cols = [
        "ret_1d", "ret_5d", "ret_20d", "ret_60d",
        "vol_5d", "vol_20d",
        "close_over_ma20", "rsi_14",
    ]
    for col in rank_cols:
        if col in panel.columns:
            panel[f"{col}_rank"] = panel.groupby("date")[col].rank(
                method="average", pct=True
            )
    return panel


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "stock_code", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    panel = (
        prices.groupby("stock_code", group_keys=False)
        .apply(_per_stock_features)
        .reset_index(drop=True)
    )
    panel = _cross_sectional_ranks(panel)
    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    df = panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()
    return df
