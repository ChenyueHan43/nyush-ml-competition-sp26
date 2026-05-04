"""
submit.py — 生成 CSI500 竞赛提交文件

模型配置（来自 self_test.py 的无泄漏评估结果）：
  - 模型    : LightGBM，52 个增强特征（features_enhanced.py）
  - 特征集  : 技术指标 + A 股 alpha 因子 + 截面排名，行业中性化
  - 训练窗口: 500 个交易日（在 val 集上选定，优于 250d 和全量）
  - 目标    : 5 日前向收益率（原始，不做市场调整）
  - 组合    : rank-weighted，取预测分数最高的 top-50，单股上限 10%

self-test 结果（无数据泄漏）：
  Rank IC          : +0.028（baseline -0.010）
  累计超额收益      : +39.34%（baseline +27.08%）
  窗口胜率          : 64.1%（baseline 53.8%）

用法：
  # Submission 1（as-of 4月30日，五一假前最后交易日）
  python submit.py --as-of 20260430 --out submissions/submission1.csv

  # Submission 2（as-of 5月9日，评估窗口前最后交易日）
  python submit.py --as-of 20260509 --out submissions/submission2.csv

  # 生成后验证
  python validate_submission.py submissions/submission1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features_enhanced import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, prediction_frame,
)
from model_lgbm import build_portfolio, MIN_STOCKS, MAX_WEIGHT

DATA_DIR = Path(__file__).parent / "data"

# ── 超参数（与 self_test.py 一致）────────────────────────────────────────────
LOOKBACK_DAYS = 500      # 训练窗口：最近 500 个交易日（在 val 集上确定）
VAL_DAYS      = 10       # 末尾 10 个交易日作为早停验证集
EMBARGO_DAYS  = 5        # val 和 train 之间的 embargo（>= FORWARD_HORIZON）
DEFAULT_TOP_K = 50       # 组合持股数

LGB_PARAMS = dict(
    objective        = "regression",
    metric           = "rmse",
    n_estimators     = 800,
    learning_rate    = 0.03,
    max_depth        = 6,
    num_leaves       = 63,
    min_child_samples= 20,
    subsample        = 0.8,
    subsample_freq   = 1,
    colsample_bytree = 0.7,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    n_jobs           = -1,
    verbose          = -1,
)


def train(panel: pd.DataFrame, as_of_ts: pd.Timestamp) -> lgb.LGBMRegressor:
    """
    以 as_of_ts 为基准，用最近 LOOKBACK_DAYS 个交易日的数据训练模型。

    时间轴示意：
      [ ... 旧数据（丢弃）... | <-- 500d --> train_start ... train_end | embargo | val | as_of ]
                                                                         ^5d gap^  ^10d^

    - train_end / val 的划分保证训练标签（5日前向收益）不与验证特征重叠
    - 500d 窗口在 val 集上通过 IC 选定（见 self_test.py select_lookback_on_val）
    """
    trading_dates = np.sort(panel["date"].unique())

    # 预测基准日对应的 target cutoff：target_t 用到 t+5 的价格，
    # 所以训练数据截止到 as_of - FORWARD_HORIZON，避免 target 泄漏
    as_of_idx   = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx  = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])

    # 取全部可用训练数据，再截取最近 LOOKBACK_DAYS 天
    train_pool = panel[panel["date"] <= train_cutoff].dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    )
    pool_dates = np.sort(train_pool["date"].unique())
    if len(pool_dates) > LOOKBACK_DAYS:
        window_start = pd.Timestamp(pool_dates[-LOOKBACK_DAYS])
        train_pool = train_pool[train_pool["date"] >= window_start]
        pool_dates = np.sort(train_pool["date"].unique())

    if len(pool_dates) < VAL_DAYS + EMBARGO_DAYS + 40:
        raise RuntimeError("训练数据不足，请检查 prices.parquet 日期范围。")

    # 末尾 10 天作为早停验证集，中间 5 天 embargo 丢弃
    val_start = pd.Timestamp(pool_dates[-VAL_DAYS])
    train_end = pd.Timestamp(pool_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_df  = train_pool[train_pool["date"] <= train_end]
    val_df    = train_pool[train_pool["date"] >= val_start]

    print(f"   训练: {len(train_df):,} 行，"
          f"{train_df['date'].min().date()} → {train_df['date'].max().date()}")
    print(f"   验证: {len(val_df):,} 行，"
          f"{val_df['date'].min().date()} → {val_df['date'].max().date()}")

    # 提交时用固定 300 轮，不做早停。
    # val 窗口只有最近 ~10 天，信噪比低，早停会在第 1 轮就触发。
    # 300 轮来自 self_test walk-forward 的典型 best_iteration 区间。
    params_submit = {**LGB_PARAMS, "n_estimators": 300}
    model = lgb.LGBMRegressor(**params_submit)
    model.fit(
        train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
    )
    print(f"   训练完成（固定 300 轮）")
    return model


def main():
    parser = argparse.ArgumentParser(description="生成 CSI500 竞赛提交文件")
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"),
                        help="价格数据路径")
    parser.add_argument("--as-of", default=None,
                        help="预测基准日 YYYYMMDD（默认：数据最新日期）")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"持股数量（默认 {DEFAULT_TOP_K}，最少 {MIN_STOCKS}）")
    parser.add_argument("--out", default="submission.csv",
                        help="输出文件路径")
    args = parser.parse_args()

    # ── 1. 加载数据 ──────────────────────────────────────────────────────────
    print(f">> 加载数据: {args.prices}")
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"   {len(prices):,} 行，{prices['stock_code'].nunique()} 只股票，"
          f"{prices['date'].min().date()} → {prices['date'].max().date()}")

    # ── 2. 构建特征（行业中性化）────────────────────────────────────────────
    print(">> 构建特征（industry_neutral=True）...")
    panel = build_features(prices, industry_neutral=True)

    # ── 3. 确定预测基准日 ────────────────────────────────────────────────────
    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    print(f">> 预测基准日: {as_of_ts.date()}")

    # ── 4. 训练模型 ──────────────────────────────────────────────────────────
    print(f">> 训练 LightGBM（最近 {LOOKBACK_DAYS} 个交易日）...")
    model = train(panel, as_of_ts)

    # ── 5. 预测 ──────────────────────────────────────────────────────────────
    pred_df = prediction_frame(panel, as_of=as_of_ts)
    if pred_df.empty:
        raise RuntimeError(f"基准日 {as_of_ts.date()} 无可用数据，请检查 --as-of 参数。")
    print(f">> 对 {len(pred_df)} 只股票打分...")
    pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))

    # ── 6. 构建组合（rank-weighted，10% 单股上限）───────────────────────────
    scores  = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(scores, top_k=args.top_k)

    # ── 7. 保存并输出摘要 ────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)

    print(f"\n>> 已写入 {len(out)} 只股票到 {out_path}")
    print(f"   权重汇总: min={out['weight'].min():.4f}  "
          f"max={out['weight'].max():.4f}  "
          f"sum={out['weight'].sum():.6f}")
    print(f"\n   验证命令: python validate_submission.py {out_path}")


if __name__ == "__main__":
    main()
