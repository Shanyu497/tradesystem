"""
features.py
============
把 indicators.py 裡的原子指標組合成完整的特徵矩陣。

設計原則：
- 所有特徵都做過「正規化」處理（比率、位置、報酬率），而不是直接用原始價格數字，
  這樣不同價位的股票（例如 2330 一股 2000+ 元 vs 0050 一股 50 元）可以放進同一個模型訓練，
  模型學到的是「相對關係」而不是「絕對價位」。
- 回傳的 DataFrame 保留 'date' 欄位方便對齊，但不含 'close' 等原始價格
  （避免模型直接學到價位高低這種無意義的捷徑）。
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ind

# 這個常數之後 model.py / signal_report.py 都會用到，
# 統一在這裡定義，避免特徵名稱在不同檔案裡打錯字或漏掉。
FEATURE_COLUMNS = [
    "rsi_14",
    "macd_hist",
    "bb_percent_b",
    "bb_bandwidth",
    "atr_pct",
    "volume_ratio_20",
    "momentum_5",
    "momentum_20",
    "sma_ratio_50",
    "channel_position_20",
    "new_high_breakout_20",
]


def build_feature_matrix(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    輸入：符合 PRICE_SCHEMA 的價量 DataFrame（單一標的，已按日期排序）
    輸出：包含 'date' + FEATURE_COLUMNS 的 DataFrame，
          指標暖身期不足的列會是 NaN（由呼叫端決定要 dropna 或保留）
    """
    df = price_df.sort_values("date").reset_index(drop=True)
    close = df["close"]

    macd_line, macd_signal, macd_hist = ind.macd(close)
    percent_b, bandwidth, _ = ind.bollinger_bands(close)
    sma_50 = ind.sma(close, 50)

    features = pd.DataFrame({
        "date": df["date"],
        "rsi_14": ind.rsi(close, 14),
        "macd_hist": macd_hist / close,  # 除以收盤價正規化，避免不同價位股票數值差太多
        "bb_percent_b": percent_b,
        "bb_bandwidth": bandwidth,
        "atr_pct": ind.atr(df, 14) / close,
        "volume_ratio_20": ind.volume_ratio(df, 20),
        "momentum_5": ind.momentum(close, 5),
        "momentum_20": ind.momentum(close, 20),
        "sma_ratio_50": close / sma_50 - 1,  # 目前價格相對 50 日均線的偏離幅度
        "channel_position_20": ind.channel_position(df, 20),
        "new_high_breakout_20": ind.is_new_high_breakout(df, 20),
    })

    return features
