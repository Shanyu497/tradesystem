"""
labels.py
==========
把預測目標從「明天價格是多少」轉換成「分類問題」，對應 Phase 1 討論的結論：
直接猜價格效果差，改猜「未來 N 天報酬率是否超過門檻」，並輸出機率而非二元判斷。

一個重要的細節：最近 horizon 天的資料，因為還沒有「未來」可以驗證，
標籤一定是 NaN——這些不是資料有問題，而正是我們最想要模型預測的「現在」這幾筆。
呼叫端要把「有標籤的歷史資料」拿去訓練，「無標籤的最新資料」拿去預測。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_label(price_df: pd.DataFrame, horizon: int = 5, threshold: float = 0.03) -> pd.Series:
    """
    horizon: 往後看幾個交易日
    threshold: 報酬率門檻，例如 0.03 代表「未來 horizon 天內，最終收盤漲幅 > 3%」才算 1

    回傳跟 price_df 等長的 Series，最後 horizon 筆會是 NaN（尚無法驗證的「現在」）。
    """
    future_return = build_future_return(price_df, horizon)

    label = pd.Series(np.nan, index=future_return.index, dtype="float64")
    valid_mask = future_return.notna()
    label[valid_mask] = (future_return[valid_mask] > threshold).astype(float)
    return label


def build_future_return(price_df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    原始的未來 N 天報酬率（不是分類標籤，是實際數值），
    回測要算「這筆交易實際賺賠多少」時需要這個，分類模型訓練則只需要 build_label() 的 0/1。

    真實市場資料偶爾會有異常值（例如某天收盤價被記錄成接近 0，可能是資料源的錯誤或
    未還原的除權息造成的價格斷層），這種情況算出來的報酬率會是無限大，
    如果不處理，會讓後面的回測/風險指標整個失真（一筆髒資料就能讓權益曲線變成 inf 或 NaN）。
    這裡直接把無限值視為缺失值，跟其他資料不足的日期一樣被下游流程排除。
    """
    close = price_df.sort_values("date")["close"].reset_index(drop=True)
    future_return = close.shift(-horizon) / close - 1
    return future_return.replace([np.inf, -np.inf], np.nan)
