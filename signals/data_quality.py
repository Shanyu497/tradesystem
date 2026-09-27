"""
data_quality.py
=================
在價量資料最源頭做一次「合理性檢查」，抓出不合理的單日價格跳動
（通常是資料源的錯誤，例如某天收盤價被記錄成接近 0，或是未正確處理的除權息斷層）。

為什麼要在這一層處理，而不是只在算報酬率的時候擋掉？
因為同一筆壞資料，不只會污染「未來報酬率」（label/future_return），
也會直接污染「動能」「移動平均」這類技術指標特徵本身
（例如 momentum_5 = close.pct_change(5)，如果某天的收盤價是錯的，
 這個特徵在那天前後好幾天都會被污染，不是只有那一天）。
在最源頭把異常值濾掉，才能確保下游所有計算（指標 + 標籤）都一致地排除掉壞資料。

處理方式：偵測日報酬率絕對值超過門檻的列，直接從價量資料中刪除（不是填補，
避免用可能同樣不準的方式「猜」一個合理值）。這是保守但誠實的做法——
寧可少幾筆資料，也不要讓一筆壞資料混進去卻假裝沒事。
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("data_quality")

# 大型權值股/ETF 正常情況下，單日漲跌幅超過這個比例的機率極低
# （台股當沖/個股單日漲跌停通常是 10%，美股偶爾會有更大波動但 50% 已經是極端值），
# 超過這個門檻，視為資料異常而非真實市場走勢。
DEFAULT_MAX_DAILY_MOVE = 0.5


def clean_price_data(price_df: pd.DataFrame, max_daily_move: float = DEFAULT_MAX_DAILY_MOVE,
                       symbol_label: str = "") -> pd.DataFrame:
    """
    偵測並移除單日報酬率絕對值超過 max_daily_move 的列，回傳清洗過的 DataFrame。
    如果有濾掉任何列，會記錄一則警告，方便你回頭去源頭資料核對是不是真的有問題
    （也有可能是真實的重大事件，例如減資、下市前的暴跌，這種情況建議手動確認後
      再決定要不要把這幾筆資料留著分析）。
    """
    df = price_df.sort_values("date").reset_index(drop=True)

    if df["close"].le(0).any():
        bad_dates = df.loc[df["close"] <= 0, "date"].tolist()
        logger.warning(f"{symbol_label} 發現收盤價 <= 0 的異常資料，已排除，日期: {bad_dates}")
        df = df[df["close"] > 0].reset_index(drop=True)

    daily_return = df["close"].pct_change()
    anomaly_mask = daily_return.abs() > max_daily_move

    if anomaly_mask.any():
        anomaly_dates = df.loc[anomaly_mask, "date"].tolist()
        logger.warning(
            f"{symbol_label} 發現 {anomaly_mask.sum()} 筆單日漲跌幅超過 {max_daily_move:.0%} 的異常資料，"
            f"已排除，日期: {anomaly_dates}（如果這是真實事件如減資/暴跌，請手動確認後決定是否保留）"
        )
        df = df[~anomaly_mask].reset_index(drop=True)

    return df
