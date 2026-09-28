"""
trade_levels.py
=================
把「機率訊號」轉換成具體的「進場價、停損價、停利價」——signals/model.py 只回答
「未來 N 天上漲超過門檻的機率是多少」，這裡回答「如果現在要進場，該設在哪裡」。

方法（刻意選擇跟隨市場真實波動的做法，不是憑感覺畫線）：
- 進場價：目前最新收盤價（機率是「從現在算起」的，進場基準自然是現在的價格）。
- 停損價：用 ATR（平均真實區間，signals/indicators.py 已經有）衡量這檔標的最近的
  正常波動幅度，停損設在「進場價 - atr_multiplier 倍 ATR」——比隨便抓一個近期低點
  更能反映「這個波動有多正常」，波動大的標的停損自然抓寬一點，不會太容易被雜訊洗出場。
- 停利價：直接用模型本身的預測目標（threshold）反推——模型是在賭「未來 horizon 天
  漲幅超過 threshold%」，停利價就設在「進場價 × (1 + threshold)」，這樣停利目標
  跟模型實際在預測的東西是同一件事，不是另外發明一個數字。

這不是「精準預測未來會漲到哪裡」，是把模型/波動度換算成具體可以下單的價位，
本質上跟 risk/kelly.py 一樣，只是把「機率/賠率」換成「進場價/停損價/停利價」的形式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from signals.indicators import atr


@dataclass
class TradeLevels:
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    atr_value: float
    risk_pct: float           # (entry - stop_loss) / entry，萬一停損會虧多少百分比
    reward_pct: float          # (take_profit - entry) / entry，達成停利目標會賺多少百分比
    risk_reward_ratio: float   # reward_pct / risk_pct，越高代表「賺的時候賺得比虧的時候多」


def suggest_trade_levels(price_df: pd.DataFrame, threshold: float,
                           atr_multiplier: float = 1.5, atr_window: int = 14) -> Optional[TradeLevels]:
    """
    price_df: 符合 PRICE_SCHEMA 的價量資料（單一標的，需要 high/low/close）
    threshold: 跟訓練模型時用的同一個 threshold（例如 0.03），停利價由此反推

    資料不足以算出 ATR（暖身期不夠）、或算出來的停損價不合理時回傳 None。
    """
    price_df = price_df.sort_values("date").reset_index(drop=True)
    atr_series = atr(price_df, window=atr_window)
    latest_atr = atr_series.iloc[-1]

    if pd.isna(latest_atr):
        return None

    entry_price = float(price_df["close"].iloc[-1])
    stop_loss_price = entry_price - atr_multiplier * float(latest_atr)
    take_profit_price = entry_price * (1 + threshold)

    if stop_loss_price <= 0 or stop_loss_price >= entry_price:
        # 防呆：ATR 異常大（例如資料剛好碰到暴漲暴跌）時停損價可能不合理，
        # 這種情況不給建議，總比給一個誤導性的數字好。
        return None

    risk_pct = (entry_price - stop_loss_price) / entry_price
    reward_pct = (take_profit_price - entry_price) / entry_price
    risk_reward_ratio = reward_pct / risk_pct if risk_pct > 0 else float("nan")

    return TradeLevels(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        atr_value=float(latest_atr),
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        risk_reward_ratio=risk_reward_ratio,
    )
