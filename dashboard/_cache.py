"""
_cache.py
==========
Streamlit 頁面共用的資料存取層：把 signal_report.run_report() 和
backtest_report.run_backtest_report() 包上 st.cache_data，避免使用者在頁面上
調整篩選器（horizon/threshold/capital...）時，每次互動都重新訓練模型
（HistGradientBoostingClassifier 訓練不是免費的，尤其標的一多就更明顯）。

cache key 是 st.cache_data 根據函式參數自動決定的，只要參數（horizon、threshold、
symbols...）不同就會重新計算，相同參數則直接吃 cache，TTL 到期或使用者按下
Home 頁面的「強制重新整理」清 cache 後才會重新訓練。

注意：st.cache_data 的參數必須是可雜湊的，所以這裡一律用 tuple 而不是 set
來傳「只看哪些標的」，跟 run_report()/run_backtest_report() 原本用 set 的介面不同，
轉換在這一層做，不影響底層函式。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from backtest_report import run_backtest_report as _run_backtest_report
from pipeline import load_watchlist as _load_watchlist
from signal_report import run_report as _run_report

CACHE_TTL_SECONDS = 3600  # 一小時，跟資料排程的觸發頻率（每天）比起來已經夠新鮮


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="正在訓練模型、計算訊號…")
def cached_run_report(horizon: int, threshold: float, symbols: tuple[str, ...] | None) -> list[dict]:
    symbol_filter = set(symbols) if symbols else None
    return _run_report(horizon=horizon, threshold=threshold, symbol_filter=symbol_filter)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="正在回測、計算建議倉位…")
def cached_run_backtest_report(
    capital: float, horizon: int, threshold: float, prob_threshold: float,
    kelly_fraction: float, max_position_pct: float, symbols: tuple[str, ...] | None,
) -> list[dict]:
    symbol_filter = set(symbols) if symbols else None
    return _run_backtest_report(
        capital=capital, horizon=horizon, threshold=threshold,
        prob_threshold=prob_threshold, kelly_fraction=kelly_fraction,
        max_position_pct=max_position_pct, symbol_filter=symbol_filter,
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def cached_load_watchlist() -> dict[str, list[str]]:
    return _load_watchlist()


def clear_all_caches() -> None:
    cached_run_report.clear()
    cached_run_backtest_report.clear()
    cached_load_watchlist.clear()
