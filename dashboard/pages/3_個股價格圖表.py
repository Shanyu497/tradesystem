"""
3_個股價格圖表.py
===================
Phase 5 新增的純視覺化頁面（Phase 1~4 沒有這個功能）：讀 Phase 2 存好的
processed 價量資料，畫 K 線圖疊加均線。均線計算重用 signals/indicators.py
的 sma()，不重新刻一份指標邏輯。
"""

from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _cache import PROJECT_ROOT, cached_load_watchlist
from signals.indicators import sma

st.set_page_config(page_title="個股價格圖表", page_icon="📉", layout="wide")
st.title("📉 個股價格圖表")

watchlist = cached_load_watchlist()

col1, col2 = st.columns(2)
with col1:
    market = st.selectbox("市場", options=list(watchlist.keys()))
with col2:
    symbol = st.selectbox("標的", options=watchlist.get(market, []))

path = PROJECT_ROOT / "data" / "processed" / market / "price" / f"{symbol}.parquet"
if not path.exists():
    st.warning(f"找不到 {market}/{symbol} 的價量資料，請先執行 `py pipeline.py` 抓取資料。")
    st.stop()

price_df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)

lookback_days = st.slider("顯示最近幾天", min_value=30, max_value=len(price_df), value=min(250, len(price_df)))
plot_df = price_df.tail(lookback_days)

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=plot_df["date"], open=plot_df["open"], high=plot_df["high"],
    low=plot_df["low"], close=plot_df["close"], name=symbol,
))
# 均線用「全部歷史資料」算完再切片顯示範圍，而不是只用畫面上這幾天算，
# 避免顯示範圍開頭那幾天因為暖身期不足而缺一截均線。
for window in (20, 50):
    sma_series = sma(price_df["close"], window).loc[plot_df.index]
    fig.add_trace(go.Scatter(x=plot_df["date"], y=sma_series, name=f"SMA{window}", line=dict(width=1)))

fig.update_layout(xaxis_rangeslider_visible=False, height=600, title=f"{market}/{symbol}")
st.plotly_chart(fig, use_container_width=True)

fig_vol = go.Figure(go.Bar(x=plot_df["date"], y=plot_df["volume"], name="成交量"))
fig_vol.update_layout(height=200, title="成交量")
st.plotly_chart(fig_vol, use_container_width=True)
