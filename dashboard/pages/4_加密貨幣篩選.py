"""
4_加密貨幣篩選.py
===================
對應 crypto_screener.py：掃描 Binance 成交量前幾大的幣種，找出可信且機率夠高的
短期上漲候選，附上用 ATR/threshold 換算出來的進場/停損/停利價，並畫在 K 線圖上。

誠實提醒（跟 crypto_screener.py 的說明一致）：這個系統對加密貨幣的預測力目前
普遍偏弱，「可信」只代表模型排序能力不是亂猜，不代表機率夠高、更不代表
「極有可能上漲」。掃描範圍內常常沒有任何一檔符合雙重門檻，這是誠實的結果。
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _cache import cached_run_screener
from adapters.crypto_adapter import CryptoAdapter
from crypto_screener import select_candidates
from signals.indicators import sma

st.set_page_config(page_title="加密貨幣篩選", page_icon="🔍", layout="wide")
st.title("🔍 加密貨幣篩選")
st.caption(
    "對應 `py crypto_screener.py`，動態掃描 Binance 成交量前幾大的幣種（不限於 watchlist.json），"
    "找出可信且機率夠高的短期上漲候選，用 ATR/threshold 換算出進場/停損/停利價。"
)
st.warning(
    "這不是「找出極有可能上漲的幣」——這個系統對加密貨幣的預測力目前普遍偏弱（AUC 大多 0.5~0.6，"
    "只比隨機猜好一點）。「可信」只代表模型排序能力不是亂猜，不代表機率夠高，兩個條件都要過門檻"
    "才會列出來；常常掃完一檔都沒有，這是誠實的結果，不是系統故障。進場價/停損價/停利價是把"
    "模型與波動度換算成具體價位的參考值，不是精準預測，不是投資建議。"
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    top_n = st.number_input("掃描幾大幣種（依 24h 成交金額排序）", min_value=5, max_value=100, value=30, step=5)
with col2:
    horizon = st.number_input("預測未來幾天報酬率（horizon）", min_value=1, max_value=60, value=5)
with col3:
    threshold = st.number_input("漲幅門檻（threshold）", min_value=0.0, max_value=1.0, value=0.03, step=0.01, format="%.2f")
with col4:
    min_probability = st.number_input("最低上漲機率門檻", min_value=0.0, max_value=1.0, value=0.5, step=0.05, format="%.2f")

with st.spinner("掃描中，幣數一多會跑比較久（每檔都要重新訓練模型）..."):
    rows = cached_run_screener(top_n, horizon, threshold, 730)

if not rows:
    st.error("取得候選幣清單失敗（Binance API 可能暫時不可用），請稍後再試。")
    st.stop()

candidates = select_candidates(rows, min_probability)

st.caption(f"本次掃描 {len(rows)} 檔有足夠歷史資料的幣，其中 {len(candidates)} 檔同時符合「可信」且「機率 >= {min_probability:.0%}」。")

if not candidates:
    st.info("今天掃描範圍內沒有任何一檔幣同時符合門檻，沒有建議進場的標的。")
    st.stop()

table_df = pd.DataFrame([{
    "代號": r["symbol"],
    "上漲機率": f"{r['probability']:.1%}",
    "AUC": f"{r['auc']:.3f}",
    "進場價": f"{r['levels'].entry_price:,.4f}",
    "停損價": f"{r['levels'].stop_loss_price:,.4f}",
    "停利價": f"{r['levels'].take_profit_price:,.4f}",
    "風險報酬比": f"{r['levels'].risk_reward_ratio:.2f}",
} for r in candidates])

st.dataframe(table_df, width="stretch", hide_index=True)

st.subheader("K 線圖（含建議進場/停損/停利價）")
symbol_options = [r["symbol"] for r in candidates]
picked_symbol = st.selectbox("選擇要看哪一檔的圖", options=symbol_options)
picked = next(r for r in candidates if r["symbol"] == picked_symbol)
lv = picked["levels"]

with st.spinner(f"讀取 {picked_symbol} 價格資料..."):
    adapter = CryptoAdapter()
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=250)).strftime("%Y-%m-%d")
    price_df = adapter.fetch_price(picked_symbol, start_date, end_date).sort_values("date").reset_index(drop=True)

if price_df.empty:
    st.warning(f"找不到 {picked_symbol} 的價格資料。")
    st.stop()

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=price_df["date"], open=price_df["open"], high=price_df["high"],
    low=price_df["low"], close=price_df["close"], name=picked_symbol,
))
for window in (20, 50):
    sma_series = sma(price_df["close"], window)
    fig.add_trace(go.Scatter(x=price_df["date"], y=sma_series, name=f"SMA{window}", line=dict(width=1)))

fig.add_hline(y=lv.entry_price, line_dash="dot", line_color="blue",
              annotation_text=f"進場 {lv.entry_price:,.4f}", annotation_position="right")
fig.add_hline(y=lv.stop_loss_price, line_dash="dot", line_color="red",
              annotation_text=f"停損 {lv.stop_loss_price:,.4f}", annotation_position="right")
fig.add_hline(y=lv.take_profit_price, line_dash="dot", line_color="green",
              annotation_text=f"停利 {lv.take_profit_price:,.4f}", annotation_position="right")

fig.update_layout(xaxis_rangeslider_visible=False, height=600, title=f"CRYPTO/{picked_symbol}")
st.plotly_chart(fig, width="stretch")

st.caption(
    f"停損 = 進場價 - 1.5×ATR({lv.atr_value:,.4f})；停利 = 進場價 × (1+{threshold:.0%})；"
    f"風險報酬比 = {lv.risk_reward_ratio:.2f}（賺的時候賺的比例 ÷ 虧的時候虧的比例）。"
)
