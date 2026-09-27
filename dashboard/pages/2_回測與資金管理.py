"""
2_回測與資金管理.py
=====================
對應 backtest_report.py 的 Phase 4 主入口：Walk-forward 回測 + 凱利公式建議倉位，
邏輯完全重用 backtest_report.run_backtest_report()，沒有另外實作。
"""

from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import pandas as pd
import streamlit as st

from _cache import cached_load_watchlist, cached_run_backtest_report

st.set_page_config(page_title="回測與資金管理", page_icon="💰", layout="wide")
st.title("💰 回測與資金管理")
st.caption("對應 `py backtest_report.py`，同一套 Walk-forward 回測與凱利公式邏輯。")
st.warning("這不是財務建議，凱利公式建議的倉位完全建立在「回測算出的勝率/賠率在未來會持續成立」這個假設上。")

watchlist = cached_load_watchlist()
all_symbols = sorted({s for symbols in watchlist.values() for s in symbols})

col1, col2 = st.columns(2)
with col1:
    capital = st.number_input("總資金", min_value=0.0, value=500_000.0, step=10_000.0, format="%.0f")
    horizon = st.number_input("預測未來幾天報酬率（horizon）", min_value=1, max_value=60, value=5)
    threshold = st.number_input("分類門檻（threshold）", min_value=0.0, max_value=1.0, value=0.03, step=0.01, format="%.2f")
with col2:
    prob_threshold = st.number_input("進場機率門檻", min_value=0.0, max_value=1.0, value=0.5, step=0.05, format="%.2f")
    kelly_fraction = st.number_input("凱利打折係數（0.5=半凱利）", min_value=0.0, max_value=1.0, value=0.5, step=0.05, format="%.2f")
    max_position_pct = st.number_input("單筆倉位上限（佔總資金）", min_value=0.0, max_value=1.0, value=0.25, step=0.05, format="%.2f")

symbol_filter = st.multiselect("只看指定標的（留空 = 全部）", options=all_symbols)

if capital <= 0:
    st.info("請輸入總資金以計算建議倉位。")
    st.stop()

with st.spinner("回測中（第一次或參數變更時需要重新訓練模型，可能要等一下）..."):
    rows = cached_run_backtest_report(
        capital, horizon, threshold, prob_threshold, kelly_fraction, max_position_pct,
        tuple(symbol_filter) if symbol_filter else None,
    )

if not rows:
    st.warning("監控清單是空的，請先確認 config/watchlist.json。")
    st.stop()

df = pd.DataFrame(rows)
normal_df = df[df["status"] == "正常"].copy()
other_df = df[df["status"] != "正常"]

if not normal_df.empty:
    normal_df = normal_df.sort_values("suggested_position_value", ascending=False)

    def _position_label(row: pd.Series) -> str:
        if row["suggested_position_value"] > 0:
            return f"{row['suggested_position_pct']:.1%}（約 {row['suggested_position_value']:,.0f}）"
        if row["current_prob"] is not None and row["current_prob"] >= prob_threshold and not row["reliable"]:
            return "不建議進場（訊號不可信）"
        return "不建議進場（機率不足）"

    normal_df["建議倉位"] = normal_df.apply(_position_label, axis=1)
    normal_df["現在機率"] = normal_df["current_prob"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    normal_df["勝率"] = normal_df["win_rate"].map(lambda x: f"{x:.1%}")
    normal_df["風險報酬比"] = normal_df["payoff_ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    normal_df["最大回撤"] = normal_df["max_drawdown"].map(lambda x: f"{x:.1%}")
    normal_df["累積報酬"] = normal_df["total_return"].map(lambda x: f"{x:.1%}")
    normal_df["AUC"] = normal_df["auc"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")

    st.dataframe(
        normal_df.rename(columns={"market": "市場", "symbol": "代號", "n_trades": "交易數"})[
            ["市場", "代號", "交易數", "勝率", "風險報酬比", "最大回撤", "累積報酬", "現在機率", "AUC", "建議倉位"]
        ],
        use_container_width=True, hide_index=True,
    )

if not other_df.empty:
    st.subheader("無法產出完整回測")
    for row in other_df.itertuples():
        st.text(f"{row.market}/{row.symbol}: {row.status}")

st.caption(
    "提醒：以上勝率/賠率來自歷史 Walk-forward 回測，不代表未來一定重演；"
    "「建議倉位」是套用凱利折扣係數與單筆上限後的參考值，不是投資建議。"
)
