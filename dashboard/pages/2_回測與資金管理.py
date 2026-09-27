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
from backtest_report import SMALL_POSITION_WARNING_THRESHOLD

st.set_page_config(page_title="回測與資金管理", page_icon="💰", layout="wide")
st.title("💰 回測與資金管理")
st.caption(
    "對應 `py backtest_report.py`，同一套 Walk-forward 回測與凱利公式邏輯。"
    "勝率/賠率已扣除手續費與交易稅（比例式，見 backtest/engine.py），不是零成本的理論值；"
    "最大回撤/累積報酬已考慮「同一檔標的還沒平倉、新訊號就又觸發」的部位重疊限制，不是天真的全額複利假設。"
)
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
    max_concurrent_positions = st.number_input(
        "同一檔標的同時最多持有幾筆部位", min_value=1, max_value=10, value=1,
        help="預設 1 最保守：還沒平倉的話，新訊號一律跳過，不會同時開兩倉。horizon 越長，這個設定影響越大。",
    )

symbol_filter = st.multiselect("只看指定標的（留空 = 全部）", options=all_symbols)

if capital <= 0:
    st.info("請輸入總資金以計算建議倉位。")
    st.stop()

with st.spinner("回測中（第一次或參數變更時需要重新訓練模型，可能要等一下）..."):
    rows = cached_run_backtest_report(
        capital, horizon, threshold, prob_threshold, kelly_fraction, max_position_pct,
        tuple(symbol_filter) if symbol_filter else None,
        max_concurrent_positions=max_concurrent_positions,
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
        normal_df.rename(columns={
            "market": "市場", "symbol": "代號", "n_trades": "訊號數", "n_trades_taken": "已採用",
        })[
            ["市場", "代號", "訊號數", "勝率", "風險報酬比", "已採用", "最大回撤", "累積報酬", "現在機率", "AUC", "建議倉位"]
        ],
        width="stretch", hide_index=True,
    )
    st.caption("訊號數＝勝率/賠率的樣本數；已採用＝扣掉部位重疊限制後真的模擬進場的交易數，最大回撤/累積報酬是用這個算的。")

if not other_df.empty:
    st.subheader("無法產出完整回測")
    for row in other_df.itertuples():
        st.text(f"{row.market}/{row.symbol}: {row.status}")

if not normal_df.empty:
    total_skipped = int(normal_df["n_signals_skipped"].sum())
    if total_skipped:
        st.caption(
            f"另外有 {total_skipped} 次訊號因為當時已經持有部位、額度滿了而被跳過，"
            "沒有算進最大回撤/累積報酬——調整上面「同時最多持有幾筆部位」可以改變這個行為。"
        )

    small_positions = normal_df[
        (normal_df["suggested_position_value"] > 0)
        & (normal_df["suggested_position_value"] < SMALL_POSITION_WARNING_THRESHOLD)
    ]
    if not small_positions.empty:
        names = "、".join(f"{r.market}/{r.symbol}" for r in small_positions.itertuples())
        st.info(
            f"提醒：{names} 的建議倉位金額偏小（< {SMALL_POSITION_WARNING_THRESHOLD:,.0f}）。"
            "手續費模型只算比例式成本，沒算最低手續費下限——金額越小，最低手續費實際佔比就越高，"
            "這裡的勝率/賠率估計會偏樂觀，請自行對照你的券商規則。"
        )

st.caption(
    "提醒：以上勝率/賠率已扣除手續費與交易稅，來自歷史 Walk-forward 回測，不代表未來一定重演；"
    "「建議倉位」是套用凱利折扣係數與單筆上限後的參考值，不是投資建議。"
)
