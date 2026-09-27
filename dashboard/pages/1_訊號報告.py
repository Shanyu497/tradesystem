"""
1_訊號報告.py
===============
對應 signal_report.py 的 Phase 3 主入口，呈現方式從終端機報表換成互動式表格，
邏輯（build_dataset -> train_and_evaluate -> predict_latest -> is_signal_reliable）
完全重用 signal_report.run_report()，沒有另外實作。
"""

from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import pandas as pd
import streamlit as st

from _cache import cached_load_watchlist, cached_run_report

st.set_page_config(page_title="訊號報告", page_icon="📡", layout="wide")
st.title("📡 訊號報告")
st.caption("對應 `py signal_report.py`，同一套訓練與可信度判斷邏輯。")

watchlist = cached_load_watchlist()
all_symbols = sorted({s for symbols in watchlist.values() for s in symbols})

col1, col2, col3 = st.columns(3)
with col1:
    horizon = st.number_input("預測未來幾天報酬率（horizon）", min_value=1, max_value=60, value=5)
with col2:
    threshold = st.number_input("漲幅門檻（threshold）", min_value=0.0, max_value=1.0, value=0.03, step=0.01, format="%.2f")
with col3:
    symbol_filter = st.multiselect("只看指定標的（留空 = 全部）", options=all_symbols)

with st.spinner("計算中（第一次或參數變更時需要重新訓練模型，可能要等一下）..."):
    rows = cached_run_report(horizon, threshold, tuple(symbol_filter) if symbol_filter else None)

if not rows:
    st.warning("監控清單是空的，請先確認 config/watchlist.json。")
    st.stop()

df = pd.DataFrame(rows)

reliable_df = df[(df["reliable"] == True) & df["probability"].notna()].sort_values("probability", ascending=False)  # noqa: E712
if not reliable_df.empty:
    best = reliable_df.iloc[0]
    st.success(f"可信且上漲機率最高：**{best['market']}/{best['symbol']}**（機率 {best['probability']:.1%}，AUC {best['auc']:.3f}）")
else:
    st.info("目前沒有任何標的的訊號被判定為可信（AUC 未達門檻），不建議依這份報告進場。")

misleading_df = df[(df["reliable"] != True) & df["probability"].notna() & (df["probability"] >= 0.5)]  # noqa: E712
if not misleading_df.empty:
    names = "、".join(f"{r.symbol}（{r.probability:.1%}）" for r in misleading_df.itertuples())
    st.warning(f"注意：{names} 機率數字不低，但模型可信度不足（AUC 跟隨機猜測沒有顯著差異），不該被當作訊號。")

display_df = df.copy()
display_df["reliable_sort"] = display_df["reliable"].fillna(False)
display_df["prob_sort"] = display_df["probability"].fillna(-1)
display_df = display_df.sort_values(["reliable_sort", "prob_sort"], ascending=[False, False])

display_df["可信度"] = display_df["reliable"].map({True: "可信", False: "不可信"}).fillna("-")
display_df["probability"] = display_df["probability"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
display_df["precision"] = display_df["precision"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
display_df["auc"] = display_df["auc"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")

st.dataframe(
    display_df.rename(columns={
        "market": "市場", "symbol": "代號", "train_samples": "訓練樣本",
        "probability": "上漲機率", "precision": "驗證精準率", "auc": "AUC", "status": "狀態",
    })[["市場", "代號", "訓練樣本", "上漲機率", "驗證精準率", "AUC", "可信度", "狀態"]],
    width="stretch", hide_index=True,
)

insufficient = df[df["train_samples"] < 100]
if not insufficient.empty:
    st.info(f"有 {len(insufficient)} 檔標的資料量不足，建議先執行 `py pipeline.py --days 730` 補齊歷史資料。")
