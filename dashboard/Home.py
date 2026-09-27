"""
Home.py
========
Phase 5 Dashboard 入口：系統總覽——監控清單有哪些標的、本機資料有多新，
方便一眼看出「今天要不要先跑 pipeline.py 補資料」，再到左側其他頁面看訊號/回測。

跟 signal_report.py / backtest_report.py 一樣，這裡不重寫任何分析邏輯，
只讀 data/processed 底下的檔案時間戳記來判斷資料新鮮度。
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import streamlit as st

from _cache import PROJECT_ROOT, cached_load_watchlist, clear_all_caches

st.set_page_config(page_title="量化交易 Dashboard", page_icon="📈", layout="wide")


def _check_password() -> bool:
    """部署到雲端且有設定 DASHBOARD_PASSWORD 時要求輸入密碼；本機開發沒設定就直接放行。"""
    # _cache 模組已經把 PROJECT_ROOT 加進 sys.path，這裡可以直接 import 專案模組
    from notifications.config import get_secret

    expected = get_secret("DASHBOARD_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("authenticated"):
        return True

    entered = st.text_input("請輸入密碼", type="password")
    if entered == expected:
        st.session_state["authenticated"] = True
        st.rerun()
    elif entered:
        st.error("密碼錯誤")
    return False


if not _check_password():
    st.stop()

st.title("📈 量化交易系統 Dashboard")
st.caption("Phase 1~4 分析結果的視覺化入口，不重複任何分析邏輯，資料來源跟終端機報表完全一致。")

if st.button("🔄 強制重新整理（清除快取，下次查看時重新訓練模型）"):
    clear_all_caches()
    st.success("快取已清除。")

watchlist = cached_load_watchlist()

st.subheader("監控清單與資料新鮮度")
for market, symbols in watchlist.items():
    st.markdown(f"**{market}**")
    rows = []
    for symbol in symbols:
        path = PROJECT_ROOT / "data" / "processed" / market / "price" / f"{symbol}.parquet"
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            rows.append({"標的": symbol, "資料狀態": "已有資料", "最後更新": mtime.strftime("%Y-%m-%d %H:%M")})
        else:
            rows.append({"標的": symbol, "資料狀態": "尚無資料", "最後更新": "-"})
    st.dataframe(rows, use_container_width=True, hide_index=True)

st.info(
    "尚無資料的標的，先在終端機執行 `py pipeline.py --days 730` 補齊歷史資料，"
    "再到左側「訊號報告」「回測與資金管理」頁面查看分析結果。"
)
