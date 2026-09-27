"""
config.py
==========
機密設定的統一讀取入口，讓 line_client.py 不用管「現在是本機執行還是
部署在 Streamlit Community Cloud」這兩種環境的差異。

讀取順序：
1. 環境變數（本機執行 scheduler.py / 手動測試時，靠 python-dotenv 把
   config/.env 載進環境變數；如果是用系統原生方式設定的環境變數也一樣吃得到）。
2. st.secrets（部署在 Streamlit Community Cloud 時，官方建議的機密管理方式，
   不會寫進檔案系統，來源是後台設定的 secrets.toml）。
   這裡用 lazy import，避免 scheduler.py 這種非 Streamlit 情境下還要安裝/初始化 Streamlit。

任何一個 key 兩邊都找不到，回傳 None，由呼叫端決定要怎麼處理
（line_client.py 會記錄錯誤但不中斷排程，呼應 base_adapter.py 的既有設計原則）。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / "config" / ".env")


def get_secret(key: str) -> str | None:
    value = os.environ.get(key)
    if value:
        return value

    try:
        import streamlit as st
        value = st.secrets.get(key)
        if value:
            return str(value)
    except Exception:
        # 沒有 streamlit、不在 streamlit 執行環境、或沒有設定 secrets.toml，
        # 都視為「這個管道沒有值」，不是需要中斷程式的錯誤。
        pass

    return None
