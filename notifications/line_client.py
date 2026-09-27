"""
line_client.py
================
封裝 LINE Messaging API 的推播（push message），純技術層——不知道也不管
訊息內容是「訊號報告」還是「回測摘要」，那是 message_formatter.py 的責任。

設計原則（呼應 adapters/base_adapter.py 的既有慣例）：
- Channel Access Token / User ID 只能從 notifications/config.get_secret() 讀，
  不接受在程式碼裡硬編碼。
- 任何呼叫失敗都包在 try/except 裡並記錄詳細 log，回傳 False 而不是丟例外，
  讓 scheduler.py 的「單一環節失敗不中斷其他環節」設計可以正常運作
  ——LINE 推播失敗不該讓當天的資料更新排程被判定為失敗。
"""

from __future__ import annotations

import logging

from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    TextMessage,
)

from .config import get_secret

logger = logging.getLogger("notifications.line_client")

LINE_TEXT_MESSAGE_LIMIT = 5000  # LINE 官方文件規定單則文字訊息上限


def send_push_message(text: str) -> bool:
    """
    推播一則文字訊息給 config 裡設定的 LINE_USER_ID。
    回傳是否成功；失敗只記 log，不丟例外（呼叫端不需要包 try/except）。
    """
    token = get_secret("LINE_CHANNEL_ACCESS_TOKEN")
    user_id = get_secret("LINE_USER_ID")

    if not token or not user_id:
        logger.error(
            "缺少 LINE_CHANNEL_ACCESS_TOKEN 或 LINE_USER_ID，略過推播"
            "（請確認 config/.env 或 Streamlit secrets 已設定）"
        )
        return False

    if len(text) > LINE_TEXT_MESSAGE_LIMIT:
        logger.warning(f"訊息長度 {len(text)} 超過 LINE 上限 {LINE_TEXT_MESSAGE_LIMIT}，將被截斷")
        text = text[: LINE_TEXT_MESSAGE_LIMIT - 1] + "…"

    try:
        configuration = Configuration(access_token=token)
        with ApiClient(configuration) as api_client:
            MessagingApi(api_client).push_message(
                PushMessageRequest(to=user_id, messages=[TextMessage(text=text)])
            )
        logger.info("LINE 推播成功")
        return True
    except Exception:
        logger.exception("LINE 推播失敗")
        return False
