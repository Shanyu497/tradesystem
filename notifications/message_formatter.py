"""
message_formatter.py
======================
把 signal_report.run_report() / backtest_report.run_backtest_report() 回傳的
list[dict]（跟終端機報表用的是同一份資料）轉成適合塞進 LINE 文字訊息的精簡摘要。

刻意只挑「可信且值得注意」的標的，不是整份報表都塞進去：
- LINE 文字訊息有長度上限（見 line_client.LINE_TEXT_MESSAGE_LIMIT）。
- 手機通知本來就該是「你需要馬上看到的重點」，完整報表留給 Dashboard。

這一層完全不依賴 LINE SDK，純字串組裝，方便之後 Dashboard 或未來的
Webhook 互動查詢直接重用同一套格式化邏輯。
"""

from __future__ import annotations

from datetime import date


def format_signal_summary(rows: list[dict], top_n: int = 5) -> str:
    """對應 signal_report.run_report() 的輸出，摘要「可信且上漲機率高」的標的"""
    reliable = [r for r in rows if r.get("reliable") and r.get("probability") is not None]
    reliable.sort(key=lambda r: r["probability"], reverse=True)

    lines = [f"【訊號報告 {date.today().isoformat()}】"]

    if not reliable:
        lines.append("今日沒有可信訊號（AUC 未達門檻或資料不足），暫無標的值得留意。")
        return "\n".join(lines)

    for r in reliable[:top_n]:
        lines.append(
            f"{r['market']}/{r['symbol']}：上漲機率 {r['probability']:.1%}"
            f"（AUC {r['auc']:.3f}）"
        )

    if len(reliable) > top_n:
        lines.append(f"...另有 {len(reliable) - top_n} 檔可信訊號，詳見 Dashboard。")

    return "\n".join(lines)


def format_position_summary(rows: list[dict]) -> str:
    """對應 backtest_report.run_backtest_report() 的輸出，摘要有建議倉位的標的"""
    actionable = [r for r in rows if r.get("status") == "正常" and r.get("suggested_position_value", 0) > 0]
    actionable.sort(key=lambda r: r["suggested_position_value"], reverse=True)

    lines = [f"【建議倉位 {date.today().isoformat()}】"]

    if not actionable:
        lines.append("今日沒有標的觸發建議倉位（機率未達門檻或訊號不可信）。")
        return "\n".join(lines)

    for r in actionable:
        lines.append(
            f"{r['market']}/{r['symbol']}：{r['suggested_position_pct']:.1%}"
            f"（約 {r['suggested_position_value']:,.0f}）"
            f"｜勝率 {r['win_rate']:.1%} 賠率 {r['payoff_ratio']:.2f}"
        )

    lines.append("\n提醒：勝率/賠率已扣手續費與交易稅，為歷史回測換算的參考值，不是投資建議。")
    return "\n".join(lines)
