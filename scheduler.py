"""
scheduler.py
=============
常駐排程腳本：用 APScheduler 排定「台股收盤後」「美股收盤後」各自觸發一次
pipeline，抓取當天最新資料；資料更新完之後，緊接著各自再觸發一次訊號通知，
把當天的可信訊號 + 建議倉位整理成訊息推播到 LINE。

設計重點：
- 直接用時區參數（Asia/Taipei / America/New_York）指定 Cron 觸發時間，
  讓 APScheduler 自動處理夏令時間、時差換算，不用自己心算。
- 兩個市場的排程互相獨立：台股排程只處理 TW 標的，美股排程只處理 US 標的，
  避免其中一個市場的 API 或模型出問題時卡住另一個市場的排程。
- 每次觸發都抓「近 7 天」而不是只抓當天，用意是：
  如果某天因為電腦沒開機、網路斷線而錯過排程，
  下一次觸發時仍能把漏掉的那幾天補回來
  （儲存層本來就會用 date+symbol 去重，重複抓不會產生髒資料）。
- 訊號通知排在資料更新之後至少 30 分鐘觸發，確保 job_taiwan/job_us 已經
  把當天最新資料存好，訊號才會是根據最新資料算出來的。
- 加密貨幣沒有「收盤」這回事，24/7 都在交易，所以 job_crypto 用固定間隔
  （預設每 4 小時，用 UTC 時間，跟哪個時區都無關）觸發，而不是套用
  TW/US 那種「收盤後才抓」的邏輯。但訊號模型本身還是用日線特徵訓練的，
  一天通知太多次意義不大（訊號不會因為多抓幾次就變準），所以訊號通知
  還是維持一天一次，抓資料的頻率跟通知的頻率刻意分開。
- 訊號通知直接重用 signal_report.run_report() / backtest_report.run_backtest_report()
  （跟手動執行 `py signal_report.py` 是同一段程式碼），只是把輸出從印到終端機
  改成組成訊息推播到 LINE，避免邏輯分岔。
- job_crypto_screener 跟 job_notify_crypto 不一樣：後者只看 watchlist.json 裡固定的
  BTC/ETH；前者主動掃描 Binance 成交量前幾大的幣種（見 crypto_screener.py），
  找「還沒被加進 watchlist 的機會」，附上用 ATR/threshold 換算出來的進場/停損/停利價。
  一樣一天通知一次，不是每次資料更新都掃。
- 每次資料更新完，會呼叫 git_sync.sync_processed_data() 把 data/processed/ 的
  變更 commit + push 回這個 repo，讓部署在雲端（Streamlit Community Cloud）的
  Dashboard 能讀到最新資料。只有在這個專案有 git remote 時才會真的推送，
  純本機使用（沒 git init 或還沒設定 remote）完全不受影響。

執行方式：
    py scheduler.py
這是一個常駐程序，會一直執行、等待排程時間到了才觸發，
按 Ctrl+C 才會停止。建議透過「Windows 工作排程器」設定成開機時自動啟動
（觸發條件選「登入時」，動作是執行 py scheduler.py），
這樣就不用每次開機都手動執行，也不用真的做成 Windows 服務那麼複雜。

安裝需求：
    pip install -r requirements.txt
    （tzdata 是必要的：Windows 系統本身沒有內建 IANA 時區資料庫，
      沒裝這個套件，時區換算會直接報錯。）
    LINE 推播需要先設定 config/.env（見 config/.env.example），
    沒有設定的話訊號通知會記錄錯誤但不會讓資料更新排程失敗。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from backtest_report import run_backtest_report
from crypto_screener import run_screener as run_crypto_screener
from git_sync import sync_processed_data
from notifications.config import get_secret
from notifications.line_client import send_push_message
from notifications.message_formatter import (
    format_position_summary,
    format_screener_summary,
    format_signal_summary,
)
from pipeline import load_watchlist, print_summary, run_pipeline
from signal_report import run_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scheduler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("scheduler")

# 每次觸發回補的天數，抵禦「錯過一次排程」的狀況
LOOKBACK_DAYS = 7

# 訊號通知用的參數，跟 signal_report.py / backtest_report.py 的 CLI 預設值一致，
# 想調整口味（例如拉高機率門檻）直接改這裡即可，不影響手動執行 CLI 時的行為。
NOTIFY_HORIZON = 5
NOTIFY_THRESHOLD = 0.03
NOTIFY_PROB_THRESHOLD = 0.5
NOTIFY_KELLY_FRACTION = 0.5
NOTIFY_MAX_POSITION_PCT = 0.25

# 加密貨幣掃描用的參數，跟 crypto_screener.py 的 CLI 預設值一致
SCREENER_TOP_N = 30
SCREENER_MIN_PROBABILITY = 0.5


def _date_range() -> tuple[str, str]:
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    return start_date, end_date


def job_taiwan() -> None:
    logger.info("=== [排程觸發] 台股資料更新開始 ===")
    start_date, end_date = _date_range()
    try:
        results = run_pipeline(start_date, end_date, markets={"TW"})
        print_summary(results, start_date, end_date)
    except Exception:
        logger.exception("台股排程執行失敗")
    sync_processed_data()  # 有設定 git remote（部署雲端）才會真的推送，否則安全略過


def job_us() -> None:
    logger.info("=== [排程觸發] 美股資料更新開始 ===")
    start_date, end_date = _date_range()
    try:
        results = run_pipeline(start_date, end_date, markets={"US"})
        print_summary(results, start_date, end_date)
    except Exception:
        logger.exception("美股排程執行失敗")
    sync_processed_data()  # 有設定 git remote（部署雲端）才會真的推送，否則安全略過


def job_crypto() -> None:
    logger.info("=== [排程觸發] 加密貨幣資料更新開始 ===")
    start_date, end_date = _date_range()
    try:
        results = run_pipeline(start_date, end_date, markets={"CRYPTO"})
        print_summary(results, start_date, end_date)
    except Exception:
        logger.exception("加密貨幣排程執行失敗")
    sync_processed_data()  # 有設定 git remote（部署雲端）才會真的推送，否則安全略過


def job_notify(market: str) -> None:
    """
    產出當天訊號報告 + 建議倉位（只看 market 這個市場的標的），組成訊息推播到 LINE。
    任何一步失敗都只記 log，不丟例外，避免排程本身被判定失敗（呼應 job_taiwan/job_us 的做法）。
    """
    logger.info(f"=== [排程觸發] {market} 訊號通知開始 ===")
    try:
        watchlist = load_watchlist()
        symbols = set(watchlist.get(market, []))
        if not symbols:
            logger.warning(f"監控清單裡沒有 {market} 的標的，略過通知")
            return

        signal_rows = run_report(
            horizon=NOTIFY_HORIZON, threshold=NOTIFY_THRESHOLD, symbol_filter=symbols,
        )
        message = format_signal_summary(signal_rows)

        capital = get_secret("TRADING_CAPITAL")
        if capital:
            backtest_rows = run_backtest_report(
                capital=float(capital),
                horizon=NOTIFY_HORIZON,
                threshold=NOTIFY_THRESHOLD,
                prob_threshold=NOTIFY_PROB_THRESHOLD,
                kelly_fraction=NOTIFY_KELLY_FRACTION,
                max_position_pct=NOTIFY_MAX_POSITION_PCT,
                symbol_filter=symbols,
            )
            message += "\n\n" + format_position_summary(backtest_rows)
        else:
            logger.warning("未設定 TRADING_CAPITAL，通知只會包含訊號報告、不含建議倉位")

        send_push_message(message)
    except Exception:
        logger.exception(f"{market} 訊號通知執行失敗")


def job_notify_taiwan() -> None:
    job_notify("TW")


def job_notify_us() -> None:
    job_notify("US")


def job_notify_crypto() -> None:
    job_notify("CRYPTO")


def job_crypto_screener() -> None:
    """
    掃描 Binance 成交量前幾大的幣種（不限於 watchlist.json 裡固定的 BTC/ETH），
    找出可信且機率夠高的短期上漲候選，附上進場/停損/停利價，推播到 LINE。
    跟 job_notify_crypto 是兩件不同的事：job_notify_crypto 只看 watchlist 裡
    固定的幾檔；這個 job 主動找「還沒被加進 watchlist 的機會」。
    """
    logger.info("=== [排程觸發] 加密貨幣掃描開始 ===")
    try:
        rows = run_crypto_screener(
            top_n=SCREENER_TOP_N, horizon=NOTIFY_HORIZON, threshold=NOTIFY_THRESHOLD,
        )
        message = format_screener_summary(rows, min_probability=SCREENER_MIN_PROBABILITY)
        send_push_message(message)
    except Exception:
        logger.exception("加密貨幣掃描執行失敗")


def main() -> None:
    scheduler = BlockingScheduler()

    # 台股收盤時間是 13:30，14:00 觸發確保收盤資料已經整理好
    tw_trigger = CronTrigger(day_of_week="mon-fri", hour=14, minute=0, timezone="Asia/Taipei")
    scheduler.add_job(
        job_taiwan,
        trigger=tw_trigger,
        id="taiwan_daily_update",
        name="台股每日資料更新",
    )

    # 美股收盤時間是美東 16:00，16:30 觸發確保收盤資料已經整理好
    # 用 America/New_York 時區，APScheduler 會自動處理夏令時間切換
    us_trigger = CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone="America/New_York")
    scheduler.add_job(
        job_us,
        trigger=us_trigger,
        id="us_daily_update",
        name="美股每日資料更新",
    )

    # 訊號通知排在對應市場的資料更新之後 30 分鐘觸發，確保吃到當天最新資料
    tw_notify_trigger = CronTrigger(day_of_week="mon-fri", hour=14, minute=30, timezone="Asia/Taipei")
    scheduler.add_job(
        job_notify_taiwan,
        trigger=tw_notify_trigger,
        id="taiwan_daily_notify",
        name="台股每日訊號通知",
    )

    us_notify_trigger = CronTrigger(day_of_week="mon-fri", hour=17, minute=0, timezone="America/New_York")
    scheduler.add_job(
        job_notify_us,
        trigger=us_notify_trigger,
        id="us_daily_notify",
        name="美股每日訊號通知",
    )

    # 加密貨幣 24/7 交易，沒有「收盤後」的概念，用固定 UTC 時間點每 4 小時抓一次，
    # 讓「今天還在形成中」的日 K 棒盡量新鮮（模型訓練用的還是日線特徵，抓更頻繁
    # 不會讓訊號更準，純粹是讓最新那根 K 棒的價格更即時）。
    crypto_trigger = CronTrigger(hour="0,4,8,12,16,20", minute=5, timezone="UTC")
    scheduler.add_job(
        job_crypto,
        trigger=crypto_trigger,
        id="crypto_4h_update",
        name="加密貨幣資料更新（每4小時）",
    )

    # 訊號通知一天一次就好（一天抓好幾次資料，但底層模型是日線特徵，
    # 通知抓得再頻繁訊號也不會變，一天多通知只會變成騷擾），排在 UTC 00:05
    # 那次資料更新之後，抓到的是「剛結束的那個 UTC 日」完整 K 棒。
    crypto_notify_trigger = CronTrigger(hour=0, minute=30, timezone="UTC")
    scheduler.add_job(
        job_notify_crypto,
        trigger=crypto_notify_trigger,
        id="crypto_daily_notify",
        name="加密貨幣每日訊號通知",
    )

    # 掃描排在訊號通知之後 10 分鐘，避開同時大量訓練模型搶 CPU
    crypto_screener_trigger = CronTrigger(hour=0, minute=40, timezone="UTC")
    scheduler.add_job(
        job_crypto_screener,
        trigger=crypto_screener_trigger,
        id="crypto_daily_screener",
        name="加密貨幣每日掃描",
    )

    logger.info("排程已啟動，等待觸發時間到達（Ctrl+C 結束）...")
    # 直接問 trigger 本身下一次觸發時間，不依賴 job.next_run_time
    # （job.next_run_time 要等 scheduler.start() 真正跑起來才會被賦值，
    #  在 start() 之前存取一定是沒有這個屬性，這是 APScheduler 的正常機制）
    scheduled = [
        ("台股每日資料更新", tw_trigger),
        ("美股每日資料更新", us_trigger),
        ("台股每日訊號通知", tw_notify_trigger),
        ("美股每日訊號通知", us_notify_trigger),
        ("加密貨幣資料更新（每4小時）", crypto_trigger),
        ("加密貨幣每日訊號通知", crypto_notify_trigger),
        ("加密貨幣每日掃描", crypto_screener_trigger),
    ]
    for name, trigger in scheduled:
        next_fire = trigger.get_next_fire_time(None, datetime.now(trigger.timezone))
        logger.info(f"已排定任務: {name}（下一次觸發: {next_fire}）")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("收到停止訊號，排程已結束")


if __name__ == "__main__":
    main()
