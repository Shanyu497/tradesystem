"""
crypto_screener.py
=====================
掃描 Binance 成交量前 N 大的加密貨幣，找出「模型判斷可信、且上漲機率最高」的標的，
再用 risk/trade_levels.py 算出進場價/停損價/停利價。

跟 signal_report.py 的差別：signal_report.py 只看 config/watchlist.json 裡
固定的幾檔標的；這裡動態掃描 Binance 當下成交量最大的幣，不受 watchlist 限制，
用的是同一套訓練/可信度判斷邏輯，沒有另外發明一套演算法。掃描到的價量資料
只留在記憶體裡算完就丟，不會寫進 data/processed（那個目錄是給固定監控清單用的，
掃描候選幣一天可能不一樣，沒必要每天塞新的檔案進去、也沒必要進 git）。

誠實地說在前面：這個系統對加密貨幣的預測力目前普遍偏弱（AUC 大多在 0.5~0.6，
只比隨機猜好一點——實測過 BTC/ETH，拉長 horizon 或調整 threshold 都沒有
根本改善）。掃更多幣只是增加「剛好有一檔可信」的機會，不代表掃出來的標的
真的「極有可能上漲」。這裡只會老實回報「掃描範圍內最可信、機率最高的是誰」，
如果沒有任何一檔跨過可信度門檻，就直接說「今天沒有可信的機會」，不會硬湊一個
出來騙自己。

用法：
    py crypto_screener.py                    # 掃 Binance 前 30 大成交量幣種
    py crypto_screener.py --top-n 50          # 掃前 50 大
    py crypto_screener.py --horizon 10 --threshold 0.05
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd

from adapters.crypto_adapter import CryptoAdapter, list_top_symbols
from risk.trade_levels import suggest_trade_levels
from signals.dataset import build_dataset
from signals.model import is_signal_reliable, predict_latest, train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("crypto_screener")

MIN_TRAIN_SAMPLES = 100
DEFAULT_LOOKBACK_DAYS = 730


def run_screener(top_n: int = 30, horizon: int = 5, threshold: float = 0.03,
                   lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> list[dict]:
    symbols = list_top_symbols(top_n)
    if not symbols:
        logger.error("取得候選幣清單失敗（Binance API 可能暫時不可用），略過本次掃描")
        return []

    adapter = CryptoAdapter()
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    rows: list[dict] = []
    for symbol in symbols:
        try:
            price_df = adapter.fetch_price(symbol, start_date, end_date)
        except Exception as e:
            logger.warning(f"[{symbol}] 抓取價量失敗，略過: {e}")
            continue

        if price_df.empty:
            continue

        train_df, live_df = build_dataset(price_df, horizon=horizon, threshold=threshold)
        if len(train_df) < MIN_TRAIN_SAMPLES:
            continue  # 資料不夠的幣（太新上市）直接跳過，掃描報告只列有意義的結果

        model, cv_metrics = train_and_evaluate(train_df)
        latest = predict_latest(model, live_df)
        if latest.empty:
            continue

        current_prob = float(latest.iloc[-1]["probability"])
        reliable = is_signal_reliable(cv_metrics)
        levels = suggest_trade_levels(price_df, threshold=threshold)

        rows.append({
            "symbol": symbol,
            "probability": current_prob,
            "reliable": reliable,
            "auc": cv_metrics.avg_auc if cv_metrics else None,
            "train_samples": len(train_df),
            "levels": levels,
        })

    return rows


def select_candidates(rows: list[dict], min_probability: float = 0.5) -> list[dict]:
    """
    篩出「真正值得看」的標的：AUC 可信只代表模型排序能力不是隨機亂猜，
    不代表機率本身夠高——一檔幣機率 9.7% 也可能是「掃描範圍內最可信的」，
    但 9.7% 代表模型認為「不太會漲」，不是進場訊號。這裡額外要求機率本身
    也要過 min_probability 門檻，兩個條件都過才算數，避免把「矮子裡拔將軍」
    誤認成「找到好機會」。
    """
    candidates = [
        r for r in rows
        if r["reliable"] and r["levels"] is not None and r["probability"] >= min_probability
    ]
    candidates.sort(key=lambda r: r["probability"], reverse=True)
    return candidates


def print_screener_report(rows: list[dict], horizon: int, threshold: float, min_probability: float = 0.5) -> None:
    print("\n" + "=" * 90)
    print(f"加密貨幣掃描報告（未來 {horizon} 天漲幅 > {threshold:.1%}，掃描 {len(rows)} 檔有足夠資料的幣）")
    print("=" * 90)

    reliable_rows = select_candidates(rows, min_probability)

    if not reliable_rows:
        print(f"\n今天掃描範圍內沒有任何一檔幣「模型可信」且「上漲機率 >= {min_probability:.0%}」，沒有建議進場的標的。")
        print("（這是誠實的結果，不代表系統故障——加密貨幣短期技術分析的預測力本來就有限，")
        print("  「可信」只代表模型排序能力不是亂猜，不代表機率夠高，兩者都要過門檻才列出來）")
    else:
        print(f"\n{'代號':<10}{'上漲機率':<10}{'AUC':<8}{'進場價':<14}{'停損價':<14}{'停利價':<14}{'風險報酬比'}")
        print("-" * 90)
        for r in reliable_rows[:10]:
            lv = r["levels"]
            print(f"{r['symbol']:<10}{r['probability']:<10.1%}{r['auc']:<8.3f}"
                  f"{lv.entry_price:<14,.4f}{lv.stop_loss_price:<14,.4f}{lv.take_profit_price:<14,.4f}{lv.risk_reward_ratio:.2f}")

        best = reliable_rows[0]
        print(f"\n目前最可信且機率最高：{best['symbol']}（機率 {best['probability']:.1%}，AUC {best['auc']:.3f}）")

    print("\n提醒：進場價=現在收盤價，停損價=進場價 - 1.5倍ATR，停利價=進場價×(1+threshold)，")
    print("      是把模型/波動度換算成具體價位的參考值，不是精準預測，不是投資建議。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="加密貨幣掃描報告：找出可信度最高的短期上漲機會")
    parser.add_argument("--top-n", type=int, default=30, help="掃描 Binance 成交量前幾大的幣種（預設 30）")
    parser.add_argument("--horizon", type=int, default=5, help="預測未來幾天的報酬率（預設 5）")
    parser.add_argument("--threshold", type=float, default=0.03, help="漲幅門檻，例如 0.03 代表 3%%（預設 0.03）")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="訓練資料回溯天數（預設 730）")
    parser.add_argument("--min-probability", type=float, default=0.5, help="上漲機率至少要達到多少才列入建議（預設 0.5）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_screener(args.top_n, args.horizon, args.threshold, args.lookback_days)
    print_screener_report(rows, args.horizon, args.threshold, args.min_probability)


if __name__ == "__main__":
    main()
