"""
pipeline.py
============
Phase 2 收尾：把「讀取監控清單 -> 呼叫對應市場的 Adapter -> 存檔 -> 產出執行摘要」
串成一條完整的資料獲取 pipeline。

用法：
    py pipeline.py                       # 抓 config/watchlist.json 裡全部標的，預設近 90 天
    py pipeline.py --days 365            # 抓近 365 天
    py pipeline.py --start 2024-01-01    # 指定起始日期（會覆蓋 --days）
    py pipeline.py --symbols 2330,AAPL   # 只抓指定的幾檔（不分市場，會自動比對清單所屬市場）

之後接上 APScheduler 或 Windows 工作排程器時，只需要排程呼叫這支腳本即可，
不需要額外的邏輯——這是刻意的設計，讓「手動跑一次」和「排程自動跑」用的是同一段程式碼。
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from adapters.base_adapter import StockDataAdapter
from adapters.crypto_adapter import CryptoAdapter
from adapters.taiwan_adapter import TaiwanStockAdapter
from adapters.us_adapter import USStockAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent
WATCHLIST_PATH = PROJECT_ROOT / "config" / "watchlist.json"


@dataclass
class SymbolResult:
    """單一標的的執行結果，用來組成最後的摘要報告"""
    market: str
    symbol: str
    price_rows: int = 0
    fundamental_rows: int = 0
    chip_rows: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0 and self.price_rows > 0


def load_watchlist(path: Path = WATCHLIST_PATH) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到監控清單設定檔: {path}\n"
            f"請確認 config/watchlist.json 存在，格式例如："
            f'{{"TW": ["2330"], "US": ["AAPL"]}}'
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_symbol(adapter: StockDataAdapter, market: str, symbol: str,
                start_date: str, end_date: str) -> SymbolResult:
    """對單一標的依序呼叫價量/基本面/籌碼面，任何一段失敗都不中斷其他段落"""
    result = SymbolResult(market=market, symbol=symbol)

    try:
        price_df = adapter.fetch_price(symbol, start_date, end_date)
        result.price_rows = len(price_df)
        if not price_df.empty:
            adapter.save_processed(price_df, "price", symbol)
        else:
            result.errors.append("價量資料為空")
    except Exception as e:
        result.errors.append(f"價量抓取失敗: {e}")
        logger.exception(f"[{market}/{symbol}] 價量抓取發生例外")

    try:
        fundamental_df = adapter.fetch_fundamental(symbol, start_date, end_date)
        result.fundamental_rows = len(fundamental_df)
        if not fundamental_df.empty:
            adapter.save_processed(fundamental_df, "fundamental", symbol)
    except Exception as e:
        # 基本面資料非核心（例如美股沒設 Finnhub key 時本來就會是空的），
        # 記錄下來但不影響整體判定是否成功
        logger.warning(f"[{market}/{symbol}] 基本面抓取失敗（不影響核心流程）: {e}")

    try:
        chip_df = adapter.fetch_chip(symbol, start_date, end_date)
        result.chip_rows = len(chip_df)
        if not chip_df.empty:
            adapter.save_processed(chip_df, "chip", symbol)
    except Exception as e:
        logger.warning(f"[{market}/{symbol}] 籌碼面抓取失敗（不影響核心流程）: {e}")

    return result


def run_pipeline(start_date: str, end_date: str,
                  symbol_filter: set[str] | None = None,
                  markets: set[str] | None = None) -> list[SymbolResult]:
    """
    markets: 只跑指定的市場（例如排程腳本要「只跑 TW」或「只跑 US」時使用）。
             None 代表跑監控清單裡的所有市場。
    """
    watchlist = load_watchlist()
    results: list[SymbolResult] = []

    adapters: dict[str, StockDataAdapter] = {
        "TW": TaiwanStockAdapter(),
        "US": USStockAdapter(),
        "CRYPTO": CryptoAdapter(),
    }

    for market, symbols in watchlist.items():
        if markets and market not in markets:
            continue
        adapter = adapters.get(market)
        if adapter is None:
            logger.warning(f"未知市場代碼 '{market}'，略過")
            continue

        for symbol in symbols:
            if symbol_filter and symbol not in symbol_filter:
                continue
            logger.info(f"開始處理 [{market}/{symbol}] ...")
            result = run_symbol(adapter, market, symbol, start_date, end_date)
            results.append(result)

    return results


def print_summary(results: list[SymbolResult], start_date: str, end_date: str) -> None:
    print("\n" + "=" * 60)
    print(f"執行摘要（資料區間 {start_date} ~ {end_date}）")
    print("=" * 60)

    success = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print(f"\n總計 {len(results)} 檔標的，成功 {len(success)} 檔，失敗 {len(failed)} 檔\n")

    print(f"{'市場':<8}{'代號':<8}{'價量筆數':<10}{'基本面筆數':<12}{'籌碼面筆數':<12}{'狀態'}")
    print("-" * 60)
    for r in results:
        status = "成功" if r.ok else f"失敗（{'; '.join(r.errors)}）"
        print(f"{r.market:<8}{r.symbol:<8}{r.price_rows:<10}{r.fundamental_rows:<12}{r.chip_rows:<12}{status}")

    if failed:
        print(f"\n有 {len(failed)} 檔標的失敗，建議檢查對應的 API 額度或網路連線。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="每日美股/台股資料獲取 pipeline")
    parser.add_argument("--days", type=int, default=90, help="抓取近 N 天的資料（預設 90 天）")
    parser.add_argument("--start", type=str, default=None, help="指定起始日期 YYYY-MM-DD（會覆蓋 --days）")
    parser.add_argument("--end", type=str, default=None, help="指定結束日期 YYYY-MM-DD（預設今天）")
    parser.add_argument("--symbols", type=str, default=None,
                         help="只抓指定標的，逗號分隔，例如 2330,AAPL")
    parser.add_argument("--markets", type=str, default=None,
                         help="只抓指定市場，逗號分隔，例如 TW 或 US 或 CRYPTO 或 TW,US")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    end_date = args.end or datetime.today().strftime("%Y-%m-%d")
    if args.start:
        start_date = args.start
    else:
        start_date = (datetime.today() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    symbol_filter = set(args.symbols.split(",")) if args.symbols else None
    markets = set(args.markets.split(",")) if args.markets else None

    logger.info(f"Pipeline 啟動，資料區間：{start_date} ~ {end_date}")
    results = run_pipeline(start_date, end_date, symbol_filter, markets)
    print_summary(results, start_date, end_date)


if __name__ == "__main__":
    main()
