"""
crypto_adapter.py
===================
加密貨幣資料 Adapter。

資料源：
    Binance 公開 REST API（/api/v3/klines）——不需要 API key、不需要註冊，
    公開市場資料端點的免費額度很寬鬆（預設帳號額度 1200 weight/分鐘，
    klines 每次請求只算 1~2 weight），對這個專案的用量完全夠用。
    選它而不是 CoinGecko，是因為 klines 直接就是我們要的 OHLCV 格式、
    可以指定任意時間區間和 K 棒粒度，不用像 CoinGecko 免費版那樣
    被固定在幾種區間/粒度組合裡。

代號對應：
    watchlist.json 裡沿用 yfinance 的習慣格式「BTC-USD」「ETH-USD」，
    這裡會轉換成 Binance 的交易對格式「BTCUSDT」「ETHUSDT」
    （Binance 沒有真正的 USD 交易對，USDT 是掛鉤美元的穩定幣，
    價格對這個系統的技術指標/報酬率計算來說跟 USD 差異可忽略）。

沒有基本面/籌碼面資料：
    加密貨幣沒有「營收、EPS、本益比」這種傳統基本面概念，也沒有三大法人/
    融資融券這種籌碼面資料，這兩個方法固定回傳空 DataFrame，
    保持跟 TW/US Adapter 一樣的介面。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from .base_adapter import (
    CHIP_SCHEMA,
    FUNDAMENTAL_SCHEMA,
    PRICE_SCHEMA,
    AdapterConfig,
    StockDataAdapter,
)

BINANCE_BASE_URL = "https://api.binance.com/api/v3"
KLINES_LIMIT = 1000  # Binance 單次請求最多回傳的 K 棒數


def _to_binance_symbol(symbol: str) -> str:
    """watchlist 格式「BTC-USD」-> Binance 交易對格式「BTCUSDT」"""
    return symbol.replace("-USD", "USDT")


def _from_binance_symbol(binance_symbol: str) -> str:
    """Binance 交易對格式「BTCUSDT」-> watchlist 格式「BTC-USD」"""
    return binance_symbol[:-4] + "-USD"  # 固定切掉結尾的 "USDT"


# 篩選候選幣種時要排除的雜訊：槓桿代幣（3x/UP/DOWN/BULL/BEAR 這種，波動被人為放大，
# 技術指標對它們沒有意義）、穩定幣互轉對（USDC/USDT 這種「漲跌」只是脫錨雜訊，不是真訊號）。
_LEVERAGED_TOKEN_MARKERS = ("UP", "DOWN", "BULL", "BEAR")
_STABLECOIN_BASES = {"DAI", "EUR", "GBP", "EURI", "FDUSD", "BUSD", "TUSD", "USDP", "PYUSD", "GUSD"}


def list_top_symbols(top_n: int = 30) -> list[str]:
    """
    回傳 Binance 上「24 小時成交金額」最高的前 top_n 個 USDT 交易對，
    轉換成 watchlist 格式（「BTC-USD」）。用成交金額排序是業界常見的
    「篩掉冷門/沒有流動性代幣」做法——流動性太差的幣，技術指標本來就容易失真，
    價格也可能被少數幾筆掛單就大幅拉動，不適合拿來做技術分析。

    這不是「挑選出會漲的幣」，只是「決定要掃描哪些候選幣」——
    真正判斷「值不值得進場」的邏輯在 signals/model.py 的可信度判斷那一層。
    """
    try:
        resp = requests.get(f"{BINANCE_BASE_URL}/ticker/24hr", timeout=15)
        resp.raise_for_status()
        tickers = resp.json()
    except Exception:
        logging.getLogger("CryptoAdapter").exception("Binance 24hr ticker 呼叫失敗，無法取得候選幣清單")
        return []

    candidates = []
    for t in tickers:
        symbol = t.get("symbol", "")
        if not symbol.endswith("USDT"):
            continue
        base = symbol[:-4]
        if base in _STABLECOIN_BASES or base.startswith("USD"):
            continue
        if any(marker in base for marker in _LEVERAGED_TOKEN_MARKERS):
            continue
        if base[0].isdigit():  # 例如 1000SHIBUSDT 這種「面額縮放」代幣，避免跟正常幣搞混
            continue
        try:
            quote_volume = float(t.get("quoteVolume", 0))
        except (TypeError, ValueError):
            continue
        candidates.append((symbol, quote_volume))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [_from_binance_symbol(sym) for sym, _ in candidates[:top_n]]


class CryptoAdapter(StockDataAdapter):
    market = "CRYPTO"

    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # 價量資料
    # ------------------------------------------------------------------
    def fetch_price(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        cached = self._load_raw_if_cached("price", symbol, start_date, end_date)
        if cached is not None:
            df = pd.DataFrame(cached["data"])
            df["date"] = pd.to_datetime(df["date"])
            return df[PRICE_SCHEMA]

        try:
            klines = self._fetch_all_klines(symbol, start_date, end_date)
            if not klines:
                return self._empty_df(PRICE_SCHEMA)

            df = pd.DataFrame({
                "date": pd.to_datetime([k[0] for k in klines], unit="ms"),
                "symbol": symbol,
                "market": self.market,
                "open": [float(k[1]) for k in klines],
                "high": [float(k[2]) for k in klines],
                "low": [float(k[3]) for k in klines],
                "close": [float(k[4]) for k in klines],
                "volume": [float(k[5]) for k in klines],
                "source": "binance",
            })
            # 同一天可能因為排程一天抓好幾次而重複，用 date 去重保留最後一筆
            # （最後一筆代表最新抓到的「當天還在形成中」K 棒，saveprocessed 也會再做一次去重)
            df = df.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)

            self._save_raw("price", symbol, {"data": df.to_dict(orient="records")}, start_date, end_date)
            return df[PRICE_SCHEMA]
        except Exception as e:
            self.logger.error(f"[{symbol}] Binance klines 呼叫失敗: {e}")
            return self._empty_df(PRICE_SCHEMA)

    def _fetch_all_klines(self, symbol: str, start_date: str, end_date: str) -> list:
        """
        分頁抓完整個區間的日 K：單次請求最多 KLINES_LIMIT 根，
        區間超過這個長度就用回傳的最後一根 K 棒的收盤時間當下一頁的起點，繼續抓。
        """
        binance_symbol = _to_binance_symbol(symbol)
        start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

        all_klines: list = []
        cursor = start_ms
        while cursor < end_ms:
            resp = requests.get(
                f"{BINANCE_BASE_URL}/klines",
                params={
                    "symbol": binance_symbol, "interval": "1d",
                    "startTime": cursor, "endTime": end_ms,
                    "limit": KLINES_LIMIT,
                },
                timeout=10,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_klines.extend(batch)
            if len(batch) < KLINES_LIMIT:
                break
            cursor = batch[-1][6] + 1  # 用最後一根的收盤時間(ms)當下一頁起點，避免重複抓到同一根

        return all_klines

    # ------------------------------------------------------------------
    # 基本面 / 籌碼面：加密貨幣沒有對應概念，回傳空 DataFrame
    # ------------------------------------------------------------------
    def fetch_fundamental(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._empty_df(FUNDAMENTAL_SCHEMA)

    def fetch_chip(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._empty_df(CHIP_SCHEMA)


# ---------------------------------------------------------------------------
# 快速測試入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    adapter = CryptoAdapter()
    symbol = "BTC-USD"
    start = "2024-01-01"
    end = datetime.today().strftime("%Y-%m-%d")

    print(f"=== 抓取 {symbol} 價量資料 ===")
    price_df = adapter.fetch_price(symbol, start, end)
    print(price_df.tail())
    adapter.save_processed(price_df, "price", symbol)
