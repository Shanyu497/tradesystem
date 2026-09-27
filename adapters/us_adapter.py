"""
us_adapter.py
==============
美股資料 Adapter。

資料源優先權：
1. yfinance：價量資料主力，免費、無需 API key，歷史資料很穩定。
2. Finnhub：備援價量 + 基本面資料（公司概況、EPS、PE 等）。
   免費額度約 60 次/分鐘，需要到 https://finnhub.io/ 註冊拿 API key，
   透過環境變數 FINNHUB_API_KEY 帶入，不要寫死在程式碼裡。

籌碼面（機構持股 13F、內部人交易）：
   免費且穩定的即時 API 選擇很少（多半是付費服務如 WhaleWisdom），
   這裡先實作介面、回傳空 DataFrame，你之後若要做這塊，
   建議先評估是否真的對訊號有幫助，再決定要不要花錢接付費源。
"""

from __future__ import annotations

import os
from datetime import datetime
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

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


class USStockAdapter(StockDataAdapter):
    market = "US"

    def __init__(self, config: Optional[AdapterConfig] = None, finnhub_api_key: Optional[str] = None):
        super().__init__(config)
        self._finnhub_key = finnhub_api_key or os.environ.get("FINNHUB_API_KEY", "")
        if not self._finnhub_key:
            self.logger.warning("未設定 FINNHUB_API_KEY，備援價量與基本面資料將無法使用")

    # ------------------------------------------------------------------
    # 價量資料
    # ------------------------------------------------------------------
    def fetch_price(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        cached = self._load_raw_if_cached("price", symbol, start_date, end_date)
        if cached is not None:
            df = pd.DataFrame(cached["data"])
            df["date"] = pd.to_datetime(df["date"])
            return df[PRICE_SCHEMA]

        df = self._fetch_price_yfinance(symbol, start_date, end_date)
        if df.empty:
            self.logger.warning(f"[{symbol}] yfinance 價量取得失敗或為空，改用 Finnhub 備援")
            df = self._fetch_price_finnhub(symbol, start_date, end_date)
        return df

    def _fetch_price_yfinance(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if yf is None:
            self.logger.error("yfinance 未安裝")
            return self._empty_df(PRICE_SCHEMA)
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            if hist.empty:
                return self._empty_df(PRICE_SCHEMA)
            hist = hist.reset_index()
            df = pd.DataFrame({
                "date": pd.to_datetime(hist["Date"]).dt.tz_localize(None),
                "symbol": symbol,
                "market": self.market,
                "open": hist["Open"],
                "high": hist["High"],
                "low": hist["Low"],
                "close": hist["Close"],
                "volume": hist["Volume"],
                "source": "yfinance",
            })
            self._save_raw("price", symbol, {"data": df.to_dict(orient="records")}, start_date, end_date)
            return df[PRICE_SCHEMA]
        except Exception as e:
            self.logger.error(f"[{symbol}] yfinance 呼叫失敗: {e}")
            return self._empty_df(PRICE_SCHEMA)

    def _fetch_price_finnhub(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self._finnhub_key:
            return self._empty_df(PRICE_SCHEMA)
        try:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
            resp = requests.get(
                f"{FINNHUB_BASE_URL}/stock/candle",
                params={
                    "symbol": symbol, "resolution": "D",
                    "from": start_ts, "to": end_ts,
                    "token": self._finnhub_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json()
            if raw.get("s") != "ok":
                return self._empty_df(PRICE_SCHEMA)
            self._save_raw("price", symbol, {"data": raw})
            df = pd.DataFrame({
                "date": pd.to_datetime(raw["t"], unit="s"),
                "symbol": symbol,
                "market": self.market,
                "open": raw["o"],
                "high": raw["h"],
                "low": raw["l"],
                "close": raw["c"],
                "volume": raw["v"],
                "source": "finnhub",
            })
            return df[PRICE_SCHEMA]
        except Exception as e:
            self.logger.error(f"[{symbol}] Finnhub 呼叫失敗: {e}")
            return self._empty_df(PRICE_SCHEMA)

    # ------------------------------------------------------------------
    # 基本面資料（用 Finnhub 的公司基本指標）
    # ------------------------------------------------------------------
    def fetch_fundamental(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self._finnhub_key:
            return self._empty_df(FUNDAMENTAL_SCHEMA)
        try:
            resp = requests.get(
                f"{FINNHUB_BASE_URL}/stock/metric",
                params={"symbol": symbol, "metric": "all", "token": self._finnhub_key},
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json().get("metric", {})
            if not raw:
                return self._empty_df(FUNDAMENTAL_SCHEMA)
            self._save_raw("fundamental", symbol, {"data": raw}, start_date, end_date)
            df = pd.DataFrame([{
                "date": datetime.today(),
                "symbol": symbol,
                "market": self.market,
                "revenue": raw.get("revenuePerShareTTM"),
                "eps": raw.get("epsTTM"),
                "pe_ratio": raw.get("peTTM"),
                "pb_ratio": raw.get("pbAnnual"),
                "dividend_yield": raw.get("dividendYieldIndicatedAnnual"),
                "source": "finnhub",
            }])
            return df[FUNDAMENTAL_SCHEMA]
        except Exception as e:
            self.logger.error(f"[{symbol}] 基本面資料取得失敗: {e}")
            return self._empty_df(FUNDAMENTAL_SCHEMA)

    # ------------------------------------------------------------------
    # 籌碼面資料：免費源有限，先留空介面
    # ------------------------------------------------------------------
    def fetch_chip(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.logger.info(f"[{symbol}] 美股籌碼面資料目前無免費穩定源，回傳空 DataFrame")
        return self._empty_df(CHIP_SCHEMA)


# ---------------------------------------------------------------------------
# 快速測試入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    adapter = USStockAdapter()
    symbol = "AAPL"
    start = "2024-01-01"
    end = datetime.today().strftime("%Y-%m-%d")

    print(f"=== 抓取 {symbol} 價量資料 ===")
    price_df = adapter.fetch_price(symbol, start, end)
    print(price_df.tail())
    adapter.save_processed(price_df, "price", symbol)

    print(f"\n=== 抓取 {symbol} 基本面資料 ===")
    fundamental_df = adapter.fetch_fundamental(symbol, start, end)
    print(fundamental_df)
