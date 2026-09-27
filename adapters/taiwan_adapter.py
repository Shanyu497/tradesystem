"""
taiwan_adapter.py
==================
台股資料 Adapter。

資料源優先權：
1. FinMind：價量、基本面（財報）、籌碼面（三大法人、融資融券）一次到位，
   免費額度足夠日常使用（個人使用大約每小時 300~600 次請求，視方案而定）。
2. yfinance：當 FinMind 額度用完或掛掉時的價量備援
   （symbol 需要轉換成 "2330.TW" 上市 / "6488.TWO" 上櫃格式）。

使用前置作業：
    pip install FinMind yfinance pandas pyarrow

FinMind 免費版不需要 API token 也能用，但有請求限制；
若之後要跑大量標的，建議到 https://finmind.github.io/ 註冊拿 token，
並用環境變數 FINMIND_TOKEN 帶入，不要寫死在程式碼裡。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd

from .base_adapter import (
    CHIP_SCHEMA,
    FUNDAMENTAL_SCHEMA,
    PRICE_SCHEMA,
    AdapterConfig,
    StockDataAdapter,
)

try:
    from FinMind.data import DataLoader
except ImportError:  # pragma: no cover
    DataLoader = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


class TaiwanStockAdapter(StockDataAdapter):
    market = "TW"

    def __init__(self, config: Optional[AdapterConfig] = None, finmind_token: Optional[str] = None):
        super().__init__(config)
        self._finmind_token = finmind_token or os.environ.get("FINMIND_TOKEN", "")
        self._finmind_client = None
        if DataLoader is not None:
            self._finmind_client = DataLoader()
            if self._finmind_token:
                self._finmind_client.login_by_token(api_token=self._finmind_token)
        else:
            self.logger.warning("未安裝 FinMind，價量/基本面/籌碼面將完全依賴 yfinance 備援")

    # ------------------------------------------------------------------
    # 價量資料
    # ------------------------------------------------------------------
    def fetch_price(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        cached = self._load_raw_if_cached("price", symbol, start_date, end_date)
        if cached is not None:
            return self._normalize_price_finmind(pd.DataFrame(cached["data"]), symbol)

        df = self._fetch_price_finmind(symbol, start_date, end_date)
        if df.empty:
            self.logger.warning(f"[{symbol}] FinMind 價量取得失敗或為空，改用 yfinance 備援")
            df = self._fetch_price_yfinance(symbol, start_date, end_date)
        return df

    def _fetch_price_finmind(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self._finmind_client is None:
            return self._empty_df(PRICE_SCHEMA)
        try:
            raw = self._finmind_client.taiwan_stock_daily(
                stock_id=symbol, start_date=start_date, end_date=end_date
            )
            if raw is None or raw.empty:
                return self._empty_df(PRICE_SCHEMA)
            self._save_raw("price", symbol, {"data": raw.to_dict(orient="records")}, start_date, end_date)
            return self._normalize_price_finmind(raw, symbol)
        except Exception as e:
            self.logger.error(f"[{symbol}] FinMind API 呼叫失敗: {e}")
            return self._empty_df(PRICE_SCHEMA)

    def _normalize_price_finmind(self, raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if raw.empty:
            return self._empty_df(PRICE_SCHEMA)
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["date"]),
            "symbol": symbol,
            "market": self.market,
            "open": raw["open"].astype(float),
            "high": raw["max"].astype(float),
            "low": raw["min"].astype(float),
            "close": raw["close"].astype(float),
            "volume": raw["Trading_Volume"].astype(float),
            "source": "finmind",
        })
        return df[PRICE_SCHEMA]

    def _fetch_price_yfinance(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if yf is None:
            self.logger.error("yfinance 未安裝，無法備援")
            return self._empty_df(PRICE_SCHEMA)
        # 台股上市代號用 .TW，上櫃用 .TWO；這裡先嘗試上市，失敗再試上櫃
        for suffix in (".TW", ".TWO"):
            try:
                ticker = yf.Ticker(f"{symbol}{suffix}")
                hist = ticker.history(start=start_date, end=end_date)
                if hist.empty:
                    continue
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
                    "source": f"yfinance{suffix}",
                })
                return df[PRICE_SCHEMA]
            except Exception as e:
                self.logger.error(f"[{symbol}{suffix}] yfinance 呼叫失敗: {e}")
        return self._empty_df(PRICE_SCHEMA)

    # ------------------------------------------------------------------
    # 基本面資料（先實作財報營收，EPS/PE 等可依需求擴充）
    # ------------------------------------------------------------------
    def fetch_fundamental(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self._finmind_client is None:
            return self._empty_df(FUNDAMENTAL_SCHEMA)
        try:
            raw = self._finmind_client.taiwan_stock_month_revenue(
                stock_id=symbol, start_date=start_date, end_date=end_date
            )
            if raw is None or raw.empty:
                return self._empty_df(FUNDAMENTAL_SCHEMA)
            self._save_raw("fundamental", symbol, {"data": raw.to_dict(orient="records")}, start_date, end_date)
            df = pd.DataFrame({
                "date": pd.to_datetime(raw["date"]),
                "symbol": symbol,
                "market": self.market,
                "revenue": raw["revenue"].astype(float),
                "eps": None,
                "pe_ratio": None,
                "pb_ratio": None,
                "dividend_yield": None,
                "source": "finmind",
            })
            return df[FUNDAMENTAL_SCHEMA]
        except Exception as e:
            self.logger.error(f"[{symbol}] 基本面資料取得失敗: {e}")
            return self._empty_df(FUNDAMENTAL_SCHEMA)

    # ------------------------------------------------------------------
    # 籌碼面資料（三大法人買賣超）
    # ------------------------------------------------------------------
    def fetch_chip(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self._finmind_client is None:
            return self._empty_df(CHIP_SCHEMA)
        try:
            raw = self._finmind_client.taiwan_stock_institutional_investors(
                stock_id=symbol, start_date=start_date, end_date=end_date
            )
            if raw is None or raw.empty:
                return self._empty_df(CHIP_SCHEMA)
            self._save_raw("chip", symbol, {"data": raw.to_dict(orient="records")}, start_date, end_date)

            # FinMind 回傳的是「長格式」（每個法人別一列），這裡轉成寬格式方便使用
            pivot = raw.pivot_table(
                index="date", columns="name",
                values=["buy", "sell"], aggfunc="sum",
            )
            pivot["foreign_net"] = pivot.get(("buy", "Foreign_Investor"), 0) - pivot.get(("sell", "Foreign_Investor"), 0)
            pivot["trust_net"] = pivot.get(("buy", "Investment_Trust"), 0) - pivot.get(("sell", "Investment_Trust"), 0)
            pivot["dealer_net"] = pivot.get(("buy", "Dealer_self"), 0) - pivot.get(("sell", "Dealer_self"), 0)
            pivot = pivot.reset_index()

            df = pd.DataFrame({
                "date": pd.to_datetime(pivot["date"]),
                "symbol": symbol,
                "market": self.market,
                "foreign_net_buy": pivot["foreign_net"],
                "investment_trust_net_buy": pivot["trust_net"],
                "dealer_net_buy": pivot["dealer_net"],
                "margin_balance": None,
                "short_balance": None,
                "source": "finmind",
            })
            return df[CHIP_SCHEMA]
        except Exception as e:
            self.logger.error(f"[{symbol}] 籌碼面資料取得失敗: {e}")
            return self._empty_df(CHIP_SCHEMA)


# ---------------------------------------------------------------------------
# 快速測試入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    adapter = TaiwanStockAdapter()
    symbol = "2330"  # 台積電
    start = "2024-01-01"
    end = datetime.today().strftime("%Y-%m-%d")

    print(f"=== 抓取 {symbol} 價量資料 ===")
    price_df = adapter.fetch_price(symbol, start, end)
    print(price_df.tail())
    adapter.save_processed(price_df, "price", symbol)

    print(f"\n=== 抓取 {symbol} 籌碼面資料 ===")
    chip_df = adapter.fetch_chip(symbol, start, end)
    print(chip_df.tail())
    adapter.save_processed(chip_df, "chip", symbol)
