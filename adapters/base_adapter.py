"""
base_adapter.py
================
所有市場（台股 / 美股）的資料 Adapter 都必須繼承這個基底類別，
並實作 fetch_price()、fetch_fundamental()、fetch_chip() 三個方法。

設計原則：
- 對外一律回傳「標準化」的 pandas DataFrame，欄位命名統一，
  這樣運算引擎層（技術指標、ML 特徵工程）完全不需要知道資料是從
  FinMind 還是 yfinance 還是其他 API 來的。
- 每次呼叫 API 前，先檢查本地 raw 層有沒有當日快取，避免重複打 API
  （尤其 FinMind、Alpha Vantage 這類有免費額度限制的 API）。
- 任何 API 呼叫都包在 try/except 裡，並記錄詳細 log，
  讓 Phase 5 的排程系統知道哪個資料源今天掛了。
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# 標準化 Schema 定義
# ---------------------------------------------------------------------------
# 價量資料（OHLCV）標準欄位
PRICE_SCHEMA = [
    "date",      # datetime64
    "symbol",    # str，統一格式如 "2330"（台股）或 "AAPL"（美股），不含後綴
    "market",    # str，"TW" / "US" / "CRYPTO"
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",    # 這筆資料是哪個 API 提供的，方便除錯與資料源優先權判斷
]

# 基本面資料標準欄位（先定義最常用的幾個，之後可擴充）
FUNDAMENTAL_SCHEMA = [
    "date", "symbol", "market",
    "revenue", "eps", "pe_ratio", "pb_ratio", "dividend_yield",
    "source",
]

# 籌碼面資料標準欄位（台股特有，美股可能對應機構持股等）
CHIP_SCHEMA = [
    "date", "symbol", "market",
    "foreign_net_buy",      # 外資買賣超（股數）
    "investment_trust_net_buy",  # 投信買賣超
    "dealer_net_buy",       # 自營商買賣超
    "margin_balance",       # 融資餘額
    "short_balance",        # 融券餘額
    "source",
]


# 專案根目錄 = 這個檔案所在的 adapters/ 資料夾的上一層（quant_system/）
# 用相對於程式檔案的位置計算，不論在哪台電腦、哪個作業系統、
# 從哪個目錄執行，都會正確指向 quant_system/data/ 底下。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AdapterConfig:
    """每個 Adapter 共用的設定"""
    raw_dir: Path = _PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = _PROJECT_ROOT / "data" / "processed"
    use_cache: bool = True  # 當天已經抓過就不重複打 API


class StockDataAdapter(ABC):
    """
    抽象基底類別。美股/台股 Adapter 都繼承這個類別。
    """

    market: str = "UNKNOWN"

    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # 子類別必須實作的方法
    # ------------------------------------------------------------------
    @abstractmethod
    def fetch_price(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """回傳符合 PRICE_SCHEMA 的 DataFrame"""
        raise NotImplementedError

    @abstractmethod
    def fetch_fundamental(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """回傳符合 FUNDAMENTAL_SCHEMA 的 DataFrame"""
        raise NotImplementedError

    @abstractmethod
    def fetch_chip(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """回傳符合 CHIP_SCHEMA 的 DataFrame（美股若無對應資料可回傳空 DataFrame）"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 共用工具方法
    # ------------------------------------------------------------------
    def _raw_cache_path(self, kind: str, symbol: str, start_date: Optional[str] = None,
                          end_date: Optional[str] = None, day: Optional[date] = None) -> Path:
        day = day or date.today()
        # 快取檔名同時帶入請求的日期區間，避免「同一天、不同範圍」的請求互相蓋過彼此的結果
        range_tag = f"{start_date}_{end_date}" if start_date and end_date else "default"
        return self.config.raw_dir / self.market / kind / f"{symbol}_{range_tag}_{day.isoformat()}.json"

    def _save_raw(self, kind: str, symbol: str, payload: dict,
                   start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        path = self._raw_cache_path(kind, symbol, start_date, end_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, default=str)
        self.logger.info(f"raw 資料已快取: {path}")

    def _load_raw_if_cached(self, kind: str, symbol: str,
                              start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[dict]:
        if not self.config.use_cache:
            return None
        path = self._raw_cache_path(kind, symbol, start_date, end_date)
        if path.exists():
            self.logger.info(f"命中今日快取，略過 API 呼叫: {path}")
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    @staticmethod
    def _empty_df(schema: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=schema)

    def save_processed(self, df: pd.DataFrame, kind: str, symbol: str) -> Path:
        """把清洗完的標準 DataFrame 存成 Parquet，供下游模組讀取"""
        out_dir = self.config.processed_dir / self.market / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{symbol}.parquet"
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            df = (
                pd.concat([existing, df])
                .drop_duplicates(subset=["date", "symbol"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
        df.to_parquet(out_path, index=False)
        self.logger.info(f"已寫入 processed 層: {out_path}（共 {len(df)} 筆）")
        return out_path
