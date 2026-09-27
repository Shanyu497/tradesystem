"""
engine.py
==========
回測引擎，分兩個步驟：

1. walk_forward_predict()：對整段歷史資料做「純樣本外」的機率預測。
   跟 signals/model.py 的 train_and_evaluate() 不同之處在於：
   那邊只回傳「彙總後的平均指標」（例如平均精準率），這裡回傳的是
   「每一天」的樣本外機率，才能拿去模擬「如果那天真的照這個機率操作」的結果。

2. simulate_strategy() + compute_performance()：把機率序列轉換成一筆一筆的
   模擬交易，算出勝率、賠率（風險報酬比）、最大回撤這些 Phase 1 要求的指標。

重要的簡化假設（务必告知使用者，不是要隱藏)：
- 每筆交易視為獨立事件，不考慮「多筆交易的持有期間重疊」對實際可動用資金的排擠。
  例如 horizon=5 天，同一檔標的可能連續好幾天都觸發訊號，實際上你不可能拿同一筆錢
  同時開好幾倉——這裡的績效數字是「訊號品質」的評估，不是「實際資金運用」的模擬。
  真正要拿來實盤，還需要加入部位管理（同時最多開幾倉、資金重疊時怎麼排序）的邏輯，
  這個複雜度留到你要正式上線前再加。
- apply_transaction_costs() 把手續費/交易稅算成「固定比例」扣在每筆交易的報酬率上，
  這是回測的標準簡化——但沒有模擬「最低手續費」這個下限（很多券商電子下單打完折
  還是有 NT$1~20 的最低收費）。資金量越小、單筆交易金額越低，最低手續費對報酬率的
  侵蝕比例就越高，這裡的估計在小額交易（例如幾千元台幣）時會偏樂觀，
  請自行對照你實際下單金額跟券商的最低手續費規則。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

from signals.features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# 交易成本假設（比例式，看得到的部分；最低手續費下限沒有模擬，見上方 docstring）
# ---------------------------------------------------------------------------
# 台股：買、賣各收一次手續費（牌告費率上限 0.1425%，多數券商電子下單會打折，
# 這裡用未打折的牌告費率當保守預設），賣出時另外收交易稅
# （一般股票 0.3%，ETF 因為政策優惠只收 0.1%）。
TW_BROKERAGE_FEE_PCT = 0.1425 / 100
TW_STOCK_TAX_PCT = 0.3 / 100
TW_ETF_TAX_PCT = 0.1 / 100

# 美股：假設用零手續費券商（Robinhood/Firstrade 等目前多數美股交易免佣金），
# 如果你用的券商有收手續費，把這個值改成實際費率（估計成一個比例）。
US_ROUND_TRIP_PCT = 0.0


def _round_trip_cost_pct(market: str, symbol: str) -> float:
    """回傳「買進+賣出」一次完整交易的成本比例（手續費+交易稅）"""
    if market == "US":
        return US_ROUND_TRIP_PCT
    if market == "TW":
        # 台股 ETF 代號慣例以 0 開頭（0050、00646...），一般個股以其他數字開頭，
        # 用這個慣例判斷該用哪一種交易稅率，不是完美規則但涵蓋目前監控清單的情況。
        tax = TW_ETF_TAX_PCT if symbol.startswith("0") else TW_STOCK_TAX_PCT
        return TW_BROKERAGE_FEE_PCT * 2 + tax
    raise ValueError(f"未知市場: {market!r}（目前只認得 'TW' 或 'US'）")


def apply_transaction_costs(trades_df: pd.DataFrame, market: str, symbol: str) -> pd.DataFrame:
    """
    把 simulate_strategy() 選出的交易，扣掉真實的手續費與交易稅，
    讓後續 compute_performance() 算出來的勝率/賠率反映「真的能落袋」的報酬，
    而不是零成本的理論值。
    """
    if trades_df.empty:
        return trades_df

    cost_pct = _round_trip_cost_pct(market, symbol)
    trades_df = trades_df.copy()
    trades_df["pnl_pct"] = trades_df["pnl_pct"] - cost_pct
    return trades_df


def walk_forward_predict(train_df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """
    回傳一段完整、沒有偷看未來的機率時間序列：'date', 'future_return', 'probability'。

    注意：第一折沒有「更早的資料」可以拿來訓練，所以結果不會涵蓋最早那一段資料
    （這是 Walk-forward 方法本身的限制，不是 bug）。
    """
    X = train_df[FEATURE_COLUMNS]
    y = train_df["label"]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []
    for train_idx, test_idx in tscv.split(X):
        if y.iloc[train_idx].nunique() < 2:
            continue  # 這折的訓練資料裡只有單一類別，無法訓練分類器，跳過

        fold_model = HistGradientBoostingClassifier(random_state=42)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = fold_model.predict_proba(X.iloc[test_idx])[:, 1]

        fold_df = train_df.iloc[test_idx][["date", "future_return"]].copy()
        fold_df["probability"] = proba
        fold_results.append(fold_df)

    if not fold_results:
        return pd.DataFrame(columns=["date", "future_return", "probability"])

    return pd.concat(fold_results).sort_values("date").reset_index(drop=True)


@dataclass
class BacktestMetrics:
    n_trades: int
    win_rate: float
    avg_win: float          # 獲利交易的平均報酬率
    avg_loss: float          # 虧損交易的平均報酬率（正數，代表虧損幅度）
    payoff_ratio: float      # avg_win / avg_loss，這就是 Kelly 公式要用的賠率 b
    max_drawdown: float
    total_return: float      # 假設每筆訊號都依序投入、獲利再滾入下一筆（複利)的總報酬


def simulate_strategy(predictions_df: pd.DataFrame, prob_threshold: float = 0.5) -> pd.DataFrame:
    """篩選出機率 >= prob_threshold 的樣本，視為「這天觸發進場訊號」的模擬交易"""
    trades = predictions_df[predictions_df["probability"] >= prob_threshold].copy()
    trades = trades.rename(columns={"future_return": "pnl_pct"})
    return trades.sort_values("date").reset_index(drop=True)


def compute_performance(trades_df: pd.DataFrame) -> Optional[BacktestMetrics]:
    if trades_df.empty:
        return None

    # 防呆：過濾掉任何殘留的無限值/NaN（理論上 labels.py 已經在源頭處理過，
    # 這裡是第二層防護，避免任何一筆髒資料拖垮整條權益曲線的計算）
    finite_mask = np.isfinite(trades_df["pnl_pct"])
    if not finite_mask.all():
        dropped = (~finite_mask).sum()
        logging.getLogger("backtest.engine").warning(
            f"回測資料中有 {dropped} 筆報酬率是異常值（inf/NaN），已排除，建議檢查原始價量資料是否有異常"
        )
    trades_df = trades_df[finite_mask]

    if trades_df.empty:
        return None

    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] <= 0]

    win_rate = len(wins) / len(trades_df)
    avg_win = float(wins["pnl_pct"].mean()) if not wins.empty else 0.0
    avg_loss = float(abs(losses["pnl_pct"].mean())) if not losses.empty else 0.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float("nan")

    # 依序（用日期排序）把每筆交易的報酬率滾入複利，得到一條簡化的權益曲線
    equity = (1 + trades_df["pnl_pct"]).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min())
    total_return = float(equity.iloc[-1] - 1)

    return BacktestMetrics(
        n_trades=len(trades_df),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        max_drawdown=max_drawdown,
        total_return=total_return,
    )
