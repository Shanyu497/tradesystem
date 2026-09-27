"""
engine.py
==========
回測引擎，分兩個步驟：

1. walk_forward_predict()：對整段歷史資料做「純樣本外」的機率預測。
   跟 signals/model.py 的 train_and_evaluate() 不同之處在於：
   那邊只回傳「彙總後的平均指標」（例如平均精準率），這裡回傳的是
   「每一天」的樣本外機率，才能拿去模擬「如果那天真的照這個機率操作」的結果。

2. simulate_strategy() + compute_performance()：把機率序列轉換成一筆一筆的
   模擬交易，算出勝率、賠率（風險報酬比）——這兩個數字是「訊號品質」的評估
   （每筆交易獨立看待，不管實際上能不能同時持有），拿來餵給 Kelly 公式剛好合適，
   因為 Kelly 公式本身就是「單筆下注」的公式，不需要知道部位重疊的狀況。

3. simulate_portfolio()：上面兩步算出來的「總報酬/最大回撤」如果直接拿來看，
   會嚴重失真——因為那是假設「每筆訊號都依序把全部資金投入、賺了馬上滾入下一筆」，
   但 horizon=20 天的策略，同一檔標的很可能還沒平倉、新訊號就又觸發了，
   實際上你不可能拿同一筆錢同時開好幾倉。simulate_portfolio() 用「同時最多持有
   幾個部位」（max_concurrent_positions，預設 1，即「這檔標的同時只會有一筆倉位，
   還沒平倉的話新訊號就跳過」）重新模擬一條權益曲線，資金沒投入時視為閒置現金
   （報酬率 0%，不是真的完全沒有機會成本，但比原本「無限複利」的假設實際很多）。
   這才是「總報酬/最大回撤」該看的數字；勝率/賠率則繼續用 compute_performance()
   的結果（訊號本身的品質，不受你同時能開幾倉影響）。

重要的簡化假設（务必告知使用者，不是要隱藏)：
- apply_transaction_costs() 把手續費/交易稅算成「固定比例」扣在每筆交易的報酬率上，
  這是回測的標準簡化——但沒有模擬「最低手續費」這個下限（很多券商電子下單打完折
  還是有 NT$1~20 的最低收費）。資金量越小、單筆交易金額越低，最低手續費對報酬率的
  侵蝕比例就越高，這裡的估計在小額交易（例如幾千元台幣）時會偏樂觀，
  請自行對照你實際下單金額跟券商的最低手續費規則。
- simulate_portfolio() 的「閒置資金報酬率 0%」是簡化：現實中你可能會把沒用到的
  資金拿去買別的標的、放定存，或乾脆留著等下一次機會——這裡選擇最保守的假設，
  不會讓總報酬看起來比實際更好。
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

# 加密貨幣：Binance 現貨交易手續費，買賣各收一次（未用 BNB 折抵手續費的一般費率），
# 沒有台股那種交易稅。如果你有開 BNB 折抵或 VIP 費率，實際成本會更低，可自行調整。
CRYPTO_TAKER_FEE_PCT = 0.1 / 100


def _round_trip_cost_pct(market: str, symbol: str) -> float:
    """回傳「買進+賣出」一次完整交易的成本比例（手續費+交易稅）"""
    if market == "US":
        return US_ROUND_TRIP_PCT
    if market == "CRYPTO":
        return CRYPTO_TAKER_FEE_PCT * 2
    if market == "TW":
        # 台股 ETF 代號慣例以 0 開頭（0050、00646...），一般個股以其他數字開頭，
        # 用這個慣例判斷該用哪一種交易稅率，不是完美規則但涵蓋目前監控清單的情況。
        tax = TW_ETF_TAX_PCT if symbol.startswith("0") else TW_STOCK_TAX_PCT
        return TW_BROKERAGE_FEE_PCT * 2 + tax
    raise ValueError(f"未知市場: {market!r}（目前只認得 'TW'、'US' 或 'CRYPTO'）")


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


@dataclass
class PortfolioMetrics:
    n_trades: int       # 真的被「接受」進場的交易數（受 max_concurrent_positions 限制）
    n_skipped: int       # 因為部位已經滿了而跳過的訊號數（同一檔標的還在持有中）
    max_drawdown: float  # 用有限部位數模擬出來的權益曲線算出的回撤，比 BacktestMetrics 真實
    total_return: float  # 同上，是「總報酬」不是「勝率/賠率」——那兩個看 BacktestMetrics


def simulate_portfolio(trades_df: pd.DataFrame, all_dates: pd.Series, horizon: int,
                         max_concurrent_positions: int = 1) -> Optional[PortfolioMetrics]:
    """
    用「同時最多持有 max_concurrent_positions 個部位」重新模擬一條權益曲線，
    解決 compute_performance() 假設「每筆交易都能同時全額進場」的失真問題。

    all_dates: 這檔標的完整的日期序列（來自 price_df，按日期排序、index 從 0 開始連續），
               用來查「這筆交易進場後，horizon 天後的那一天是哪一天」，藉此判斷
               「這筆交易平倉前，是不是有新訊號想擠進來但部位已經滿了」。
    max_concurrent_positions: 同一檔標的同時最多持有幾筆部位，預設 1
               （最保守：還沒平倉，新訊號一律跳過，不加碼、不同時開兩倉）。

    回傳的 n_trades 可能比 compute_performance() 算出來的少很多——這是正常的，
    代表很多訊號在「上一筆還沒平倉」的時候就被略過了，這正是這個函式存在的目的。
    """
    if trades_df.empty or max_concurrent_positions < 1:
        return None

    finite_mask = np.isfinite(trades_df["pnl_pct"])
    trades_df = trades_df[finite_mask].sort_values("date").reset_index(drop=True)
    if trades_df.empty:
        return None

    all_dates = all_dates.reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_dates = len(all_dates)

    open_exit_dates: list = []  # 目前還在持有中的部位，各自的平倉日期
    accepted: list = []
    skipped = 0

    for row in trades_df.itertuples(index=False):
        entry_date = row.date
        # 先把「平倉日已經在今天(含)之前」的部位釋放掉，騰出額度
        open_exit_dates = [d for d in open_exit_dates if d > entry_date]

        if len(open_exit_dates) >= max_concurrent_positions:
            skipped += 1
            continue

        entry_idx = date_to_idx.get(entry_date)
        if entry_idx is None:
            # 理論上 entry_date 一定來自 all_dates，這裡只是防呆
            skipped += 1
            continue
        exit_idx = entry_idx + horizon
        exit_date = all_dates.iloc[exit_idx] if exit_idx < n_dates else entry_date

        open_exit_dates.append(exit_date)
        accepted.append({"exit_date": exit_date, "pnl_pct": row.pnl_pct})

    if not accepted:
        return None

    accepted_df = pd.DataFrame(accepted).sort_values("exit_date").reset_index(drop=True)

    # 依「平倉時間」滾複利，每筆只佔 1/max_concurrent_positions 的資金比例，
    # 其餘資金視為閒置現金（報酬率 0%），這樣才不會出現「部位重疊也照樣全額複利」
    # 的失真權益曲線。
    slot_fraction = 1.0 / max_concurrent_positions
    equity = (1 + accepted_df["pnl_pct"] * slot_fraction).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    return PortfolioMetrics(
        n_trades=len(accepted_df),
        n_skipped=skipped,
        max_drawdown=float(drawdown.min()),
        total_return=float(equity.iloc[-1] - 1),
    )
