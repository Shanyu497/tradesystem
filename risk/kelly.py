"""
kelly.py
=========
資金管理模組：把 Phase 3 模型輸出的機率（勝率 p）跟 Phase 4 回測算出的賠率（payoff ratio），
轉換成「這筆交易應該用多少比例的資金去下」。

提供兩種方法（對應 Phase 1 的需求）：
1. 凱利公式（Kelly Criterion）：理論上「長期資金成長最大化」的下注比例，
   但原始公式（Full Kelly）在真實世界很危險——因為它假設你對勝率/賠率的估計是精確的，
   但你的估計一定有誤差，Full Kelly 對估計誤差極度敏感，容易建議出過度激進的倉位，
   一次估計錯誤就可能大幅回撤。所以實務上幾乎沒有人真的用 Full Kelly，
   都會打折扣用（Half Kelly、Quarter Kelly），這裡預設半凱利（fraction=0.5）。
2. 固定風險比例法（Fixed Fractional）：更保守、更直觀的替代方案，
   不需要精確估計勝率賠率，只要設定「每筆交易最多虧總資金的 X%」，
   根據停損距離反推部位大小。

兩種方法都內建「單筆倉位上限」的安全閥，避免任何一種公式在極端輸入下
（例如勝率被高估）建議出全押的荒謬結果。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KellyResult:
    win_prob: float
    payoff_ratio: float
    full_kelly_fraction: float       # 原始公式算出的比例（可能為負，代表不該進場）
    fractional_kelly_fraction: float  # 打折後的比例（實際建議使用這個）
    capped_fraction: float            # 再套用單筆上限後的最終比例
    position_value: float             # 換算成金額


def kelly_fraction(win_prob: float, payoff_ratio: float) -> float:
    """
    凱利公式：f* = p - (1-p)/b
    p: 勝率（0~1）
    b: 賠率，也就是「平均獲利 / 平均虧損」，b=2 代表贏的時候賺的是輸的時候賠的 2 倍

    如果算出負值，代表這個「勝率+賠率」組合長期是虧錢的，不該進場，回傳 0。
    """
    if payoff_ratio <= 0:
        return 0.0
    f_star = win_prob - (1 - win_prob) / payoff_ratio
    return max(f_star, 0.0)


def kelly_position_size(capital: float, win_prob: float, payoff_ratio: float,
                          fraction: float = 0.5, max_position_pct: float = 0.25) -> KellyResult:
    """
    capital: 總資金
    win_prob: 勝率 p（建議直接用 Phase 3 模型輸出的機率，或用回測算出的歷史勝率）
    payoff_ratio: 賠率 b（建議用回測算出的「平均獲利 / 平均虧損」）
    fraction: 打折係數，0.5 = 半凱利（預設，也是最常見的實務選擇）
    max_position_pct: 單筆倉位佔總資金的上限，不論公式算出多少都不會超過這個比例
    """
    f_star = kelly_fraction(win_prob, payoff_ratio)
    f_adjusted = f_star * fraction
    f_capped = min(f_adjusted, max_position_pct)

    return KellyResult(
        win_prob=win_prob,
        payoff_ratio=payoff_ratio,
        full_kelly_fraction=f_star,
        fractional_kelly_fraction=f_adjusted,
        capped_fraction=f_capped,
        position_value=capital * f_capped,
    )


@dataclass
class FixedRiskResult:
    risk_amount: float      # 這筆交易願意承受的最大虧損金額
    shares: float           # 建議買進股數（可能要再依實際單位如台股一張=1000股做整數化）
    position_value: float   # 換算成部位總金額


def fixed_risk_position_size(capital: float, risk_per_trade_pct: float,
                               entry_price: float, stop_loss_price: float) -> FixedRiskResult:
    """
    固定風險比例法：不論勝率賠率怎麼估，每筆交易最多虧總資金的 risk_per_trade_pct，
    根據「進場價 - 停損價」的距離，反推應該買幾股才不會虧超過設定金額。

    entry_price / stop_loss_price: 用你打算進場的價格跟停損價格（例如用 ATR 或近期低點設停損）
    """
    risk_amount = capital * risk_per_trade_pct
    risk_per_share = abs(entry_price - stop_loss_price)

    if risk_per_share <= 0:
        return FixedRiskResult(risk_amount=risk_amount, shares=0.0, position_value=0.0)

    shares = risk_amount / risk_per_share
    return FixedRiskResult(
        risk_amount=risk_amount,
        shares=shares,
        position_value=shares * entry_price,
    )
