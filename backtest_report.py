"""
backtest_report.py
====================
Phase 4 主入口：把「機率訊號」轉換成實際的策略績效數字（勝率、賠率/風險報酬比、
最大回撤），再用這些數字幫你算出凱利公式建議的倉位大小。

用法：
    py backtest_report.py --capital 500000
    py backtest_report.py --capital 500000 --kelly-fraction 0.25   # 更保守，四分之一凱利
    py backtest_report.py --capital 500000 --prob-threshold 0.6    # 機率門檻拉高，訊號更少但更嚴格
    py backtest_report.py --capital 500000 --symbols 2330,AAPL

這不是財務建議，只是把 Phase 1~3 產出的數字套進公式算出來的參考值——
凱利公式建議的倉位完全建立在「回測算出的勝率/賠率在未來會持續成立」這個假設上，
過去的統計結果不保證未來會重演，實際下單前務必自己再三確認。

勝率/賠率已經扣掉手續費與交易稅（見 backtest/engine.py 的 apply_transaction_costs()），
不是零成本的理論值；但沒有模擬「最低手續費」這個下限，資金量越小、單筆交易金額越低，
這裡的估計就會越偏樂觀，小額交易請自行對照你的券商規則。

「最大回撤」「累積報酬」是用 simulate_portfolio() 算出來的，會考慮「同一檔標的
還沒平倉、新訊號就又觸發」這種部位重疊的情況（見 backtest/engine.py 開頭的說明），
不是天真地假設每筆訊號都能同時全額進場——horizon 越長，這個差異越明顯。
--max-concurrent-positions 可以調整「同時最多持有幾筆」，預設 1（最保守）。
"""

from __future__ import annotations

import argparse
import logging

from backtest.engine import (
    apply_transaction_costs,
    compute_performance,
    simulate_portfolio,
    simulate_strategy,
    walk_forward_predict,
)
from pipeline import load_watchlist
from risk.kelly import kelly_position_size
from signal_report import load_price_data
from signals.dataset import build_dataset
from signals.model import is_signal_reliable, predict_latest, train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtest_report")

MIN_TRAIN_SAMPLES = 100

# 建議倉位金額低於這個門檻時，額外提醒「最低手續費」可能吃掉報酬——
# apply_transaction_costs() 只算比例式成本，金額越小，最低手續費（很多券商電子下單
# 仍有 NT$1~20 或等值美元的下限）佔比就越高，比例模型會低估實際成本。
SMALL_POSITION_WARNING_THRESHOLD = 5000


def run_backtest_report(capital: float, horizon: int, threshold: float,
                          prob_threshold: float, kelly_fraction: float,
                          max_position_pct: float,
                          symbol_filter: set[str] | None = None,
                          max_concurrent_positions: int = 1) -> list[dict]:
    watchlist = load_watchlist()
    rows: list[dict] = []

    for market, symbols in watchlist.items():
        for symbol in symbols:
            if symbol_filter and symbol not in symbol_filter:
                continue

            price_df = load_price_data(market, symbol)
            if price_df is None or price_df.empty:
                rows.append({"market": market, "symbol": symbol, "status": "無資料"})
                continue

            train_df, live_df = build_dataset(price_df, horizon=horizon, threshold=threshold)

            if len(train_df) < MIN_TRAIN_SAMPLES:
                rows.append({
                    "market": market, "symbol": symbol,
                    "status": f"資料不足（僅 {len(train_df)} 筆）",
                })
                continue

            # 1. Walk-forward 樣本外回測，算出勝率/賠率（訊號品質，不管部位重疊）
            #    apply_transaction_costs() 把手續費/交易稅扣進每筆交易的報酬率，
            #    讓勝率/賠率反映「真的能落袋」的數字，不是零成本的理論值
            predictions = walk_forward_predict(train_df)
            trades = simulate_strategy(predictions, prob_threshold=prob_threshold)
            trades = apply_transaction_costs(trades, market, symbol)
            metrics = compute_performance(trades)

            if metrics is None:
                rows.append({
                    "market": market, "symbol": symbol,
                    "status": f"回測期間沒有任何訊號觸發（機率門檻 {prob_threshold:.0%} 太高）",
                })
                continue

            # 1b. 用有限部位數重新模擬總報酬/最大回撤，才是「實際資金運用」的估計
            #     （見 backtest/engine.py 的說明：compute_performance 的總報酬/回撤
            #      假設每筆訊號都能同時全額進場，horizon 一長就會嚴重失真）
            all_dates = price_df.sort_values("date")["date"].reset_index(drop=True)
            portfolio = simulate_portfolio(
                trades, all_dates, horizon=horizon,
                max_concurrent_positions=max_concurrent_positions,
            )
            if portfolio is None:
                # 極端情況（例如部位額度一直被佔滿，一筆都擠不進去）才會落到這裡，
                # 保守 fallback：用 compute_performance 的天真數字，總比沒有好，
                # 但這種情況應該很罕見，值得留 log 觀察。
                logger.warning(f"[{market}/{symbol}] simulate_portfolio 沒有任何交易被接受，改用未考慮部位重疊的數字")
                portfolio_max_drawdown = metrics.max_drawdown
                portfolio_total_return = metrics.total_return
                n_trades_taken = metrics.n_trades
                n_signals_skipped = 0
            else:
                portfolio_max_drawdown = portfolio.max_drawdown
                portfolio_total_return = portfolio.total_return
                n_trades_taken = portfolio.n_trades
                n_signals_skipped = portfolio.n_skipped

            # 2. 用全部歷史資料訓練「現在」要用的模型，拿到今天的即時機率，
            #    同時檢查這個模型本身準不準（AUC），不可信的模型不管機率多高都不該進場
            model, cv_metrics = train_and_evaluate(train_df)
            latest = predict_latest(model, live_df)
            current_prob = float(latest.iloc[-1]["probability"]) if not latest.empty else None
            reliable = is_signal_reliable(cv_metrics)

            # 3. 用回測的勝率/賠率，幫「今天的訊號」算出凱利公式建議倉位
            #    （勝率賠率來自歷史回測，不是今天的機率——今天的機率只決定要不要進場）
            #    只有「機率達門檻」且「模型本身可信（AUC 夠高）」才會給出建議倉位
            kelly_result = None
            if current_prob is not None and current_prob >= prob_threshold and reliable:
                kelly_result = kelly_position_size(
                    capital=capital,
                    win_prob=metrics.win_rate,
                    payoff_ratio=metrics.payoff_ratio,
                    fraction=kelly_fraction,
                    max_position_pct=max_position_pct,
                )

            rows.append({
                "market": market, "symbol": symbol, "status": "正常",
                "n_trades": metrics.n_trades,  # 訊號品質評估用的交易數（勝率/賠率的樣本數）
                "win_rate": metrics.win_rate,
                "payoff_ratio": metrics.payoff_ratio,
                "max_drawdown": portfolio_max_drawdown,   # 有考慮部位重疊限制，比較真實
                "total_return": portfolio_total_return,    # 同上
                "n_trades_taken": n_trades_taken,          # 實際「接受」的交易數（部位額度限制後）
                "n_signals_skipped": n_signals_skipped,    # 因為部位已滿而跳過的訊號數
                "current_prob": current_prob,
                "reliable": reliable,
                "auc": cv_metrics.avg_auc if cv_metrics else None,
                "suggested_position_pct": kelly_result.capped_fraction if kelly_result else 0.0,
                "suggested_position_value": kelly_result.position_value if kelly_result else 0.0,
            })

    return rows


def print_report(rows: list[dict], capital: float, prob_threshold: float) -> None:
    print("\n" + "=" * 100)
    print(f"Phase 4 回測與資金管理報告（總資金 {capital:,.0f}，進場機率門檻 {prob_threshold:.0%}）")
    print("=" * 100)

    normal_rows = [r for r in rows if r["status"] == "正常"]
    other_rows = [r for r in rows if r["status"] != "正常"]

    if normal_rows:
        print(f"{'市場':<8}{'代號':<8}{'訊號數':<8}{'勝率':<8}{'風險報酬比':<12}{'已採用':<8}{'最大回撤':<10}{'累積報酬':<10}{'現在機率':<10}{'AUC':<8}{'建議倉位'}")
        print("（訊號數=勝率/賠率的樣本數；已採用=扣掉部位重疊限制後真的模擬進場的交易數，最大回撤/累積報酬是用這個算的）")
        print("-" * 100)
        for r in sorted(normal_rows, key=lambda x: -(x["suggested_position_value"])):
            prob_str = f"{r['current_prob']:.1%}" if r["current_prob"] is not None else "-"
            payoff_str = f"{r['payoff_ratio']:.2f}" if r["payoff_ratio"] == r["payoff_ratio"] else "-"  # NaN 檢查
            auc_str = f"{r['auc']:.3f}" if r["auc"] is not None and r["auc"] == r["auc"] else "-"

            if r["suggested_position_value"] > 0:
                position_str = f"{r['suggested_position_pct']:.1%}（約 {r['suggested_position_value']:,.0f}）"
            elif r["current_prob"] is not None and r["current_prob"] >= prob_threshold and not r["reliable"]:
                position_str = "不建議進場（訊號不可信）"
            else:
                position_str = "不建議進場（機率不足）"

            print(f"{r['market']:<8}{r['symbol']:<8}{r['n_trades']:<8}{r['win_rate']:<8.1%}{payoff_str:<12}"
                  f"{r['n_trades_taken']:<8}{r['max_drawdown']:<10.1%}{r['total_return']:<10.1%}{prob_str:<10}{auc_str:<8}{position_str}")

    if other_rows:
        print("\n以下標的無法產出完整回測：")
        for r in other_rows:
            print(f"  {r['market']}/{r['symbol']}: {r['status']}")

    small_positions = [
        r for r in normal_rows
        if 0 < r["suggested_position_value"] < SMALL_POSITION_WARNING_THRESHOLD
    ]
    if small_positions:
        names = "、".join(f"{r['market']}/{r['symbol']}" for r in small_positions)
        print(
            f"\n提醒：{names} 的建議倉位金額偏小（< {SMALL_POSITION_WARNING_THRESHOLD:,.0f}）。"
            "手續費模型只算比例式成本，沒算最低手續費下限——金額越小，最低手續費"
            "實際佔比就越高，這裡的勝率/賠率估計會偏樂觀，請自行對照你的券商規則。"
        )

    total_skipped = sum(r.get("n_signals_skipped", 0) for r in normal_rows)
    if total_skipped:
        print(f"\n（另外有 {total_skipped} 次訊號因為當時已經持有部位、額度滿了而被跳過，"
              f"沒有算進最大回撤/累積報酬——用 --max-concurrent-positions 可以調整同時最多持有幾筆）")

    print("\n提醒：以上勝率/賠率已扣除手續費與交易稅（比例式），來自歷史 Walk-forward 回測，不代表未來一定重演；")
    print("      最大回撤/累積報酬已考慮部位重疊限制，「建議倉位」是套用半凱利/你設定的折扣係數與單筆上限後的參考值，不是投資建議。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4：回測與資金管理報告")
    parser.add_argument("--capital", type=float, required=True, help="總資金（必填，例如 500000）")
    parser.add_argument("--horizon", type=int, default=5, help="預測未來幾天的報酬率（預設 5，須與訓練時一致）")
    parser.add_argument("--threshold", type=float, default=0.03, help="分類門檻，例如 0.03 代表 3%%（預設 0.03）")
    parser.add_argument("--prob-threshold", type=float, default=0.5, help="模型機率超過多少才算觸發訊號（預設 0.5）")
    parser.add_argument("--kelly-fraction", type=float, default=0.5, help="凱利公式打折係數，0.5=半凱利（預設 0.5）")
    parser.add_argument("--max-position", type=float, default=0.25, help="單筆倉位佔總資金上限（預設 0.25，即 25%%）")
    parser.add_argument("--max-concurrent-positions", type=int, default=1,
                         help="同一檔標的同時最多持有幾筆部位（預設 1，最保守：還沒平倉的話新訊號一律跳過）")
    parser.add_argument("--symbols", type=str, default=None, help="只跑指定標的，逗號分隔")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol_filter = set(args.symbols.split(",")) if args.symbols else None

    rows = run_backtest_report(
        capital=args.capital,
        horizon=args.horizon,
        threshold=args.threshold,
        prob_threshold=args.prob_threshold,
        kelly_fraction=args.kelly_fraction,
        max_position_pct=args.max_position,
        symbol_filter=symbol_filter,
        max_concurrent_positions=args.max_concurrent_positions,
    )
    print_report(rows, args.capital, args.prob_threshold)


if __name__ == "__main__":
    main()
