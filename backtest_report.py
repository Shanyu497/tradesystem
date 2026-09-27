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
"""

from __future__ import annotations

import argparse
import logging

from backtest.engine import compute_performance, simulate_strategy, walk_forward_predict
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


def run_backtest_report(capital: float, horizon: int, threshold: float,
                          prob_threshold: float, kelly_fraction: float,
                          max_position_pct: float,
                          symbol_filter: set[str] | None = None) -> list[dict]:
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

            # 1. Walk-forward 樣本外回測，算出勝率/賠率/最大回撤
            predictions = walk_forward_predict(train_df)
            trades = simulate_strategy(predictions, prob_threshold=prob_threshold)
            metrics = compute_performance(trades)

            if metrics is None:
                rows.append({
                    "market": market, "symbol": symbol,
                    "status": f"回測期間沒有任何訊號觸發（機率門檻 {prob_threshold:.0%} 太高）",
                })
                continue

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
                "n_trades": metrics.n_trades,
                "win_rate": metrics.win_rate,
                "payoff_ratio": metrics.payoff_ratio,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
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
        print(f"{'市場':<6}{'代號':<8}{'交易數':<8}{'勝率':<8}{'風險報酬比':<12}{'最大回撤':<10}{'累積報酬':<10}{'現在機率':<10}{'AUC':<8}{'建議倉位'}")
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

            print(f"{r['market']:<6}{r['symbol']:<8}{r['n_trades']:<8}{r['win_rate']:<8.1%}{payoff_str:<12}"
                  f"{r['max_drawdown']:<10.1%}{r['total_return']:<10.1%}{prob_str:<10}{auc_str:<8}{position_str}")

    if other_rows:
        print("\n以下標的無法產出完整回測：")
        for r in other_rows:
            print(f"  {r['market']}/{r['symbol']}: {r['status']}")

    print("\n提醒：以上勝率/賠率來自歷史 Walk-forward 回測，不代表未來一定重演；")
    print("      「建議倉位」是套用半凱利/你設定的折扣係數與單筆上限後的參考值，不是投資建議。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4：回測與資金管理報告")
    parser.add_argument("--capital", type=float, required=True, help="總資金（必填，例如 500000）")
    parser.add_argument("--horizon", type=int, default=5, help="預測未來幾天的報酬率（預設 5，須與訓練時一致）")
    parser.add_argument("--threshold", type=float, default=0.03, help="分類門檻，例如 0.03 代表 3%%（預設 0.03）")
    parser.add_argument("--prob-threshold", type=float, default=0.5, help="模型機率超過多少才算觸發訊號（預設 0.5）")
    parser.add_argument("--kelly-fraction", type=float, default=0.5, help="凱利公式打折係數，0.5=半凱利（預設 0.5）")
    parser.add_argument("--max-position", type=float, default=0.25, help="單筆倉位佔總資金上限（預設 0.25，即 25%%）")
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
    )
    print_report(rows, args.capital, args.prob_threshold)


if __name__ == "__main__":
    main()
