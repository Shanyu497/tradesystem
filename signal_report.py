"""
signal_report.py
=================
Phase 3 的主入口：對監控清單裡每一檔標的，讀取 Phase 2 存好的 processed 價量資料，
組出訓練集、訓練分類模型、算出「目前」這一筆資料的上漲機率，並印出整份報告。

用法：
    py signal_report.py                        # 用預設參數（未來 5 天漲幅 > 3% 分類）跑全部標的
    py signal_report.py --horizon 10 --threshold 0.05   # 改成「未來 10 天漲幅 > 5%」
    py signal_report.py --symbols 2330,AAPL    # 只跑指定標的

重要前提：這個模組需要「足夠長」的歷史資料才有意義。技術指標暖身期最長要 50 個交易日，
再扣掉最近 horizon 天沒有標籤、再扣掉 Walk-forward 驗證要切好幾份，
建議先用 `py pipeline.py --days 730`（兩年資料）把歷史資料補齊，
再跑這支程式，不然大概率會看到「資料不足，無法訓練」的提示。
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import pandas as pd

from pipeline import load_watchlist
from signals.dataset import build_dataset
from signals.model import is_signal_reliable, train_and_evaluate, predict_latest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("signal_report")

PROJECT_ROOT = Path(__file__).resolve().parent
MIN_TRAIN_SAMPLES = 100  # 跟 signals/model.py 的 MIN_SAMPLES_FOR_CV 一致，用來決定要不要嘗試訓練


def load_price_data(market: str, symbol: str) -> pd.DataFrame | None:
    path = PROJECT_ROOT / "data" / "processed" / market / "price" / f"{symbol}.parquet"
    if not path.exists():
        logger.warning(f"[{market}/{symbol}] 找不到價量資料: {path}（先執行 pipeline.py 抓取資料）")
        return None
    return pd.read_parquet(path)


def run_report(horizon: int, threshold: float, symbol_filter: set[str] | None = None) -> list[dict]:
    watchlist = load_watchlist()
    rows: list[dict] = []

    for market, symbols in watchlist.items():
        for symbol in symbols:
            if symbol_filter and symbol not in symbol_filter:
                continue

            price_df = load_price_data(market, symbol)
            if price_df is None or price_df.empty:
                rows.append({
                    "market": market, "symbol": symbol,
                    "status": "無資料", "train_samples": 0,
                    "probability": None, "precision": None, "auc": None,
                })
                continue

            train_df, live_df = build_dataset(price_df, horizon=horizon, threshold=threshold)

            if len(train_df) < MIN_TRAIN_SAMPLES:
                rows.append({
                    "market": market, "symbol": symbol,
                    "status": f"資料不足（僅 {len(train_df)} 筆，建議至少 {MIN_TRAIN_SAMPLES} 筆）",
                    "train_samples": len(train_df),
                    "probability": None, "precision": None, "auc": None,
                })
                continue

            model, metrics = train_and_evaluate(train_df)
            latest = predict_latest(model, live_df)

            latest_prob = float(latest.iloc[-1]["probability"]) if not latest.empty else None
            reliable = is_signal_reliable(metrics)

            rows.append({
                "market": market, "symbol": symbol,
                "status": "正常",
                "train_samples": len(train_df),
                "probability": latest_prob,
                "precision": metrics.avg_precision if metrics else None,
                "auc": metrics.avg_auc if metrics else None,
                "reliable": reliable,
            })

    return rows


def print_report(rows: list[dict], horizon: int, threshold: float) -> None:
    print("\n" + "=" * 88)
    print(f"訊號報告（預測目標：未來 {horizon} 天報酬率 > {threshold:.1%}）")
    print("=" * 88)
    print(f"{'市場':<6}{'代號':<8}{'訓練樣本':<10}{'上漲機率':<12}{'驗證精準率':<12}{'AUC':<8}{'可信度':<8}{'狀態'}")
    print("-" * 88)

    def sort_key(r: dict) -> tuple:
        # 可信的訊號排最前面，同樣可信度的話機率高的排前面；不可信/無資料的排最後
        reliable = r.get("reliable", False)
        prob = r["probability"] or 0
        return (not reliable, -(prob if reliable else 0), r["probability"] is None)

    for r in sorted(rows, key=sort_key):
        prob_str = f"{r['probability']:.1%}" if r["probability"] is not None else "-"
        precision_str = f"{r['precision']:.1%}" if r["precision"] is not None and not math.isnan(r["precision"]) else "-"
        auc_str = f"{r['auc']:.3f}" if r["auc"] is not None and not math.isnan(r["auc"]) else "-"
        reliability_str = "可信" if r.get("reliable") else "不可信"
        print(f"{r['market']:<6}{r['symbol']:<8}{r['train_samples']:<10}{prob_str:<12}{precision_str:<12}{auc_str:<8}{reliability_str:<8}{r['status']}")

    reliable_rows = [r for r in rows if r.get("reliable") and r["probability"] is not None]
    if reliable_rows:
        best = max(reliable_rows, key=lambda x: x["probability"])
        print(f"\n可信且上漲機率最高的標的：{best['symbol']}（機率 {best['probability']:.1%}，AUC {best['auc']:.3f}）")
    else:
        print("\n目前沒有任何標的的訊號被判定為可信（AUC 未達門檻），不建議依這份報告進場。")

    # 特別標出「機率很高但不可信」的標的，這種最容易被誤判成好訊號，最需要提醒
    misleading = [r for r in rows if not r.get("reliable") and r["probability"] is not None and r["probability"] >= 0.5]
    if misleading:
        names = "、".join(f"{r['symbol']}（{r['probability']:.1%}）" for r in misleading)
        print(f"\n注意：{names} 的機率數字雖然不低，但 AUC 顯示模型對這些標的的判斷力跟隨機猜測沒有顯著差異，機率數字不該被當作訊號。")

    insufficient = [r for r in rows if r["train_samples"] < MIN_TRAIN_SAMPLES]
    if insufficient:
        print(f"\n有 {len(insufficient)} 檔標的資料量不足，建議先執行 py pipeline.py --days 730 補齊歷史資料。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="訊號生成報告")
    parser.add_argument("--horizon", type=int, default=5, help="預測未來幾天的報酬率（預設 5）")
    parser.add_argument("--threshold", type=float, default=0.03, help="漲幅門檻，例如 0.03 代表 3%%（預設 0.03）")
    parser.add_argument("--symbols", type=str, default=None, help="只跑指定標的，逗號分隔")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol_filter = set(args.symbols.split(",")) if args.symbols else None

    rows = run_report(args.horizon, args.threshold, symbol_filter)
    print_report(rows, args.horizon, args.threshold)


if __name__ == "__main__":
    main()
