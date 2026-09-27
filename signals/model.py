"""
model.py
=========
訓練分類模型並輸出機率，搭配 Walk-forward（時間序列）交叉驗證來評估「這個模型
是不是真的有效」，而不是只看訓練集準確率（那沒有意義，模型可以直接背答案）。

為什麼用 HistGradientBoostingClassifier 而不是 xgboost/lightgbm？
- sklearn 內建，不需要額外安裝、不需要編譯，在任何 Python 版本上都能裝
  （xgboost 在很新的 Python 版本上常常還沒有對應的預編譯 wheel，容易裝不起來）。
- 效果跟 xgboost/lightgbm 在大多數表格資料任務上相差不大，之後真的需要更極致的
  效能時，再評估要不要換，介面設計上也預留了容易替換的空間（train_model 只回傳
  一個有 predict_proba 方法的物件，換模型不影響呼叫端程式碼）。

樣本數過少時的處理：
- 金融時間序列資料量本來就有限（一年才 250 個交易日），如果訓練樣本不足以支撐
  合理的 Walk-forward 切分，會回傳 None 的 metrics 並提示需要更多歷史資料，
  而不是硬跑一個沒有統計意義的結果誤導你。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .features import FEATURE_COLUMNS

MIN_SAMPLES_FOR_CV = 100  # 少於這個樣本數，Walk-forward 驗證的結果不具參考價值
MIN_RELIABLE_AUC = 0.55   # AUC 低於這個門檻，視為模型排序能力跟隨機猜測沒有顯著差異，
                            # 不管機率數字看起來多高，都不該當作進場依據
                            # （0.5 是完全隨機，這裡抓 0.55 是留一點緩衝，不是卡在剛好 0.5）


@dataclass
class EvalMetrics:
    n_splits: int
    avg_precision: float  # 模型判斷「會漲」時，實際真的漲的比例（對應之後 Kelly 公式的勝率 p）
    avg_auc: float         # 模型排序能力的整體指標，0.5 代表跟隨機猜一樣
    avg_positive_rate: float  # 模型判斷「會漲」的訊號佔全部樣本的比例（訊號多不多）


def train_and_evaluate(train_df: pd.DataFrame, n_splits: int = 3) -> tuple[object, Optional[EvalMetrics]]:
    """
    回傳 (最終模型, 評估指標)。最終模型是用「全部」歷史資料訓練的，
    評估指標則是用 Walk-forward（只用過去資料預測未來，不偷看未來）算出來的，
    避免用同一批資料「訓練也用它、驗證也用它」造成過度樂觀的假象。
    """
    X = train_df[FEATURE_COLUMNS]
    y = train_df["label"]

    metrics: Optional[EvalMetrics] = None

    if len(train_df) >= MIN_SAMPLES_FOR_CV:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        precisions, aucs, positive_rates = [], [], []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if y_train.nunique() < 2:
                # 這個切分裡全部都是同一個類別（例如都沒漲），沒辦法訓練分類器，跳過
                continue

            fold_model = HistGradientBoostingClassifier(random_state=42)
            fold_model.fit(X_train, y_train)
            proba = fold_model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)

            if pred.sum() > 0:
                precisions.append(precision_score(y_test, pred, zero_division=0))
            if y_test.nunique() >= 2:
                aucs.append(roc_auc_score(y_test, proba))
            positive_rates.append(pred.mean())

        if precisions or aucs:
            metrics = EvalMetrics(
                n_splits=len(positive_rates),
                avg_precision=float(np.mean(precisions)) if precisions else float("nan"),
                avg_auc=float(np.mean(aucs)) if aucs else float("nan"),
                avg_positive_rate=float(np.mean(positive_rates)) if positive_rates else float("nan"),
            )

    # 最終部署用的模型：用「全部」歷史資料訓練，這是實際拿去 predict 最新資料的模型
    final_model = HistGradientBoostingClassifier(random_state=42)
    final_model.fit(X, y)

    return final_model, metrics


def predict_latest(model, live_df: pd.DataFrame) -> pd.DataFrame:
    """
    對 live_df（特徵齊全但標籤未知的最新資料）輸出機率。
    回傳只保留 'date' 和 'probability' 欄位，probability 就是之後 Kelly 公式要用的勝率 p。
    """
    if live_df.empty:
        return pd.DataFrame(columns=["date", "probability"])

    X_live = live_df[FEATURE_COLUMNS]
    proba = model.predict_proba(X_live)[:, 1]
    return pd.DataFrame({"date": live_df["date"], "probability": proba})


def is_signal_reliable(metrics: Optional[EvalMetrics]) -> bool:
    """
    判斷這個模型的機率輸出「值不值得信」，跟機率本身高低是兩件事：
    機率決定「要不要進場」，這個函式決定「這次判斷有沒有參考價值」。

    metrics 為 None（樣本數不足，沒有算過 CV）或 avg_auc 是 NaN（無法計算）
    或 avg_auc 低於 MIN_RELIABLE_AUC，都視為不可信。
    """
    if metrics is None:
        return False
    if metrics.avg_auc != metrics.avg_auc:  # NaN 檢查（NaN 是唯一「不等於自己」的浮點數）
        return False
    return metrics.avg_auc >= MIN_RELIABLE_AUC
