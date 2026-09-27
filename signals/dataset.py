"""
dataset.py
===========
把 features.py 的特徵矩陣跟 labels.py 的標籤對齊組合，
並切分成兩份：
- train_df：有標籤（歷史上已經知道結果）的資料，拿去訓練/驗證模型
- live_df：特徵齊全但標籤還是 NaN（因為未來還沒發生）的最新資料，
           這是我們真正想要模型輸出機率的「現在」

任何一列只要特徵裡有 NaN（通常是指標暖身期不足），就會被排除在 train_df 之外；
live_df 只保留「特徵齊全、但標籤未知」的列，通常就是最後 horizon 筆。
"""

from __future__ import annotations

import pandas as pd

from .data_quality import clean_price_data
from .features import FEATURE_COLUMNS, build_feature_matrix
from .labels import build_future_return, build_label


def build_dataset(price_df: pd.DataFrame, horizon: int = 5,
                    threshold: float = 0.03) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    回傳 (train_df, live_df)，兩者都包含 'date' + FEATURE_COLUMNS + 'label' + 'future_return' 欄位，
    差別只在 train_df 的 label 一定有值，live_df 的 label 一定是 NaN。

    'future_return' 是原始報酬率數值（不是分類標籤），分類模型訓練時不會用到這欄，
    但 Phase 4 的回測引擎需要它來計算每筆交易實際賺賠多少。

    在計算特徵跟標籤之前，會先用 data_quality.clean_price_data() 過濾掉不合理的
    單日價格跳動，避免一筆髒資料同時污染技術指標特徵跟未來報酬率標籤。
    """
    price_df = clean_price_data(price_df)
    features = build_feature_matrix(price_df)
    label = build_label(price_df, horizon=horizon, threshold=threshold)
    future_return = build_future_return(price_df, horizon=horizon)

    combined = features.copy()
    combined["label"] = label.values
    combined["future_return"] = future_return.values

    features_complete = combined[FEATURE_COLUMNS].notna().all(axis=1)

    train_df = combined[features_complete & combined["label"].notna()].reset_index(drop=True)
    live_df = combined[features_complete & combined["label"].isna()].reset_index(drop=True)

    return train_df, live_df
