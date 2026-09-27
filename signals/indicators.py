"""
indicators.py
==============
純用 pandas/numpy 手刻的技術指標函式庫。

為什麼不用 pandas-ta 或 ta-lib？
- ta-lib 需要編譯 C 擴充套件，Windows 上常常裝不起來。
- pandas-ta 在新版 numpy（2.0+）上有已知的相容性問題
  （它內部寫死 `from numpy import NaN`，新版 numpy 已經把 NaN 改成 nan，會直接 ImportError）。
- 這裡的指標都是通用公式，自己刻不到 100 行，換來的是「不會被套件版本鎖死」。

所有函式都接受單一 pd.Series（收盤價等）或整個 OHLCV DataFrame，
回傳一個或多個 pd.Series，方便組進 features.py 的特徵矩陣。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """簡單移動平均"""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """指數移動平均"""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    相對強弱指標（RSI），用 Wilder's smoothing（等同 alpha=1/window 的 EWM）,
    這是最標準的 RSI 計算方式，跟大多數看盤軟體算出來的數字會一致。
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50)  # avg_loss 為 0 時代表連續上漲，RSI 應趨近 100，但先用中性值 50 避免極端值汙染訓練


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """回傳 (macd_line, signal_line, histogram)"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """回傳 (percent_b, bandwidth, mid)。percent_b: 目前價格在通道中的相對位置（0~1，>1 代表突破上緣）"""
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std

    band_range = (upper - lower).replace(0, np.nan)
    percent_b = (series - lower) / band_range
    bandwidth = band_range / mid.replace(0, np.nan)
    return percent_b, bandwidth, mid


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """平均真實區間（ATR），衡量波動度，用 Wilder's smoothing"""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """今日量能相對於過去 N 日均量的倍數，>1.5 通常代表有異常關注"""
    avg_volume = df["volume"].rolling(window=window, min_periods=window).mean()
    return df["volume"] / avg_volume.replace(0, np.nan)


def momentum(series: pd.Series, window: int) -> pd.Series:
    """N 日報酬率（動能）"""
    return series.pct_change(periods=window)


def channel_position(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    價格在最近 N 日高低區間中的相對位置，1.0 代表收在區間最高點附近（潛在突破），
    0.0 代表收在區間最低點附近（潛在破底）。
    """
    rolling_high = df["high"].rolling(window=window, min_periods=window).max()
    rolling_low = df["low"].rolling(window=window, min_periods=window).min()
    channel_range = (rolling_high - rolling_low).replace(0, np.nan)
    return (df["close"] - rolling_low) / channel_range


def is_new_high_breakout(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    是否創下「不含今日」的最近 N 日新高（用 shift(1) 避免用到今天的高點跟自己比較）。
    這是最直接對應「高勝率突破」這個目標的特徵：布林值 1/0。
    """
    prior_high = df["high"].shift(1).rolling(window=window, min_periods=window).max()
    return (df["close"] > prior_high).astype(float)
