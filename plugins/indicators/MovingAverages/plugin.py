import pandas as pd
import numpy as np
import talib

def compute_MovingAverages(ohlcv_df: pd.DataFrame,
                           ma_type: str = "SMA",
                           periods: list = [14, 50, 100]) -> pd.DataFrame:
    """
    ohlcv_df: DataFrame с ['OpenTime', 'Close'] как минимум.
    ma_type:  "SMA" или "EMA".
    periods:  список целых периодов, например [14, 50, 100].

    Возвращает DataFrame с колонками:
      ['OpenTime', 'SMA_14_pos', 'SMA_50_pos', ...] 
      или ['OpenTime', 'EMA_14_pos', 'EMA_50_pos', ...] 
    где каждая колонка = +1, если Close > MA, иначе -1 (NaN→–1).
    """

    # ——— 1. Проверка входных данных —————————————————————————————————————————————————————————————
    if "OpenTime" not in ohlcv_df.columns or "Close" not in ohlcv_df.columns:
        raise ValueError("Нужны колонки 'OpenTime' и 'Close' в ohlcv_df.")

    df = pd.DataFrame({"OpenTime": ohlcv_df["OpenTime"].values})
    close = ohlcv_df["Close"].values

    # ——— 2. Проходим по каждому периоду и считаем MA ——————————————————————————————————————————
    for p in periods:
        if ma_type.upper() == "SMA":
            ma_values = talib.SMA(close, timeperiod=p)
            col_name = f"SMA_{p}_pos"
        elif ma_type.upper() == "EMA":
            ma_values = talib.EMA(close, timeperiod=p)
            col_name = f"EMA_{p}_pos"
        else:
            raise ValueError("ma_type должен быть 'SMA' или 'EMA'.")

        # Превращаем массив ma_values в +1/–1:
        # Если ma_values[i] < close[i] → +1, иначе –1. При NaN → –1.
        pos = np.where(
            (~np.isnan(ma_values)) & (close > ma_values),
            1, -1
        )
        df[col_name] = pos

    return df
