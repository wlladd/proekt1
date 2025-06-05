# plugins/indicators/CandlestickPatterns/plugin.py

import pandas as pd
import numpy as np
import talib

def compute_CandlestickPatterns(
    ohlcv_df: pd.DataFrame,
    patterns: list = ["CDLDOJI", "CDLHAMMER", "CDLENGULFING"]
) -> pd.DataFrame:
    """
    ohlcv_df:   DataFrame с колонками ['OpenTime', 'Open', 'High', 'Low', 'Close'].
    patterns:   список строковых имён функций TA-Lib для свечных паттернов, 
                например ["CDLDOJI", "CDLHAMMER"].

    Возвращает DataFrame с колонками:
      ['OpenTime', '<pattern1>_pos', '<pattern2>_pos', ...],
    где <pattern>_pos = +1 (бычий), –1 (медвежий), 0 (нет паттерна).
    """

    # Проверка входных данных
    required_cols = {"OpenTime", "Open", "High", "Low", "Close"}
    if not required_cols.issubset(ohlcv_df.columns):
        raise ValueError(f"В ohlcv_df не хватает колонок {required_cols}.")

    # Проверяем, что все указанные паттерны действительно есть в TA-Lib
    for pat in patterns:
        if pat not in talib.get_functions():
            raise ValueError(f"Функция TA-Lib '{pat}' не найдена.")

    df_out = pd.DataFrame({"OpenTime": ohlcv_df["OpenTime"].values})

    open_arr  = ohlcv_df["Open"].values
    high_arr  = ohlcv_df["High"].values
    low_arr   = ohlcv_df["Low"].values
    close_arr = ohlcv_df["Close"].values

    for pat in patterns:
        func = getattr(talib, pat)
        raw = func(open_arr, high_arr, low_arr, close_arr)

        # raw > 0 → +1, raw < 0 → –1, raw == 0 → 0
        pos = np.zeros_like(raw, dtype=int)
        pos[raw > 0] = 1
        pos[raw < 0] = -1

        col_name = f"{pat}_pos"
        df_out[col_name] = pos

    return df_out
