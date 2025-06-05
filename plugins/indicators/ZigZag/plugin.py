# plugins/indicators/ZigZag/plugin.py

import pandas as pd
import numpy as np
import talib

def compute_ZigZag(
    ohlcv_df: pd.DataFrame,
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    ohlcv_df: DataFrame со столбцами ['OpenTime','Open','High','Low','Close','Volume'].
    threshold: порог в долях (например, 0.05 = 5%) для построения ZigZag.

    Возвращает DataFrame с колонками:
      ['OpenTime', 'ZigZag_low', 'ZigZag_high'], где:
        ZigZag_low  = значение закрытого локального минимума (int64 мс), иначе 0
        ZigZag_high = значение закрытого локального максимума (int64 мс), иначе 0
    """

    # Проверка входных данных
    required_cols = {"OpenTime", "High", "Low", "Close"}
    if not required_cols.issubset(ohlcv_df.columns):
        raise ValueError(f"В ohlcv_df не хватает колонок {required_cols}.")

    # Будем хранить результаты: по каждой строке ts мы запишем либо 0, либо int64-метку времени
    n = len(ohlcv_df)
    zigzag_low  = np.zeros(n, dtype="int64")
    zigzag_high = np.zeros(n, dtype="int64")

    close_arr = ohlcv_df["Close"].values
    high_arr  = ohlcv_df["High"].values
    low_arr   = ohlcv_df["Low"].values
    times_arr = ohlcv_df["OpenTime"].values.astype("int64")  # предполагаем, что это int64

    # Используем talib.ZIG (возвращает точки ZigZag как значения цен; все остальные — NaN)
    zz = talib.ZIG(close_arr, high_arr, low_arr, threshold)

    # Найдём индексы, где в zz появляются «точки» (локальные экстремумы):
    # Если zz[i] == low_arr[i] → локальный минимум
    # Если zz[i] == high_arr[i] → локальный максимум
    # Всё остальное — NaN
    for i in range(n):
        if not np.isnan(zz[i]):
            if np.isclose(zz[i], low_arr[i], atol=1e-8):
                # локальный минимум — запишем int64-метку времени
                zigzag_low[i] = times_arr[i]
            elif np.isclose(zz[i], high_arr[i], atol=1e-8):
                # локальный максимум — запишем int64-метку времени
                zigzag_high[i] = times_arr[i]
            # иначе (теоретически не должно быть), оставляем 0

    # Собираем результирующий DataFrame
    df_out = pd.DataFrame({
        "OpenTime":    times_arr,       # int64-метка времени (ms)
        "ZigZag_low":  zigzag_low,      # int64-метка (ms) локального минимума или 0
        "ZigZag_high": zigzag_high      # int64-метка (ms) локального максимума или 0
    })

    return df_out
