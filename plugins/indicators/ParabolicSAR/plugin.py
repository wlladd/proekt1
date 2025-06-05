# plugins/indicators/ParabolicSAR/plugin.py

import pandas as pd
import numpy as np
import talib

def resample_higher_tf(ohlcv_df: pd.DataFrame, higher_tf: str) -> pd.DataFrame:
    """
    Пересемпливает ohlcv_df (младший TF) в старший TF, заданный строкой higher_tf.
    Формат higher_tf: "5T", "15T", "1H", "4H", "1D" и т. д.
    Возвращает DataFrame с ['OpenTime','Open','High','Low','Close','Volume'] для старшего TF.
    """
    df = ohlcv_df.copy()
    # Переводим OpenTime (ms) в datetime
    df["dt"] = pd.to_datetime(df["OpenTime"], unit="ms")
    df.set_index("dt", inplace=True)

    # Агрегируем по higher_tf
    df_higher = df.resample(higher_tf).agg({
        "Open":  "first",
        "High":  "max",
        "Low":   "min",
        "Close": "last",
        "Volume":"sum"
    })
    # Удаляем «пустые» интервалы без сделок
    df_higher = df_higher.dropna(subset=["Open", "Close"])

    # Восстанавливаем индекс dt → колонка и переводим dt обратно в OpenTime (ms)
    df_higher = df_higher.reset_index()
    df_higher["OpenTime"] = (df_higher["dt"].astype("int64") // 1_000_000).astype("int64")

    return df_higher[["OpenTime", "Open", "High", "Low", "Close", "Volume"]]

def compute_ParabolicSAR(
    ohlcv_df: pd.DataFrame,
    higher_tf: str = "5T",
    af_step: float = 0.02,
    af_max: float = 0.2
) -> pd.DataFrame:
    """
    ohlcv_df:   DataFrame со столбцами ['OpenTime','Open','High','Low','Close','Volume']
                (младший таймфрейм, например 1 минута).
    higher_tf:  строка-период для ресемплинга в старший TF, например '5T', '15T', '1H' и т. д.
    af_step:    шаг ускорения (acceleration factor) для PSAR (по умолчанию 0.02).
    af_max:     максимальный AF (по умолчанию 0.2).

    Возвращает DataFrame с колонками:
      ['OpenTime', 'PSAR_current_pos', 'PSAR_higher_pos'], где:
        PSAR_current_pos = +1, если PSAR на младшем TF < Close текущей свечи, иначе –1.
        PSAR_higher_pos  = +1, если PSAR на старшем TF < Close старшей свечи, иначе –1.
    """

    # Проверка входных данных
    required_cols = {"OpenTime", "High", "Low", "Close"}
    if not required_cols.issubset(ohlcv_df.columns):
        raise ValueError(f"В ohlcv_df не хватает колонок {required_cols}.")

    # 1) Считаем PSAR на младшем TF
    high = ohlcv_df["High"].values
    low  = ohlcv_df["Low"].values
    psar_current = talib.SAR(high, low, acceleration=af_step, maximum=af_max)
    df_current = pd.DataFrame({
        "OpenTime":     ohlcv_df["OpenTime"].values,
        "PSAR_current": psar_current
    })

    # 2) Ресемплим и считаем PSAR на старшем TF
    higher_df = resample_higher_tf(ohlcv_df, higher_tf)
    high_h = higher_df["High"].values
    low_h  = higher_df["Low"].values
    psar_higher = talib.SAR(high_h, low_h, acceleration=af_step, maximum=af_max)
    df_higher = pd.DataFrame({
        "OpenTime":    higher_df["OpenTime"].values,
        "PSAR_higher": psar_higher
    })

    # 3) Объединяем результаты с ohlcv_df по OpenTime
    df_result = ohlcv_df[["OpenTime", "Close"]].copy()
    df_result = df_result.merge(df_current, on="OpenTime", how="left")
    df_result = df_result.merge(df_higher, on="OpenTime", how="left")

    # 4) Преобразуем PSAR-значения в позиции +1/–1
    def position(psar_arr: np.ndarray, close_arr: np.ndarray) -> np.ndarray:
        pos = np.zeros_like(close_arr, dtype=int)
        mask_valid = ~np.isnan(psar_arr)
        pos[mask_valid] = np.where(psar_arr[mask_valid] < close_arr[mask_valid], 1, -1)
        return pos

    arr_close = df_result["Close"].values
    arr_psar_c = df_result["PSAR_current"].values
    arr_psar_h = df_result["PSAR_higher"].values

    df_result["PSAR_current_pos"] = position(arr_psar_c, arr_close)
    df_result["PSAR_higher_pos"]  = position(arr_psar_h, arr_close)

    # 5) Возвращаем только нужные столбцы
    output = df_result[["OpenTime", "PSAR_current_pos", "PSAR_higher_pos"]].copy()
    return output
