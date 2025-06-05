import pandas as pd
import numpy as np
import talib

def compute_RSI_Custom(
    ohlcv_df: pd.DataFrame,
    rsi_period: int = 14,
    diverge_period: int = 20,
    upper_level: int = 80,
    lower_level: int = 20
) -> pd.DataFrame:
    """
    ohlcv_df:       DataFrame с колонками ['OpenTime', 'High', 'Low', 'Close'].
    rsi_period:     период расчёта RSI (обычно 14).
    diverge_period: период, за который ищем дивергенцию (обычно 20).
    upper_level:    уровень перекупленности (обычно 80).
    lower_level:    уровень перепроданности (обычно 20).

    Возвращает DataFrame с колонками:
      ['OpenTime',
       'RSI_trend',         # +1, если RSI[i] > RSI[i-1], иначе -1 (NaN→–1)
       'RSI_80',            # +1, если RSI > upper_level, иначе -1
       'RSI_20',            # +1, если RSI > lower_level, иначе -1
       'RSI_divergence']    # +1 бычья див., –1 медвежья, 0 иначе
    """

    # ——— 1. Проверка входных данных —————————————————————————————————————————————————————————————
    required_cols = {"OpenTime", "High", "Low", "Close"}
    if not required_cols.issubset(ohlcv_df.columns):
        raise ValueError(f"В ohlcv_df не хватает колонок {required_cols}.")

    close = ohlcv_df["Close"].values
    high  = ohlcv_df["High"].values
    low   = ohlcv_df["Low"].values

    # ——— 2. Рассчитываем RSI через TA-Lib ——————————————————————————————————————————————————————
    rsi = talib.RSI(close, timeperiod=rsi_period)  # длина = len(ohlcv_df), первые rsi_period-1 будут NaN

    n = len(ohlcv_df)
    df = pd.DataFrame({
        "OpenTime": ohlcv_df["OpenTime"].values,
        "RSI":       rsi,
        "High":      high,
        "Low":       low
    })

    # ——— 3. Столбец RSI_trend: +1, если RSI > предыдущего, иначе –1 ——————————————————————————
    RSI_trend = np.zeros(n, dtype=int)
    RSI_trend[0] = -1
    for i in range(1, n):
        if np.isnan(df["RSI"].iat[i]) or np.isnan(df["RSI"].iat[i-1]):
            RSI_trend[i] = -1
        else:
            RSI_trend[i] = 1 if df["RSI"].iat[i] > df["RSI"].iat[i-1] else -1

    df["RSI_trend"] = RSI_trend

    # ——— 4. Пересечение уровней 80 и 20 ——————————————————————————————————————————————————————
    # RSI_80: +1, если RSI > upper_level, иначе –1 (NaN → –1)
    df["RSI_80"] = np.where((~np.isnan(df["RSI"].values)) & (df["RSI"].values > upper_level), 1, -1)
    # RSI_20: +1, если RSI > lower_level, иначе –1 (NaN → –1)
    df["RSI_20"] = np.where((~np.isnan(df["RSI"].values)) & (df["RSI"].values > lower_level), 1, -1)

    # ——— 5. Дивергенция за период diverge_period —————————————————————————————————————————————
    divergence = np.zeros(n, dtype=int)

    for i in range(diverge_period, n):
        # Окно [i-diverge_period, i]
        window_slice = slice(i - diverge_period, i + 1)
        lows_prices  = df["Low"].values[window_slice]
        highs_prices = df["High"].values[window_slice]
        values_rsi   = df["RSI"].values[window_slice]

        # Если в окне RSI полностью NaN – пропускаем
        if np.all(np.isnan(values_rsi)):
            divergence[i] = 0
            continue

        # Находим индексы минимума Low и минимума RSI в окне
        idx_price_low = np.nanargmin(lows_prices)
        idx_RSI_low   = np.nanargmin(values_rsi)

        # Находим индексы максимума High и максимума RSI в окне
        idx_price_high = np.nanargmax(highs_prices)
        idx_RSI_high   = np.nanargmax(values_rsi)

        # Пересчитываем индексы относительно всего DataFrame
        abs_idx_price_low  = i - diverge_period + idx_price_low
        abs_idx_RSI_low    = i - diverge_period + idx_RSI_low
        abs_idx_price_high = i - diverge_period + idx_price_high
        abs_idx_RSI_high   = i - diverge_period + idx_RSI_high

        # — Проверяем бычью дивергенцию:
        # Цена делает более низкий минимум, а RSI – более высокий минимум:
        if (not np.isnan(df["Low"].iat[abs_idx_price_low]) and
            not np.isnan(df["Low"].iat[abs_idx_RSI_low])   and
            not np.isnan(df["RSI"].iat[abs_idx_price_low])  and
            not np.isnan(df["RSI"].iat[abs_idx_RSI_low])    and
            (df["Low"].iat[abs_idx_price_low] < df["Low"].iat[abs_idx_RSI_low]) and
            (df["RSI"].iat[abs_idx_price_low] > df["RSI"].iat[abs_idx_RSI_low])):
            divergence[i] = 1
            continue  # если нашли бычью дивергенцию, дальше не проверяем медвежью

        # — Проверяем медвежью дивергенцию:
        # Цена делает более высокий максимум, а RSI – более низкий максимум:
        if (not np.isnan(df["High"].iat[abs_idx_price_high]) and
            not np.isnan(df["High"].iat[abs_idx_RSI_high])   and
            not np.isnan(df["RSI"].iat[abs_idx_price_high])  and
            not np.isnan(df["RSI"].iat[abs_idx_RSI_high])    and
            (df["High"].iat[abs_idx_price_high] > df["High"].iat[abs_idx_RSI_high]) and
            (df["RSI"].iat[abs_idx_price_high] < df["RSI"].iat[abs_idx_RSI_high])):
            divergence[i] = -1
        else:
            divergence[i] = 0

    df["RSI_divergence"] = divergence

    # ——— 6. Формируем итоговый DataFrame с нужными столбцами ————————————————————————————————————
    output = df[[
        "OpenTime",
        "RSI_trend",
        "RSI_80",
        "RSI_20",
        "RSI_divergence"
    ]].copy()

    return output
