import pandas as pd
import numpy as np
import talib

def compute_ADX_DI(ohlcv_df: pd.DataFrame,
                   timeperiod: int = 14) -> pd.DataFrame:
    """
    ohlcv_df:   DataFrame с колонками ['OpenTime', 'High', 'Low', 'Close'].
    timeperiod: период для DI (обычно 14).

    Возвращает DataFrame с колонками:
      ['OpenTime',
       'DIp_trend',      # +1 если +DI > предыдущего, иначе -1
       'DIn_trend',      # +1 если -DI > предыдущего, иначе -1
       'DIp_over_DIn',   # +1 если +DI > -DI, иначе -1
       'DIp_turn',       # 1 при первой смене направления +DI, иначе 0
       'DIn_turn']       # 1 при первой смене направления -DI, иначе 0
    """

    # ——— 1. Проверка входных данных —————————————————————————————————————————————————————————————
    required_cols = {"OpenTime", "High", "Low", "Close"}
    if not required_cols.issubset(ohlcv_df.columns):
        raise ValueError(f"В ohlcv_df не хватает колонок {required_cols}.")

    # ——— 2. Вычисляем +DI и -DI через TA-Lib ————————————————————————————————————————————————
    high = ohlcv_df["High"].values
    low  = ohlcv_df["Low"].values
    close= ohlcv_df["Close"].values

    # TA-Lib возвращает массивы длины len(ohlcv_df), первые timeperiod−1 будут NaN
    di_plus  = talib.PLUS_DI(high, low, close, timeperiod=timeperiod)
    di_minus = talib.MINUS_DI(high, low, close, timeperiod=timeperiod)

    df = pd.DataFrame({
        "OpenTime": ohlcv_df["OpenTime"].values,
        "di_plus":  di_plus,
        "di_minus": di_minus
    })

    # ——— 3. Строим трендовые столбцы (рост/падение для di_plus и di_minus) ——————————————————
    # DIp_trend[i] = +1, если di_plus[i] > di_plus[i-1], иначе -1 (включая NaN -> -1).
    # Аналогично для DIn_trend.
    n = len(df)
    DIp_trend = np.zeros(n, dtype=int)
    DIn_trend = np.zeros(n, dtype=int)

    for i in range(1, n):
        # Если текущий или предыдущий NaN – считаем тренд «–1»
        if np.isnan(df["di_plus"].iat[i]) or np.isnan(df["di_plus"].iat[i-1]):
            DIp_trend[i] = -1
        else:
            DIp_trend[i] = 1 if df["di_plus"].iat[i] > df["di_plus"].iat[i-1] else -1

        if np.isnan(df["di_minus"].iat[i]) or np.isnan(df["di_minus"].iat[i-1]):
            DIn_trend[i] = -1
        else:
            DIn_trend[i] = 1 if df["di_minus"].iat[i] > df["di_minus"].iat[i-1] else -1

    # Первый бар (i=0) объявляем –1 (нет предыдущего), либо можно сделать 0, но чтобы не было NaN → возьмём –1.
    DIp_trend[0] = -1
    DIn_trend[0] = -1

    df["DIp_trend"] = DIp_trend
    df["DIn_trend"] = DIn_trend

    # ——— 4. Столбец отношения +DI / -DI ——————————————————————————————————————————————————————
    # DIp_over_DIn = +1, если di_plus > di_minus, иначе –1. Если NaN – ставим –1.
    DIp_over_DIn = np.where(
        (~np.isnan(df["di_plus"].values)) & (~np.isnan(df["di_minus"].values)) &
        (df["di_plus"].values > df["di_minus"].values),
        1, -1
    )
    df["DIp_over_DIn"] = DIp_over_DIn

    # ——— 5. Столбцы «первая смена направления» для +DI и –DI ——————————————————————————————
    # Правила: 
    #  - Если DIp_trend[i] != DIp_trend[i-1], то DIp_turn[i] = 1, иначе 0, но только если
    #    до этого ещё не встречалось такое же изменение в рамках одного «тренда»?
    #  - Точнее: DIp_turn[i] = 1 в тот момент, когда тренд +DI впервые изменился на противоположный,
    #    затем до следующей смены тренда – ставим 0, и при следующей смене снова 1.
    #
    # Аналогично для DIn_turn.
    DIp_turn = np.zeros(n, dtype=int)
    DIn_turn = np.zeros(n, dtype=int)

    # Идем по бару, фиксируем момент, когда DIp_trend меняется
    for i in range(1, n):
        if DIp_trend[i] != DIp_trend[i-1]:
            DIp_turn[i] = 1
        else:
            DIp_turn[i] = 0

        if DIn_trend[i] != DIn_trend[i-1]:
            DIn_turn[i] = 1
        else:
            DIn_turn[i] = 0

    # Первый бар – 0
    DIp_turn[0] = 0
    DIn_turn[0] = 0

    df["DIp_turn"] = DIp_turn
    df["DIn_turn"] = DIn_turn

    # ——— 6. Формируем итоговый вывод ——————————————————————————————————————————————————————
    output = df[[
        "OpenTime",
        "DIp_trend",
        "DIn_trend",
        "DIp_over_DIn",
        "DIp_turn",
        "DIn_turn"
    ]].copy()

    return output
