# app.py

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# Импортируем все ваши плагины, которые были до этого
from plugins.indicators.ParabolicSAR.plugin import compute_ParabolicSAR
from plugins.indicators.ADX_DI.plugin import compute_ADX_DI
from plugins.indicators.MovingAverages.plugin import compute_MovingAverages
from plugins.indicators.RSI_Custom.plugin import compute_RSI_Custom
from plugins.indicators.CandlestickPatterns.plugin import compute_CandlestickPatterns
from plugins.indicators.ZigZag.plugin import compute_ZigZag

# Для списка свечных паттернов
import talib

st.set_page_config(
    page_title="Графический интерфейс для расчёта индикаторов",
    layout="wide"
)

st.title("Графический интерфейс для расчёта индикаторов и подготовки данных")

st.markdown("""
**Описание:**
1. Загрузите файл со свечами (младший таймфрейм) в формате CSV или JSON.  
2. Данные старшего таймфрейма **не требуются** — ParabolicSAR сам агрегирует из младших свечей.  
3. При необходимости загрузите файл со стаканом цен (глубина рынка) в формате CSV или JSON (опционально).  
4. Выберите индикаторы и укажите параметры.  
5. Нажмите кнопку **«Вычислить»** — готовый датафрейм можно скачать в формате CSV или JSON для обучения нейросети.
""")

# ========== 1. Загрузка файлов ========== #
st.sidebar.header("1. Загрузка данных")

ohlcv_current_file = st.sidebar.file_uploader(
    "Файл со свечами (младший таймфрейм, CSV или JSON)", type=['csv', 'json']
)

depth_file = st.sidebar.file_uploader(
    "Файл со стаканом цен (глубина рынка), CSV или JSON (опционально)", type=['csv', 'json']
)

# ========== 2. Выбор индикаторов и их параметров ========== #
st.sidebar.header("2. Выбор и настройка индикаторов")

available_plugins = [
    "ParabolicSAR",
    "ADX_DI",
    "MovingAverages",
    "RSI_Custom",
    "CandlestickPatterns",
    "ZigZag"
]

selected_plugins = st.sidebar.multiselect(
    "Выберите плагины (индикаторы) для вычисления:", options=available_plugins, default=available_plugins
)

# ----- Параметры ParabolicSAR -----
with st.sidebar.expander("Параметры ParabolicSAR"):
    af_step = st.number_input(
        "Шаг ускорения (af_step)", min_value=0.001, max_value=1.0, value=0.02, step=0.001
    )
    af_max = st.number_input(
        "Максимальный AF (af_max)", min_value=0.01, max_value=2.0, value=0.2, step=0.01
    )
    higher_tf = st.selectbox(
        "Старший таймфрейм (ресемплинг младших свечей):",
        options=["5T", "15T", "1H", "4H", "1D"], index=0,
        help="Пример: 5T=5 минут, 15T=15 минут, 1H=1 час, 4H=4 часа, 1D=1 день"
    )

# ----- Параметры ADX_DI -----
with st.sidebar.expander("Параметры ADX_DI"):
    di_period = st.number_input(
        "Период DI (timeperiod)", min_value=1, max_value=100, value=14, step=1
    )

# ----- Параметры MovingAverages -----
with st.sidebar.expander("Параметры MovingAverages"):
    ma_type = st.selectbox("Тип скользящей средней", options=["SMA", "EMA"], index=0)
    ma_periods_input = st.text_input(
        "Периоды (через запятую), например: 14,50,100", value="14,50,100"
    )
    try:
        ma_periods = [int(x.strip()) for x in ma_periods_input.split(",") if x.strip().isdigit()]
        if len(ma_periods) == 0:
            ma_periods = [14, 50, 100]
    except:
        ma_periods = [14, 50, 100]

# ----- Параметры RSI_Custom -----
with st.sidebar.expander("Параметры RSI_Custom"):
    rsi_period = st.number_input("Период RSI (rsi_period)", min_value=1, max_value=100, value=14, step=1)
    diverge_period = st.number_input("Период для дивергенции (diverge_period)", min_value=1, max_value=100, value=20, step=1)
    upper_level = st.number_input("Уровень перекупленности (upper_level)", min_value=1, max_value=100, value=80, step=1)
    lower_level = st.number_input("Уровень перепроданности (lower_level)", min_value=1, max_value=100, value=20, step=1)

# ----- Параметры CandlestickPatterns -----
with st.sidebar.expander("Параметры CandlestickPatterns"):
    st.write("Выберите, какие свечные паттерны считать (TA-Lib):")
    all_functions = talib.get_functions()
    candle_functions = [f for f in all_functions if f.startswith("CDL")]
    default_candles = ["CDLDOJI", "CDLHAMMER", "CDLENGULFING"]
    default_candles = [p for p in default_candles if p in candle_functions]

    selected_patterns = st.multiselect(
        "Список паттернов:", options=candle_functions, default=default_candles
    )

# ----- Параметры ZigZag -----
with st.sidebar.expander("Параметры ZigZag"):
    zz_threshold = st.number_input("Порог ZigZag (threshold, доля)", min_value=0.001, max_value=1.0, value=0.05, step=0.001)

# ========== 3. Кнопка «Вычислить» ========== #
compute_button = st.sidebar.button("Вычислить")

# ========== 4. Вспомогательная функция для загрузки CSV/JSON ========== #
def load_dataframe(uploaded_file):
    """
    Читает CSV или JSON из загруженного файла и возвращает pandas.DataFrame.
    """
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_json(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка при чтении файла {uploaded_file.name}: {e}")
        return None

# ========== 4.1. Кешируем предварительную обработку depth_df ========== #
@st.cache_data(show_spinner=False)
def prepare_depth_dataframe(depth_file):
    """
    Читает depth_file (CSV или JSON), определяет все уровни bid/ask (i = 1..N),
    переводит timestamp → OpenTime (int64, ms), и вычисляет набор фич:
      – sum_bid_top3, sum_ask_top3, bid_ask_diff_top3
      – sum_bid_top5, sum_ask_top5, bid_ask_diff_top5
      – sum_bid_top10, sum_ask_top10, bid_ask_diff_top10
      – bid_price_1, ask_price_1, spread_1
      – imbalance_1 (bid_qty_1 / (ask_qty_1 + ε))
      – bid_vwap_5, ask_vwap_5, vwap_spread_5  ← ЗДЕСЬ исправлено
      – max_bid_top5, max_ask_top5

    Автоматически определяет, сколько уровней реально есть в файле.
    """
    df = load_dataframe(depth_file)
    if df is None:
        return None

    # 1) timestamp → OpenTime
    if "timestamp" not in df.columns:
        st.error("В файле depth отсутствует обязательная колонка 'timestamp'.")
        return None

    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["OpenTime"] = df["timestamp"].astype("int64") // 1_000_000
    elif df["timestamp"].dtype == "object":
        df["OpenTime"] = pd.to_datetime(df["timestamp"]).astype("int64") // 1_000_000
    elif pd.api.types.is_numeric_dtype(df["timestamp"]):
        ts_max = df["timestamp"].max()
        if ts_max < 1e12:
            df["OpenTime"] = df["timestamp"].astype("int64") * 1000
        else:
            df["OpenTime"] = df["timestamp"].astype("int64")
    else:
        df["OpenTime"] = pd.to_datetime(df["timestamp"]).astype("int64") // 1_000_000

    # 2) Собираем все уровни bid_i и ask_i
    bid_qty_cols   = [c for c in df.columns if c.startswith("bid_qty_")]
    ask_qty_cols   = [c for c in df.columns if c.startswith("ask_qty_")]
    bid_price_cols = [c for c in df.columns if c.startswith("bid_price_")]
    ask_price_cols = [c for c in df.columns if c.startswith("ask_price_")]

    if not bid_qty_cols or not ask_qty_cols:
        st.error("В файле depth не найдены колонки 'bid_qty_i' или 'ask_qty_i'.")
        return None

    def sort_by_index(col_list, prefix):
        return sorted(col_list, key=lambda name: int(name.replace(prefix + "_", "")))

    bid_qty_cols   = sort_by_index(bid_qty_cols,   "bid_qty")
    ask_qty_cols   = sort_by_index(ask_qty_cols,   "ask_qty")
    bid_price_cols = sort_by_index(bid_price_cols, "bid_price")
    ask_price_cols = sort_by_index(ask_price_cols, "ask_price")

    max_bid_levels = max(int(c.split("_")[2]) for c in bid_qty_cols)
    max_ask_levels = max(int(c.split("_")[2]) for c in ask_qty_cols)

    top3_n  = min(3,  max_bid_levels, max_ask_levels)
    top5_n  = min(5,  max_bid_levels, max_ask_levels)
    top10_n = min(10, max_bid_levels, max_ask_levels)

    bid_top3  = bid_qty_cols[:top3_n]
    ask_top3  = ask_qty_cols[:top3_n]
    bid_top5  = bid_qty_cols[:top5_n]
    ask_top5  = ask_qty_cols[:top5_n]
    bid_top10 = bid_qty_cols[:top10_n]
    ask_top10 = ask_qty_cols[:top10_n]

    # 3) Считаем суммы по первым 3 / 5 / 10 уровням
    df["sum_bid_top3"]      = df[bid_top3].sum(axis=1)
    df["sum_ask_top3"]      = df[ask_top3].sum(axis=1)
    df["bid_ask_diff_top3"] = df["sum_bid_top3"] - df["sum_ask_top3"]

    df["sum_bid_top5"]      = df[bid_top5].sum(axis=1)
    df["sum_ask_top5"]      = df[ask_top5].sum(axis=1)
    df["bid_ask_diff_top5"] = df["sum_bid_top5"] - df["sum_ask_top5"]

    df["sum_bid_top10"]      = df[bid_top10].sum(axis=1)
    df["sum_ask_top10"]      = df[ask_top10].sum(axis=1)
    df["bid_ask_diff_top10"] = df["sum_bid_top10"] - df["sum_ask_top10"]

    # 4) bid_price_1, ask_price_1, spread_1
    if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
        df["bid_price_1"] = df["bid_price_1"].astype(float)
        df["ask_price_1"] = df["ask_price_1"].astype(float)
        df["spread_1"]    = df["ask_price_1"] - df["bid_price_1"]
    else:
        df["bid_price_1"] = np.nan
        df["ask_price_1"] = np.nan
        df["spread_1"]    = np.nan

    # 5) Имбаланс 1 уровня
    if "bid_qty_1" in df.columns and "ask_qty_1" in df.columns:
        df["imbalance_1"] = df["bid_qty_1"] / (df["ask_qty_1"] + 1e-9)
    else:
        df["imbalance_1"] = np.nan

    # 6) ====== Правильный расчёт VWAP top-5 (bid_vwap_5, ask_vwap_5) ======
    if len(bid_price_cols) >= 5 and len(bid_top5) >= 5 and len(ask_price_cols) >= 5 and len(ask_top5) >= 5:
        bid_prices_5 = [f"bid_price_{i}" for i in range(1, 6)]
        ask_prices_5 = [f"ask_price_{i}" for i in range(1, 6)]

        bid_prices_arr = df[bid_prices_5].to_numpy(dtype=float)       # (n_rows, 5)
        bid_qtys_arr   = df[bid_top5].to_numpy(dtype=float)           # (n_rows, 5)
        ask_prices_arr = df[ask_prices_5].to_numpy(dtype=float)
        ask_qtys_arr   = df[ask_top5].to_numpy(dtype=float)

        bid_pq    = (bid_prices_arr * bid_qtys_arr).sum(axis=1)   # по каждому ряду
        ask_pq    = (ask_prices_arr * ask_qtys_arr).sum(axis=1)

        bid_qsum  = bid_qtys_arr.sum(axis=1)
        ask_qsum  = ask_qtys_arr.sum(axis=1)

        df["bid_vwap_5"]    = bid_pq / (bid_qsum + 1e-9)
        df["ask_vwap_5"]    = ask_pq / (ask_qsum + 1e-9)
        df["vwap_spread_5"] = df["ask_vwap_5"] - df["bid_vwap_5"]
    else:
        df["bid_vwap_5"]    = np.nan
        df["ask_vwap_5"]    = np.nan
        df["vwap_spread_5"] = np.nan

    # 7) Макс объём по топ-5
    df["max_bid_top5"] = df[bid_top5].max(axis=1) if bid_top5 else np.nan
    df["max_ask_top5"] = df[ask_top5].max(axis=1) if ask_top5 else np.nan

    # 8) Формируем список колонок для merge’а
    result_cols = [
        "OpenTime",
        "sum_bid_top3", "sum_ask_top3", "bid_ask_diff_top3",
        "sum_bid_top5", "sum_ask_top5", "bid_ask_diff_top5",
        "sum_bid_top10", "sum_ask_top10", "bid_ask_diff_top10",
        "bid_price_1", "ask_price_1", "spread_1", "imbalance_1",
        "bid_vwap_5", "ask_vwap_5", "vwap_spread_5",
        "max_bid_top5", "max_ask_top5"
    ]
    return df[result_cols]

# ===== 5. Основной блок вычислений =====
if compute_button:
    # 5.1 Проверяем свечи
    if ohlcv_current_file is None:
        st.error("Пожалуйста, загрузите файл со свечами (младший таймфрейм).")
        st.stop()

    ohlcv_current = load_dataframe(ohlcv_current_file)
    if ohlcv_current is None:
        st.stop()

    needed_cols = {"OpenTime", "Open", "High", "Low", "Close", "Volume"}
    if not needed_cols.issubset(ohlcv_current.columns):
        st.error(f"В файле {ohlcv_current_file.name} не хватает колонок {needed_cols}.")
        st.stop()

    # 5.2 Подготавливливаем df с фичами depth (если есть)
    if depth_file is not None:
        depth_subset = prepare_depth_dataframe(depth_file)
        if depth_subset is None:
            st.stop()
    else:
        depth_subset = None

    # 5.3 Стартуем df_result по колонке OpenTime
    df_result = pd.DataFrame({"OpenTime": ohlcv_current["OpenTime"].values.astype("int64")})

    # ----- 5.4 ParabolicSAR -----
    if "ParabolicSAR" in selected_plugins:
        st.info("Вычисляем ParabolicSAR...")
        try:
            psar_df = compute_ParabolicSAR(
                ohlcv_df=ohlcv_current,
                higher_tf=higher_tf,
                af_step=af_step,
                af_max=af_max
            )
            df_result = df_result.merge(psar_df, on="OpenTime", how="left")
        except Exception as e:
            st.error(f"Ошибка в ParabolicSAR: {e}")
            st.stop()

    # ----- 5.5 ADX_DI -----
    if "ADX_DI" in selected_plugins:
        st.info("Вычисляем ADX_DI...")
        try:
            adx_df = compute_ADX_DI(
                ohlcv_df=ohlcv_current,
                timeperiod=di_period
            )
            df_result = df_result.merge(adx_df, on="OpenTime", how="left")
        except Exception as e:
            st.error(f"Ошибка в ADX_DI: {e}")
            st.stop()

    # ----- 5.6 MovingAverages -----
    if "MovingAverages" in selected_plugins:
        st.info("Вычисляем MovingAverages...")
        try:
            ma_df = compute_MovingAverages(
                ohlcv_df=ohlcv_current,
                ma_type=ma_type,
                periods=ma_periods
            )
            df_result = df_result.merge(ma_df, on="OpenTime", how="left")
        except Exception as e:
            st.error(f"Ошибка в MovingAverages: {e}")
            st.stop()

    # ----- 5.7 RSI_Custom -----
    if "RSI_Custom" in selected_plugins:
        st.info("Вычисляем RSI_Custom...")
        try:
            rsi_df = compute_RSI_Custom(
                ohlcv_df=ohlcv_current,
                rsi_period=rsi_period,
                diverge_period=diverge_period,
                upper_level=upper_level,
                lower_level=lower_level
            )
            df_result = df_result.merge(rsi_df, on="OpenTime", how="left")
        except Exception as e:
            st.error(f"Ошибка в RSI_Custom: {e}")
            st.stop()

    # ----- 5.8 CandlestickPatterns -----
    if "CandlestickPatterns" in selected_plugins:
        st.info("Вычисляем CandlestickPatterns...")
        try:
            candles_df = compute_CandlestickPatterns(
                ohlcv_df=ohlcv_current,
                patterns=selected_patterns
            )
            df_result = df_result.merge(candles_df, on="OpenTime", how="left")
        except Exception as e:
            st.error(f"Ошибка в CandlestickPatterns: {e}")
            st.stop()

    # ----- 5.9 ZigZag -----
    if "ZigZag" in selected_plugins:
        st.info("Вычисляем ZigZag...")
        try:
            zz_df = compute_ZigZag(
                ohlcv_df=ohlcv_current,
                threshold=zz_threshold
            )
            df_result = df_result.merge(zz_df, on="OpenTime", how="left")
        except Exception as e:
            st.error(f"Ошибка в ZigZag: {e}")
            st.stop()

    # ----- 5.10 Обрабатываем depth-данные (join по OpenTime) -----
    if depth_subset is not None:
        st.info("Обрабатываем данные стакана...")
        try:
            df_result    = df_result.set_index("OpenTime")
            depth_subset = depth_subset.set_index("OpenTime")
            df_result    = df_result.join(depth_subset, how="left")
            df_result    = df_result.reset_index()
        except Exception as e:
            st.error(f"Ошибка при обработке стакана: {e}")
            st.stop()

    # ----- 5.11 Очищаем пропуски и заполняем нулями -----
    st.info("Очищаем пропуски и заполняем нулями…")
    df_clean = df_result.fillna(0)

    # ----- 5.12 Показываем результат и даём ссылки для скачивания -----
    st.success("Готово! Первые 10 строк итогового набора признаков:")
    st.dataframe(df_clean.head(10))

    # Скачивание CSV
    csv_buffer = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать результат как CSV",
        data=csv_buffer,
        file_name="features_dataset.csv",
        mime="text/csv"
    )

    # Скачивание JSON
    json_str = df_clean.to_json(orient="records", force_ascii=False)
    st.download_button(
        label="Скачать результат как JSON",
        data=json_str,
        file_name="features_dataset.json",
        mime="application/json"
    )
