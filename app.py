import streamlit as st
import pandas as pd
import plotly.express as px
import re

# -----------------------
# Настройки и словари
# -----------------------

# Расшифровка поллютантов
POLLUTANT_NAMES = {
    "6001": "PM2.5 (суточная концентрация)"
}

# Расшифровка станций Берлина
STATION_NAMES = {
    "DEBE010": "Amrumer Straße (Wedding, городской фон)",
    "DEBE034": "Nansenstraße (Neukölln, городской фон)",
    "DEBE051": "Buch / Hobrechtsfelder Chaussee (Pankow, пригородный фон)",
    "DEBE065": "Frankfurter Allee (Friedrichshain, транспортная станция)",
    "DEBE068": "Brückenstraße (Mitte, городской фон)",
}


def extract_station_id(s: str) -> str | None:
    """
    Из строки вида 'DE/SPO.DE_DEBE010_PM2_dataGroup2'
    достаём 'DEBE010'
    """
    m = re.search(r"_(DE[A-Z]{2}\d{3})_", str(s))
    return m.group(1) if m else None


# -----------------------
# Загрузка и подготовка данных
# -----------------------

@st.cache_data
def load_data():
    df = pd.read_parquet("data_combined.parquet")

    # Базовые проверки
    base_cols = ["station", "pollutant", "value", "date"]
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        st.error(f"В данных не хватает колонок: {missing}")
        st.stop()

    # Типы
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["pollutant"] = df["pollutant"].astype(str)

    # Убираем мусор: отрицательные и слишком большие значения
    df = df[(df["value"] >= 0) & (df["value"] <= 500)]

    # Короткий ID станции
    df["station_id"] = df["station"].apply(extract_station_id)

    # Человекочитаемое название станции
    df["station_name"] = df["station_id"].map(STATION_NAMES)

    # Если вдруг остались строки без названия — подставим station_id
    df["station_name"] = df["station_name"].fillna(df["station_id"])

    return df


df = load_data()

# -----------------------
# Настройки страницы
# -----------------------

st.set_page_config(page_title="Качество воздуха в Берлине", layout="wide")

st.title("Анализ качества воздуха в Берлине (EEA, PM2.5)")
st.markdown("""
Данные: суточные концентрации **PM2.5** за 2024 год по пяти официальным станциям мониторинга в Берлине.

Станции:

- **Amrumer Straße (Wedding, городской фон)** — код DEBE010. Жилой район на севере города, типичный городской фон.
- **Nansenstraße (Neukölln, городской фон)** — код DEBE034. Плотная жилая застройка в районе Neukölln.
- **Buch / Hobrechtsfelder Chaussee (Pankow, пригородный фон)** — код DEBE051. Пригородный район с более чистым воздухом.
- **Frankfurter Allee (Friedrichshain, транспортная станция)** — код DEBE065. Одна из загруженных магистралей, сильное влияние дорожного трафика.
- **Brückenstraße (Mitte, городской фон)** — код DEBE068. Центральная часть города, городской фон в районе Mitte.

Это позволяет сравнить: **пригородный фон**, **городской фон** и **транспортную магистраль**.
""")

# -----------------------
# Боковая панель
# -----------------------

with st.sidebar:
    st.header("Фильтры")

    # Поллютанты (у нас по факту один, но делаем нормально)
    pollutant_codes = sorted(df["pollutant"].unique())
    pollutant_labels = [
        f"{code} – {POLLUTANT_NAMES.get(code, 'неизвестный поллютант')}"
        for code in pollutant_codes
    ]
    code_by_label = dict(zip(pollutant_labels, pollutant_codes))

    pollutant_label_sel = st.selectbox("Поллютант", pollutant_labels)
    pollutant_sel = code_by_label[pollutant_label_sel]

    # Станции по ЧЕЛОВЕЧЕСКИМ названиям
    stations = sorted(df["station_name"].dropna().unique())
    if not stations:
        st.error("В данных не найдены станции (station_name). Проверь подготовку данных.")
        st.stop()

    stations_sel = st.multiselect(
        "Станции мониторинга",
        stations,
        default=stations,
    )

    # Диапазон дат
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_from, date_to = st.slider(
        "Период наблюдений",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )

# -----------------------
# Фильтрация данных
# -----------------------

mask = (
    (df["pollutant"] == pollutant_sel) &
    (df["station_name"].isin(stations_sel)) &
    (df["date"].dt.date.between(date_from, date_to))
)

df_filt = df[mask].copy()

if df_filt.empty:
    st.warning("По выбранным фильтрам данных нет. Попробуй выбрать больше станций или расширить период.")
    st.stop()

poll_name = POLLUTANT_NAMES.get(pollutant_sel, pollutant_sel)

# -----------------------
# Описательная статистика
# -----------------------

st.subheader(f"Описательная статистика ({poll_name})")

stats = df_filt["value"].describe().rename(
    {
        "count": "Количество измерений",
        "mean": "Среднее",
        "std": "Ст. отклонение",
        "min": "Минимум",
        "25%": "25-й перцентиль",
        "50%": "Медиана",
        "75%": "75-й перцентиль",
        "max": "Максимум",
    }
)
st.table(stats.to_frame(name="Концентрация, µg/m³"))

# -----------------------
# Временной ряд
# -----------------------

st.subheader(f"Средняя концентрация во времени ({poll_name})")

ts = (
    df_filt.groupby("date")["value"]
    .mean()
    .reset_index()
    .sort_values("date")
)

fig_ts = px.line(
    ts,
    x="date",
    y="value",
    title=f"Временной ряд средних концентраций {poll_name} по выбранным станциям",
    labels={"date": "Дата", "value": "Концентрация, µg/m³"},
)
st.plotly_chart(fig_ts, use_container_width=True)

# -----------------------
# Сравнение станций (бар-чарт)
# -----------------------

st.subheader(f"Сравнение станций по среднему уровню {poll_name}")

by_station = (
    df_filt.groupby("station_name")["value"]
    .mean()
    .reset_index()
    .sort_values("value", ascending=False)
)

fig_bar = px.bar(
    by_station,
    x="station_name",
    y="value",
    title=f"Средняя концентрация {poll_name} по станциям Берлина",
    labels={"station_name": "Станция", "value": "Концентрация, µg/m³"},
)
st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------
# Тепловая карта (станция × дата)
# -----------------------

st.subheader(f"Тепловая карта концентраций {poll_name}")

# 1) агрегируем по дню и станции
heat_df = (
    df_filt.copy()
    .groupby(["station_name", "date"])["value"]
    .mean()
    .reset_index()
)

if heat_df.empty:
    st.info("Недостаточно данных для построения тепловой карты.")
else:
    # 2) делаем матрицу: строки = станции, колонки = даты
    pivot = heat_df.pivot(index="station_name", columns="date", values="value")

    # подписи осей
    y_labels = list(pivot.index)
    x_labels = list(pivot.columns)

    # 3) реальные численные значения
    z_values = pivot.values

    vmin = float(pd.DataFrame(z_values).min().min())
    vmax = float(pd.DataFrame(z_values).max().max())

    import plotly.express as px

    fig_hm = px.imshow(
        z_values,
        x=x_labels,
        y=y_labels,
        color_continuous_scale="Turbo",
        aspect="auto",
        zmin=vmin,
        zmax=vmax,
        labels={"x": "Дата", "y": "Станция", "color": "Концентрация, µg/m³"},
        title=f"Тепловая карта концентраций {poll_name}",
    )

    # красивый тултип с реальными числами
    fig_hm.update_traces(
        hovertemplate=(
            "Дата: %{x}<br>"
            "Станция: %{y}<br>"
            "Концентрация, µg/m³: %{z:.2f}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(fig_hm, use_container_width=True)


# -----------------------
# Таблица с данными
# -----------------------

with st.expander("Показать исходные строки данных"):
    st.dataframe(
        df_filt.sort_values(["date", "station_name"]).reset_index(drop=True)
    )
