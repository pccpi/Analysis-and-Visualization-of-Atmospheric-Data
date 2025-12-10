# data_prepare.py
# Подготовка данных EEA:
# 1) распаковываем ParquetFiles.zip
# 2) объединяем все parquet-файлы
# 3) приводим имена колонок к нормальному виду
# 4) чистим аномальные значения (отрицательные и > 500)
# 5) добавляем короткий ID станции и человекочитаемое название
# 6) сохраняем итоговый датасет в parquet и csv

import zipfile
import pathlib
import pandas as pd
import re

# ---- настройки путей ----
ZIP_PATH = pathlib.Path("ParquetFiles.zip")   # имя архива в корне репозитория
RAW_DIR = pathlib.Path("data_raw")           # куда распакуем файлы
RAW_DIR.mkdir(exist_ok=True)

print("➡ Распаковываем архив...")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(RAW_DIR)

# ---- читаем все parquet-файлы ----
files = list(RAW_DIR.rglob("*.parquet"))
print(f"Найдено parquet-файлов: {len(files)}")

dfs = []
for path in files:
    try:
        df = pd.read_parquet(path)
        df["source_file"] = path.name
        dfs.append(df)
    except Exception as e:
        print("⚠ Ошибка при чтении", path, ":", e)

if not dfs:
    raise SystemExit("Не удалось прочитать ни одного parquet-файла")

data = pd.concat(dfs, ignore_index=True)

print("Колонки в исходных данных:")
print(list(data.columns))

# ---- переименуем колонки в более удобный вид ----
rename_map = {}
for col in data.columns:
    lower = col.lower()

    if lower.startswith("samplingpoint"):
        rename_map[col] = "station"          # код станции (длинный)
    elif "pollutant" in lower:
        rename_map[col] = "pollutant"        # код поллютанта (6001 = PM2.5)
    elif lower == "value":
        rename_map[col] = "value"            # значение концентрации
    elif lower == "start":
        rename_map[col] = "datetime_start"   # начало интервала
    elif lower == "end":
        rename_map[col] = "datetime_end"     # конец интервала

data = data.rename(columns=rename_map)

# ---- обработка дат ----
if "datetime_start" in data.columns:
    data["datetime_start"] = pd.to_datetime(data["datetime_start"])
    data["date"] = data["datetime_start"].dt.date
    data["year"] = data["datetime_start"].dt.year
    data["month"] = data["datetime_start"].dt.month

print("\nКолонки после переименования:")
print(list(data.columns))

# ---- приведение value к числу и очистка аномалий ----
# отрицательные и слишком большие значения для PM2.5 физически невозможны
data["value"] = pd.to_numeric(data["value"], errors="coerce")

before_rows = len(data)
data = data[(data["value"] >= 0) & (data["value"] <= 500)]
after_rows = len(data)

print(f"\nСтрок до очистки: {before_rows}")
print(f"Строк после очистки (0 <= value <= 500): {after_rows}")

# ---- короткий ID станции (DEBE010 и т.п.) ----
def extract_station_id(s: str) -> str | None:
    """
    Из строки вида 'DE/SPO.DE_DEBE010_PM2_dataGroup2'
    достаём 'DEBE010'
    """
    m = re.search(r"\.(DE[A-Z]{2}\d{3})", str(s))
    return m.group(1) if m else None

data["station_id"] = data["station"].apply(extract_station_id)

# ---- человекочитаемые названия станций Берлина ----
station_names = {
    "DEBE010": "Amrumer Straße (Wedding, городской фон)",
    "DEBE034": "Nansenstraße (Neukölln, городской фон)",
    "DEBE051": "Buch / Hobrechtsfelder Chaussee (Pankow, пригородный фон)",
    "DEBE065": "Frankfurter Allee (Friedrichshain, транспортная станция)",
    "DEBE068": "Brückenstraße (Mitte, городской фон)",
}

data["station_name"] = data["station_id"].map(station_names)

print("\nПримеры строк после всех преобразований:")
print(data.head())

# ---- сохраняем итоговый датасет ----
OUT_PARQUET = pathlib.Path("data_combined.parquet")
OUT_CSV = pathlib.Path("data_combined.csv")

data.to_parquet(OUT_PARQUET, index=False)
data.to_csv(OUT_CSV, index=False)

print("\n✅ Готово!")
print("Сохранено в:", OUT_PARQUET, "и", OUT_CSV)
print("Итоговое количество строк:", len(data))
