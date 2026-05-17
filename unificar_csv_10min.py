from pathlib import Path
import pandas as pd


INPUT_DIR = Path("PV+dataset")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "weather.tsv"

FILES = {
    "humidity": "humidity.csv",
    "irradiation": "irradiation.csv",
    "precipitation": "precipitation.csv",
    "production": "production.csv",
    "temperature": "temperature.csv",
    "winddirection": "winddirection.csv",
    "windspeed": "windspeed.csv",
}

FREQ = "5min"


def read_sensor_csv(path: Path, column_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "time" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{path} debe tener columnas 'time' y 'value'")

    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["time", "value"])
    df = df.sort_values("time")

    df = df.groupby("time", as_index=True)["value"].median().to_frame()
    df.columns = [column_name]

    return df


def main():
    dataframes = {}

    print("Leyendo ficheros...")

    for column_name, filename in FILES.items():
        path = INPUT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"No existe el fichero: {path}")

        df = read_sensor_csv(path, column_name)
        dataframes[column_name] = df

        print(
            f"{path}: "
            f"{len(df):,} filas | "
            f"inicio={df.index.min()} | "
            f"fin={df.index.max()}"
        )

    common_start = max(df.index.min() for df in dataframes.values())
    common_end = min(df.index.max() for df in dataframes.values())

    print()
    print("Rango común:")
    print(f"  inicio = {common_start}")
    print(f"  fin    = {common_end}")

    if common_start >= common_end:
        raise ValueError(
            "No hay solape temporal entre todos los ficheros. "
            "El inicio común es posterior o igual al final común."
        )

    unified = []
    fill_report = {}

    for column_name, df in dataframes.items():
        raw_10min = df[column_name].resample(FREQ).median()
        series_10min = raw_10min.ffill()

        series_10min = series_10min.loc[
            (series_10min.index >= common_start.floor(FREQ))
            & (series_10min.index <= common_end.floor(FREQ))
        ]

        series_10min = series_10min.loc[series_10min.index >= common_start]

        raw_common = raw_10min.reindex(series_10min.index)

        filled_count = raw_common.isna().sum()
        total_count = len(raw_common)
        fill_pct = 100 * filled_count / total_count if total_count else 0

        fill_report[column_name] = {
            "total_intervalos": total_count,
            "rellenados": int(filled_count),
            "porcentaje_rellenado": fill_pct,
        }

        unified.append(series_10min.rename(column_name))

    matrix = pd.concat(unified, axis=1)

    # Por si queda algún NaN inicial en alguna serie
    matrix = matrix.ffill().bfill()

    matrix = matrix.reset_index()
    matrix = matrix.rename(columns={"time": "timestamp", "index": "timestamp"})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matrix.to_csv(OUTPUT_FILE, sep="\t", index=False, float_format="%.2f")

    print()
    print("Resumen de rellenado por último valor anterior:")
    for column_name, report in fill_report.items():
        print(
            f"  {column_name}: "
            f"{report['rellenados']:,}/{report['total_intervalos']:,} "
            f"intervalos rellenados "
            f"({report['porcentaje_rellenado']:.2f}%)"
        )

    print()
    print(f"Fichero generado: {OUTPUT_FILE}")
    print(f"Shape final: {matrix.shape}")
    print()
    print(matrix.head())


if __name__ == "__main__":
    main()