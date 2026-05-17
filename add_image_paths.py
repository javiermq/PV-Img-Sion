from pathlib import Path
import pandas as pd
import re

WEATHER_TSV = Path("data/weather.tsv")
IMAGES_DIR = Path("data/procesadas")
OUTPUT_TSV = Path("data/weather_with_images.tsv")

# Formato esperado:
# 20250701_103500_Pi03_capture_47153_11_Cam0_crop
# También acepta extensión: .jpg, .png, etc.
pattern = re.compile(
    r"(?P<date>\d{8})_(?P<time>\d{6}).*11_Cam0_crop(?:\.\w+)?$"
)

# 1. Leer weather.tsv
df = pd.read_csv(WEATHER_TSV, sep="\t")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Creamos clave comparable:
# 2025-07-01 10:35:00+00:00 -> 20250701_103500
df["image_key"] = df["timestamp"].dt.strftime("%Y%m%d_%H%M%S")

# 2. Buscar imágenes
image_map = {}

for path in IMAGES_DIR.rglob("*"):
    if not path.is_file():
        continue

    match = pattern.search(path.name)
    if not match:
        continue

    key = f"{match.group('date')}_{match.group('time')}"

    # Si hay duplicados para el mismo timestamp, dejamos el primero
    if key not in image_map:
        image_map[key] = str(path)

# 3. Añadir columna image_path
df["image_path"] = df["image_key"].map(image_map)

# Opcional: dejar vacío en vez de NaN
df["image_path"] = df["image_path"].fillna("")

# 4. Quitar columna auxiliar
df = df.drop(columns=["image_key"])

# 5. Guardar
df.to_csv(OUTPUT_TSV, sep="\t", index=False)

# 6. Resumen
total = len(df)
with_image = (df["image_path"] != "").sum()
without_image = total - with_image

print(f"Filas totales: {total}")
print(f"Filas con imagen: {with_image}")
print(f"Filas sin imagen: {without_image}")
print(f"% con imagen: {with_image / total * 100:.2f}%")
print(f"% sin imagen: {without_image / total * 100:.2f}%")
print(f"Guardado en: {OUTPUT_TSV}")