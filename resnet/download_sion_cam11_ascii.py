"""ASCII-safe wrapper around ``sion_dataset_download_cam11.py``.

The original downloader prints a folder emoji, which can fail when the Windows
Python executable is launched from a non-UTF-8 console. This wrapper reuses the
same downloader functions but keeps console output plain ASCII.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

import sion_dataset_download_cam11 as sion  # noqa: E402


def iter_month_file_paths_ascii(session, month: str):
    month_path = f"/June-Aug/{month}"
    print(f"Listando carpeta remota: {month_path}")
    days = sion.list_folder(session, month_path)

    for day in days:
        if not day.get("isdir"):
            yield f"{month_path}/{day['name']}"
            continue

        day_name = day["name"]
        day_path = f"{month_path}/{day_name}"
        print(f"\nDIR {day_path}")

        files = sion.list_folder(session, day_path)
        print(f"  archivos listados: {len(files)}")
        for file_info in files:
            remote_file_path = f"{day_path}/{file_info['name']}"
            yield remote_file_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASCII-safe SION Cam11 downloader.")
    parser.add_argument("--month", required=True, help="Month under /June-Aug, e.g. 06.")
    parser.add_argument("--out", default="data/sion")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--no-list", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    month = str(args.month).zfill(2)
    out_dir = Path(args.out)
    listing_file = out_dir / f"files_month_{month}.txt"

    session = sion.login()

    if args.no_list:
        if not listing_file.exists():
            raise FileNotFoundError(f"No existe el listado previo: {listing_file}")
        files = [line.strip() for line in listing_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"Listado cargado desde: {listing_file} ({len(files)} archivos)")
    else:
        files = list(iter_month_file_paths_ascii(session, month))
        sion.save_listing(files, month, out_dir)

    before = len(files)
    files = [file for file in files if sion.is_every_10_minutes_cam0(file)]
    print("\nFiltro aplicado: *_11_Cam0.jpg cada 5 min")
    print(f"Archivos seleccionados: {len(files)} de {before}")

    filtered_listing_file = out_dir / f"files_month_{month}_filtered_11_Cam0_jpg.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_listing_file.write_text("\n".join(files), encoding="utf-8")
    print(f"Listado filtrado guardado en: {filtered_listing_file}")

    if args.list_only:
        return 0

    if args.download:
        sion.download_files(session, files, out_dir, overwrite=args.overwrite, retries=args.retries)
    else:
        print("No se ha pedido descarga. Usa --download para bajar imagenes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
