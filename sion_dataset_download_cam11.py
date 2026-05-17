
"""
Descarga/listado del dataset SION desde Synology Sharing, sin Jupyter.

Versión robusta:
  - crea carpetas locales automáticamente
  - lista archivos del mes y guarda un TXT
  - descarga archivo a archivo preservando la estructura de carpetas
  - opcionalmente intenta descargar el mes como ZIP con --zip

Uso típico:
    python sion_dataset_download_cam11.py --month 07 --download

Por defecto descarga solo imágenes cuyo nombre termina en 11_Cam0.jpg.

Salida por defecto:
    data/sion/files_month_07.txt
    data/sion/June-Aug/07/12/archivo.jpg
    data/sion/June-Aug/07/13/archivo.jpg
    ...

Variables de entorno opcionales:
    SION_SHARING_ID
    SION_PASSWORD
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import requests

import re
from pathlib import Path

def is_every_10_minutes_cam0(path: str) -> bool:
    name = Path(path).name

    # Ejemplo:
    # 20250701_061000_Pi03_capture_46888_11_Cam0.jpg
    match = re.search(r"^\d{8}_(\d{2})(\d{2})(\d{2})_", name)
    if not match:
        return False

    hour, minute, second = match.groups()

    return (
        second == "00"
        and minute in {"00", "05", "10", "15","20", "25", "30", "35","40", "45","50", "55"}
        and name.endswith("_11_Cam0.jpg")
    )
    

BASE = "https://dsm.n32x.hevs.ch"
SHARING_ID = os.getenv("SION_SHARING_ID", "qkgeLrdRT")
PASSWORD = os.getenv("SION_PASSWORD", "Sion2026")


def quoted(value: str) -> str:
    """Synology espera algunos valores como strings entrecomillados."""
    return f'"{value}"'


def login() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            )
        }
    )

    auth_body = {
        "api": "SYNO.Core.Sharing.Login",
        "method": "login",
        "version": "1",
        "sharing_id": quoted(SHARING_ID),
        "password": quoted(PASSWORD),
    }

    response = session.post(f"{BASE}/sharing/webapi/auth.cgi", data=auth_body, timeout=60)
    response.raise_for_status()

    data = response.json()
    if not data.get("success"):
        raise RuntimeError(f"Login failed: {data}")

    sharing_sid = data["data"]["sharing_sid"]
    session.cookies.set("sharing_sid", sharing_sid, domain="dsm.n32x.hevs.ch", path="/")
    return session


def list_folder(session: requests.Session, remote_path: str, limit: int = 1000) -> list[dict[str, Any]]:
    all_files: list[dict[str, Any]] = []
    offset = 0

    while True:
        body = {
            "api": "SYNO.FolderSharing.List",
            "method": "list",
            "version": "2",
            "folder_path": quoted(remote_path),
            "_sharing_id": quoted(SHARING_ID),
            "offset": str(offset),
            "limit": str(limit),
            "sort_by": quoted("name"),
            "sort_direction": quoted("asc"),
        }

        response = session.post(
            f"{BASE}/sharing/webapi/entry.cgi",
            data=body,
            headers={"Origin": BASE},
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        if not data.get("success"):
            raise RuntimeError(f"List failed for {remote_path}: {data}")

        files = data["data"].get("files", [])
        all_files.extend(files)

        if len(files) < limit:
            break

        offset += limit

    return all_files


def iter_month_file_paths(session: requests.Session, month: str) -> Iterable[str]:
    """Devuelve rutas remotas tipo /June-Aug/07/12/file.jpg."""
    month_path = f"/June-Aug/{month}"
    print(f"Listando carpeta remota: {month_path}")

    days = list_folder(session, month_path)

    for day in days:
        if not day.get("isdir"):
            # Por si hubiese archivos directamente dentro del mes.
            yield f"{month_path}/{day['name']}"
            continue

        day_name = day["name"]
        day_path = f"{month_path}/{day_name}"
        print(f"\n📁 {day_path}")

        files = list_folder(session, day_path)
        for file_info in files:
            remote_file_path = f"{day_path}/{file_info['name']}"
            size = file_info.get("size", "")
            filetype = file_info.get("type", "")
            print(f"  {remote_file_path} | size={size} | type={filetype}")
            yield remote_file_path


def save_listing(files: list[str], month: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"files_month_{month}.txt"
    out_file.write_text("\n".join(files), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Total archivos en mes {month}: {len(files)}")
    print(f"Listado guardado en: {out_file}")
    return out_file


def local_path_for_remote(remote_path: str, out_dir: Path) -> Path:
    """Convierte /June-Aug/07/12/a.jpg en data/sion/June-Aug/07/12/a.jpg."""
    clean = remote_path.lstrip("/")
    return out_dir / clean


def check_download_response(response: requests.Response, remote_path: str) -> None:
    """Detecta errores silenciosos: HTML/JSON en lugar de imagen/ZIP."""
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" in content_type or "text/html" in content_type:
        preview = response.content[:500].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"La descarga de {remote_path} no devolvió un fichero binario. "
            f"Content-Type={content_type}. Respuesta inicial: {preview}"
        )


def download_one_file(
    session: requests.Session,
    remote_path: str,
    out_dir: Path,
    overwrite: bool = False,
    retries: int = 3,
) -> Path:
    """Descarga un archivo individual preservando carpetas."""
    local_file = local_path_for_remote(remote_path, out_dir)
    local_file.parent.mkdir(parents=True, exist_ok=True)

    if local_file.exists() and local_file.stat().st_size > 0 and not overwrite:
        print(f"SKIP existe: {local_file}")
        return local_file

    filename = Path(remote_path).name
    url = f"{BASE}/fsdownload/webapi/file_download.cgi/{filename}"
    body = {
        "api": "SYNO.FolderSharing.Download",
        "method": "download",
        "version": "2",
        "mode": quoted("download"),
        "stdhtml": "false",
        "dlname": quoted(filename),
        "path": f'["{remote_path}"]',
        "_sharing_id": quoted(SHARING_ID),
        "codepage": quoted("spn"),
    }

    last_error: Exception | None = None
    tmp_file = local_file.with_suffix(local_file.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            with session.post(
                url,
                data=body,
                headers={"Origin": BASE},
                stream=True,
                timeout=(30, 180),
            ) as response:
                response.raise_for_status()

                # Leemos un primer bloque para poder detectar HTML/JSON sin guardar basura.
                iterator = response.iter_content(chunk_size=1024 * 1024)
                first_chunk = next(iterator, b"")
                if not first_chunk:
                    raise RuntimeError(f"Respuesta vacía descargando {remote_path}")

                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type or "text/html" in content_type:
                    preview = first_chunk[:500].decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"La descarga de {remote_path} no devolvió un fichero binario. "
                        f"Content-Type={content_type}. Respuesta inicial: {preview}"
                    )

                with tmp_file.open("wb") as file:
                    file.write(first_chunk)
                    for chunk in iterator:
                        if chunk:
                            file.write(chunk)

            tmp_file.replace(local_file)
            print(f"OK: {remote_path} -> {local_file}")
            return local_file

        except Exception as exc:  # reintento controlado
            last_error = exc
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
            print(f"WARN intento {attempt}/{retries} fallido: {remote_path} ({exc})")
            time.sleep(min(2 * attempt, 10))

    raise RuntimeError(f"No se pudo descargar {remote_path}: {last_error}")


def download_files(
    session: requests.Session,
    files: list[str],
    out_dir: Path,
    overwrite: bool = False,
    retries: int = 3,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(files)
    ok = 0
    failed: list[str] = []

    print(f"\nDescargando {total} archivos uno a uno en: {out_dir}")

    for index, remote_path in enumerate(files, start=1):
        print(f"\n[{index}/{total}] {remote_path}")
        try:
            download_one_file(
                session=session,
                remote_path=remote_path,
                out_dir=out_dir,
                overwrite=overwrite,
                retries=retries,
            )
            ok += 1
        except Exception as exc:
            failed.append(remote_path)
            print(f"ERROR: {remote_path}: {exc}")

    print("\n" + "=" * 60)
    print(f"Descarga finalizada. OK={ok} | Fallidos={len(failed)} | Total={total}")

    if failed:
        failed_file = out_dir / "failed_downloads.txt"
        failed_file.write_text("\n".join(failed), encoding="utf-8")
        print(f"Rutas fallidas guardadas en: {failed_file}")
        print("Puedes relanzar el script; los archivos ya descargados se saltan automáticamente.")


def download_month_zip(session: requests.Session, month: str, out_dir: Path, overwrite: bool = False) -> Path:
    """Intenta descargar el mes completo como ZIP. No siempre funciona en Synology para carpetas grandes."""
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_name = f"{month}.zip"
    out_zip = out_dir / zip_name

    if out_zip.exists() and out_zip.stat().st_size > 0 and not overwrite:
        print(f"Ya existe {out_zip}. Usa --overwrite para descargarlo de nuevo.")
        return out_zip

    remote_month_path = f"/June-Aug/{month}"
    body = {
        "api": "SYNO.FolderSharing.Download",
        "method": "download",
        "version": "2",
        "mode": quoted("download"),
        "stdhtml": "false",
        "dlname": quoted(zip_name),
        "path": f'["{remote_month_path}"]',
        "_sharing_id": quoted(SHARING_ID),
        "codepage": quoted("spn"),
    }

    url = f"{BASE}/fsdownload/webapi/file_download.cgi/{zip_name}"
    print(f"\nIntentando descargar ZIP {remote_month_path} -> {out_zip}")

    tmp_zip = out_zip.with_suffix(".zip.part")
    with session.post(url, data=body, headers={"Origin": BASE}, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        iterator = response.iter_content(chunk_size=1024 * 1024)
        first_chunk = next(iterator, b"")
        if not first_chunk:
            raise RuntimeError("Respuesta vacía descargando el ZIP")

        content_type = response.headers.get("content-type", "").lower()
        if "application/json" in content_type or "text/html" in content_type:
            preview = first_chunk[:500].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"El ZIP no parece un binario. Content-Type={content_type}. "
                f"Respuesta inicial: {preview}"
            )

        with tmp_zip.open("wb") as file:
            file.write(first_chunk)
            for chunk in iterator:
                if chunk:
                    file.write(chunk)

    tmp_zip.replace(out_zip)
    print(f"ZIP guardado en: {out_zip} ({out_zip.stat().st_size / 1024 / 1024:.1f} MB)")
    return out_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lista y/o descarga el dataset SION desde Synology Sharing.")
    parser.add_argument("--month", default="07", help="Mes dentro de /June-Aug. Ejemplo: 06, 07, 08.")
    parser.add_argument("--out", default="data/sion", help="Carpeta local de salida. Default: data/sion.")
    parser.add_argument("--download", action="store_true", help="Descarga archivo a archivo preservando carpetas.")
    parser.add_argument("--zip", action="store_true", help="Intenta descargar el mes completo como ZIP.")
    parser.add_argument("--list-only", action="store_true", help="Solo lista archivos y guarda el TXT.")
    parser.add_argument("--no-list", action="store_true", help="No hace listado nuevo. Usa files_month_MM.txt existente.")
    parser.add_argument("--pattern", default="11_Cam0.jpg", help="Descarga solo archivos cuyo nombre termine con este texto. Default: 11_Cam0.jpg.")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe archivos existentes.")
    parser.add_argument("--retries", type=int, default=3, help="Reintentos por archivo. Default: 3.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    month = str(args.month).zfill(2)
    out_dir = Path(args.out)
    listing_file = out_dir / f"files_month_{month}.txt"

    if not SHARING_ID or not PASSWORD:
        print("Faltan credenciales. Define SION_SHARING_ID y SION_PASSWORD o edita el script.", file=sys.stderr)
        return 1

    try:
        session = login()

        files: list[str] = []
        if args.no_list:
            if not listing_file.exists():
                raise FileNotFoundError(f"No existe el listado previo: {listing_file}")
            files = [line.strip() for line in listing_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            print(f"Listado cargado desde: {listing_file} ({len(files)} archivos)")
        else:
            files = list(iter_month_file_paths(session, month))
            save_listing(files, month, out_dir)

        # Filtra solo los archivos deseados, por defecto los que acaban en 11_Cam0.jpg
        if args.pattern:
            before = len(files)
           
            #files = [f for f in files if Path(f).name.endswith(args.pattern)]
            files = [f for f in files if is_every_10_minutes_cam0(f)]
            print(f"\nFiltro aplicado: *{args.pattern}")
            print(f"Archivos seleccionados: {len(files)} de {before}")

            filtered_listing_file = out_dir / f"files_month_{month}_filtered_{args.pattern.replace('.', '_')}.txt"
            out_dir.mkdir(parents=True, exist_ok=True)
            filtered_listing_file.write_text("\n".join(files), encoding="utf-8")
            print(f"Listado filtrado guardado en: {filtered_listing_file}")

        if args.list_only:
            return 0

        if args.zip:
            download_month_zip(session, month, out_dir, overwrite=args.overwrite)

        if args.download:
            download_files(session, files, out_dir, overwrite=args.overwrite, retries=args.retries)

        if not args.download and not args.zip and not args.list_only:
            print("No se ha pedido descarga. Usa --download para bajar imágenes o --zip para intentar el ZIP.")

    except requests.HTTPError as exc:
        print(f"Error HTTP: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Error de conexión: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
