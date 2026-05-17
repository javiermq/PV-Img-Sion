#!/usr/bin/env python3
"""
Descarga/listado del dataset SION desde Synology Sharing, sin Jupyter.

Uso típico:
    python sion_dataset_download.py --month 07 --download
    python sion_dataset_download.py --month 06 --list-only
    python sion_dataset_download.py --month 07 --download --out data/sion

Por defecto:
  - crea la carpeta de salida si no existe
  - lista los archivos del mes
  - guarda el listado en files_month_MM.txt
  - si se pasa --download, descarga /June-Aug/MM como MM.zip
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import requests


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


def list_month_files(session: requests.Session, month: str) -> list[str]:
    month_path = f"/June-Aug/{month}"
    print(f"Listando carpeta remota: {month_path}")

    days = list_folder(session, month_path)
    all_paths: list[str] = []

    for day in days:
        if not day.get("isdir"):
            continue

        day_name = day["name"]
        day_path = f"{month_path}/{day_name}"

        print(f"\n📁 {day_path}")
        files = list_folder(session, day_path)

        for file_info in files:
            name = file_info["name"]
            remote_file_path = f"{day_path}/{name}"
            size = file_info.get("size", "")
            filetype = file_info.get("type", "")

            print(f"  {remote_file_path} | size={size} | type={filetype}")
            all_paths.append(remote_file_path)

    return all_paths


def save_listing(files: list[str], month: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"files_month_{month}.txt"
    out_file.write_text("\n".join(files), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Total archivos en mes {month}: {len(files)}")
    print(f"Listado guardado en: {out_file}")

    return out_file


def download_month_zip(session: requests.Session, month: str, out_dir: Path, overwrite: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_name = f"{month}.zip"
    out_zip = out_dir / zip_name

    if out_zip.exists() and not overwrite:
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

    print(f"\nDescargando {remote_month_path} → {out_zip}")

    with session.post(
        url,
        data=body,
        headers={"Origin": BASE},
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with out_zip.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024 * 8):
                if not chunk:
                    continue

                file.write(chunk)
                downloaded += len(chunk)

                if total:
                    print(f"\r{downloaded / total * 100:.1f}% descargado", end="", flush=True)
                else:
                    print(
                        f"\r{downloaded / 1024 / 1024:.1f} MB descargados",
                        end="",
                        flush=True,
                    )

    print(f"\nDescarga terminada: {out_zip}")
    return out_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lista y/o descarga el dataset SION desde Synology Sharing."
    )
    parser.add_argument(
        "--month",
        default="07",
        help="Mes a procesar dentro de /June-Aug. Ejemplo: 06, 07, 08. Default: 07.",
    )
    parser.add_argument(
        "--out",
        default="data/sion",
        help="Carpeta local donde guardar listados y ZIPs. Default: data/sion.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Descarga la carpeta mensual como ZIP.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Solo lista archivos y guarda el TXT; no descarga ZIP.",
    )
    parser.add_argument(
        "--no-list",
        action="store_true",
        help="No hace listado previo; útil si solo quieres descargar el ZIP.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe el ZIP si ya existe.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    month = str(args.month).zfill(2)
    out_dir = Path(args.out)

    if not SHARING_ID or not PASSWORD:
        print(
            "Faltan credenciales. Define SION_SHARING_ID y SION_PASSWORD "
            "o edita las constantes del script.",
            file=sys.stderr,
        )
        return 1

    try:
        session = login()

        if not args.no_list:
            files = list_month_files(session, month)
            save_listing(files, month, out_dir)

        if args.download and not args.list_only:
            download_month_zip(session, month, out_dir, overwrite=args.overwrite)

        if not args.download and args.no_list:
            print("No se ha pedido ninguna acción. Usa --download o quita --no-list.")

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
