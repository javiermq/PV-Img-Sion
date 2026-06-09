"""Download only original SION images referenced by metadata image_path values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

import sion_dataset_download_cam11 as sion  # noqa: E402


def remote_from_processed_path(processed_path: str) -> str:
    rel = Path(processed_path)
    parts = rel.parts
    try:
        index = parts.index("procesadas")
    except ValueError as exc:
        raise ValueError(f"Expected path under data/procesadas: {processed_path}") from exc

    tail = Path(*parts[index + 1 :])
    month, day, filename = tail.parts[0], tail.parts[1], tail.name
    stem = Path(filename).stem
    if stem.endswith("_crop"):
        stem = stem[: -len("_crop")]
    return f"/June-Aug/{month}/{day}/{stem}.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download originals referenced by weather_with_images.tsv.")
    parser.add_argument("--metadata", type=Path, default=Path("data/weather_with_images.tsv"))
    parser.add_argument("--out", type=Path, default=Path("data/sion"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.metadata, sep="\t")
    paths = sorted(
        {
            str(value).strip()
            for value in df["image_path"].dropna().tolist()
            if str(value).strip().startswith("data/procesadas/")
        }
    )
    remote_paths = [remote_from_processed_path(path) for path in paths]

    args.out.mkdir(parents=True, exist_ok=True)
    listing = args.out / "files_from_metadata.txt"
    listing.write_text("\n".join(remote_paths), encoding="utf-8")
    print(f"Originals to download: {len(remote_paths)}", flush=True)
    print(f"Listing written: {listing}", flush=True)

    session = sion.login()
    sion.download_files(
        session=session,
        files=remote_paths,
        out_dir=args.out,
        overwrite=args.overwrite,
        retries=args.retries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
