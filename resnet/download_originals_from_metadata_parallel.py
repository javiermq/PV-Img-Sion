"""Parallel download of original SION images referenced by metadata image_path values."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

import sion_dataset_download_cam11 as sion  # noqa: E402

_thread_local = threading.local()


def remote_from_processed_path(processed_path: str) -> str:
    rel = Path(processed_path)
    parts = rel.parts
    index = parts.index("procesadas")
    tail = Path(*parts[index + 1 :])
    month, day, filename = tail.parts[0], tail.parts[1], tail.name
    stem = Path(filename).stem
    if stem.endswith("_crop"):
        stem = stem[: -len("_crop")]
    return f"/June-Aug/{month}/{day}/{stem}.jpg"


def get_session():
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = sion.login()
        _thread_local.session = session
    return session


def local_path_for_remote(remote_path: str, out_dir: Path) -> Path:
    return out_dir / remote_path.lstrip("/")


def download_one(remote_path: str, out_dir: Path, overwrite: bool, retries: int) -> tuple[str, str]:
    local_file = local_path_for_remote(remote_path, out_dir)
    if local_file.exists() and local_file.stat().st_size > 0 and not overwrite:
        return remote_path, "skip"

    session = get_session()
    sion.download_one_file(
        session=session,
        remote_path=remote_path,
        out_dir=out_dir,
        overwrite=overwrite,
        retries=retries,
    )
    return remote_path, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel download originals referenced by weather_with_images.tsv.")
    parser.add_argument("--metadata", type=Path, default=Path("data/weather_with_images.tsv"))
    parser.add_argument("--out", type=Path, default=Path("data/sion"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=100)
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

    total = len(remote_paths)
    print(f"Originals to download: {total}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(f"Listing written: {listing}", flush=True)

    ok = 0
    skipped = 0
    failed: list[str] = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, remote, args.out, args.overwrite, args.retries): remote
            for remote in remote_paths
        }
        for index, future in enumerate(as_completed(futures), start=1):
            remote = futures[future]
            try:
                _, status = future.result()
                if status == "skip":
                    skipped += 1
                else:
                    ok += 1
            except Exception as exc:
                failed.append(remote)
                print(f"ERROR {remote}: {exc}", flush=True)

            if index == 1 or index % args.progress_every == 0 or index == total:
                elapsed = max(time.time() - start, 1e-6)
                rate = index / elapsed
                remaining = (total - index) / rate if rate else 0
                print(
                    f"[{index}/{total}] ok={ok} skipped={skipped} failed={len(failed)} "
                    f"rate={rate:.2f}/s eta={remaining/60:.1f} min",
                    flush=True,
                )

    if failed:
        failed_file = args.out / "failed_downloads_parallel.txt"
        failed_file.write_text("\n".join(failed), encoding="utf-8")
        print(f"Failed list written: {failed_file}", flush=True)
        return 1

    print(f"Done. ok={ok} skipped={skipped} failed=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
