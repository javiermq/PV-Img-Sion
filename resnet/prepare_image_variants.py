"""Regenerate processed sky-image variants from the original SION images.

This keeps the same manual crop geometry used by ``crop_then_project_fisheye.py``
and only changes the final output resolution:

* ``512`` writes 512x512 crops.
* ``original`` writes the crop without the final resize.

It also writes metadata TSV variants whose ``image_path`` column points at the
new processed-image folders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from crop_then_project_fisheye import process_image  # noqa: E402


VARIANTS = {
    "512": ("data/procesadas_512", 512),
    "original": ("data/procesadas_original", None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 512/original processed image variants and matching TSV files."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/weather_with_images.tsv"),
        help="Input TSV with image_path values pointing to data/procesadas.",
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path("data/sion/June-Aug"),
        help="Root containing original images organized as MM/DD/*.jpg.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Base path for relative metadata and image paths.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        action="append",
        help="Variant to generate. Can be passed multiple times. Default: all.",
    )
    parser.add_argument("--cx", type=float, default=2000)
    parser.add_argument("--cy", type=float, default=1600)
    parser.add_argument("--r", type=float, default=800)
    parser.add_argument("--crop-pad", type=float, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate images even if the destination file already exists.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N generated/skipped images.",
    )
    return parser.parse_args()


def source_from_processed_path(processed_path: str, original_root: Path) -> Path:
    rel = Path(processed_path)
    parts = rel.parts
    try:
        index = parts.index("procesadas")
    except ValueError as exc:
        raise ValueError(f"Expected image_path under data/procesadas: {processed_path}") from exc

    tail = Path(*parts[index + 1 :])
    if len(tail.parts) < 3:
        raise ValueError(f"Expected path like data/procesadas/MM/DD/file_crop.jpg: {processed_path}")

    month, day, filename = tail.parts[0], tail.parts[1], tail.name
    stem = Path(filename).stem
    if stem.endswith("_crop"):
        stem = stem[: -len("_crop")]

    return original_root / month / day / f"{stem}.jpg"


def variant_path(processed_path: str, variant_root: str) -> str:
    return processed_path.replace("data/procesadas/", f"{variant_root}/", 1)


def write_variant_metadata(df: pd.DataFrame, metadata_path: Path, variant: str, variant_root: str) -> Path:
    out_path = metadata_path.with_name(f"{metadata_path.stem}_{variant}{metadata_path.suffix}")
    out_df = df.copy()
    mask = out_df["image_path"].fillna("").astype(str).str.startswith("data/procesadas/")
    out_df.loc[mask, "image_path"] = out_df.loc[mask, "image_path"].astype(str).map(
        lambda value: variant_path(value, variant_root)
    )
    out_df.to_csv(out_path, sep="\t", index=False)
    return out_path


def generate_variant(
    df: pd.DataFrame,
    project_root: Path,
    original_root: Path,
    variant: str,
    overwrite: bool,
    cx: float,
    cy: float,
    r: float,
    crop_pad: float,
    progress_every: int,
) -> tuple[int, int, list[str]]:
    variant_root, crop_size = VARIANTS[variant]
    paths = sorted(
        {
            str(value).strip()
            for value in df["image_path"].dropna().tolist()
            if str(value).strip().startswith("data/procesadas/")
        }
    )

    made_or_existing = 0
    missing: list[str] = []

    for index, processed in enumerate(paths, start=1):
        source = project_root / source_from_processed_path(processed, original_root)
        destination = project_root / variant_path(processed, variant_root)

        if not source.exists():
            missing.append(str(source))
            continue

        if destination.exists() and not overwrite:
            made_or_existing += 1
        else:
            process_image(
                image_path=source,
                output_dir=destination.parent,
                cx=cx,
                cy=cy,
                r=r,
                crop_pad=crop_pad,
                crop_size=crop_size,
                save_crop=True,
                crop_alpha=False,
                debug=False,
            )
            made_or_existing += 1

        if index == 1 or index % progress_every == 0 or index == len(paths):
            print(
                f"[{variant}] {index}/{len(paths)} checked | "
                f"ready={made_or_existing} | missing={len(missing)}",
                flush=True,
            )

    return len(paths), made_or_existing, missing


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    metadata_path = project_root / args.metadata
    original_root = args.original_root

    df = pd.read_csv(metadata_path, sep="\t")
    if "image_path" not in df.columns:
        raise ValueError("Metadata TSV must contain image_path.")

    variants = args.variant or sorted(VARIANTS)

    for variant in variants:
        variant_root, _ = VARIANTS[variant]
        out_tsv = write_variant_metadata(df, metadata_path, variant, variant_root)
        print(f"[{variant}] metadata written: {out_tsv}")

        total, ready, missing = generate_variant(
            df=df,
            project_root=project_root,
            original_root=original_root,
            variant=variant,
            overwrite=args.overwrite,
            cx=args.cx,
            cy=args.cy,
            r=args.r,
            crop_pad=args.crop_pad,
            progress_every=args.progress_every,
        )
        print(f"[{variant}] done: ready={ready}/{total}, missing={len(missing)}")
        if missing:
            missing_file = project_root / f"missing_originals_{variant}.txt"
            missing_file.write_text("\n".join(missing), encoding="utf-8")
            print(f"[{variant}] missing list written: {missing_file}")


if __name__ == "__main__":
    main()
