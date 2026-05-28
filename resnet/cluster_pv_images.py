"""Cluster sky images by PV production and extract CNN features.

The labels are derived from the numeric ``production`` column:

* 5 classes: very_low, low, medium, high, very_high
* 3 classes: low, medium, high

Images are copied into one folder per class and a final feature matrix is saved
for downstream work.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


LABELS_BY_CLUSTER_COUNT = {
    5: ["very_low", "low", "medium", "high", "very_high"],
    3: ["low", "medium", "high"],
}


@dataclass(frozen=True)
class ImageRecord:
    row_id: int
    timestamp: str
    image_path: Path
    image_path_source: str
    production: float
    label_5: str
    label_3: str


@dataclass(frozen=True)
class PvHourFilter:
    enabled: bool
    min_production: float
    min_hour_max_production_ratio: float
    active_hours: str | None
    rows_before: int = 0
    rows_after: int = 0
    kept_hours: tuple[int, ...] = ()
    removed_hours: tuple[int, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign PV-production classes to sky images, copy images into class "
            "folders, and save ResNet/VGG16 feature vectors."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/weather_with_images.tsv"),
        help="TSV file with production and image_path columns.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Base path used to resolve relative image_path values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pv_clusters"),
        help="Directory where class folders, assignments, and features are written.",
    )
    parser.add_argument(
        "--model",
        choices=["resnet18", "resnet50", "vgg16"],
        default="resnet50",
        help="CNN backbone used to extract final feature vectors.",
    )
    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Use ImageNet pretrained weights or random weights.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink", "none"],
        default="copy",
        help="How to place images into class folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    parser.add_argument(
        "--save-feature-csv",
        action="store_true",
        help="Also save a wide CSV with one feature column per dimension.",
    )
    parser.add_argument(
        "--torch-cache-dir",
        type=Path,
        default=None,
        help="Directory for downloaded torchvision model weights.",
    )
    parser.add_argument(
        "--no-filter-pv-hours",
        action="store_true",
        help="Disable the default filter that removes hours without useful PV production.",
    )
    parser.add_argument(
        "--min-production",
        type=float,
        default=0.0,
        help="Drop rows with production less than or equal to this value.",
    )
    parser.add_argument(
        "--min-hour-max-production-ratio",
        type=float,
        default=0.05,
        help=(
            "When active hours are inferred automatically, keep only hours whose "
            "maximum production is at least this fraction of the global maximum."
        ),
    )
    parser.add_argument(
        "--active-hours",
        default=None,
        help=(
            "Optional UTC hour window override, for example 6-18. "
            "The end hour is inclusive."
        ),
    )
    return parser.parse_args()


def read_records(
    metadata_path: Path,
    project_root: Path,
    pv_hour_filter: PvHourFilter | None = None,
) -> tuple[list[ImageRecord], PvHourFilter | None]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path, sep="\t")
    required = {"production", "image_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    df = df.copy()
    df["row_id"] = np.arange(len(df))
    df["production"] = pd.to_numeric(df["production"], errors="coerce")
    df["image_path"] = df["image_path"].fillna("").astype(str).str.strip()
    df = df[(df["image_path"] != "") & df["production"].notna()].copy()
    df["timestamp"] = pd.to_datetime(df.get("timestamp", ""), errors="coerce")

    if df.empty:
        raise ValueError("No rows have both image_path and numeric production.")

    if pv_hour_filter and pv_hour_filter.enabled:
        df, pv_hour_filter = apply_pv_hour_filter(df, pv_hour_filter)

    df["label_5"] = make_quantile_labels(df["production"], LABELS_BY_CLUSTER_COUNT[5])
    df["label_3"] = make_quantile_labels(df["production"], LABELS_BY_CLUSTER_COUNT[3])

    records: list[ImageRecord] = []
    missing_files: list[str] = []
    root = project_root.resolve()

    for row in df.itertuples(index=False):
        raw_image_path = getattr(row, "image_path")
        image_path = Path(raw_image_path)
        if not image_path.is_absolute():
            image_path = root / image_path
        image_path = image_path.resolve()

        if not image_path.exists():
            missing_files.append(raw_image_path)
            continue

        records.append(
            ImageRecord(
                row_id=int(getattr(row, "row_id")),
                timestamp=str(getattr(row, "timestamp", "")),
                image_path=image_path,
                image_path_source=raw_image_path,
                production=float(getattr(row, "production")),
                label_5=str(getattr(row, "label_5")),
                label_3=str(getattr(row, "label_3")),
            )
        )

    if not records:
        raise FileNotFoundError("No image_path values resolved to existing files.")

    if missing_files:
        print(f"Warning: skipped {len(missing_files)} rows with missing image files.")

    return records, pv_hour_filter


def apply_pv_hour_filter(df: pd.DataFrame, pv_hour_filter: PvHourFilter) -> tuple[pd.DataFrame, PvHourFilter]:
    rows_before = len(df)
    df = df[df["production"] > pv_hour_filter.min_production].copy()

    if "timestamp" not in df or df["timestamp"].isna().all():
        raise ValueError("PV hour filtering requires a parseable timestamp column.")

    df["hour_utc"] = df["timestamp"].dt.hour
    available_hours = sorted(int(hour) for hour in df["hour_utc"].dropna().unique())

    if pv_hour_filter.active_hours:
        kept_hours = parse_active_hours(pv_hour_filter.active_hours)
    else:
        max_production = float(df["production"].max())
        min_hour_max = max_production * pv_hour_filter.min_hour_max_production_ratio
        hour_max = df.groupby("hour_utc")["production"].max()
        kept_hours = sorted(int(hour) for hour, value in hour_max.items() if float(value) >= min_hour_max)

    filtered = df[df["hour_utc"].isin(kept_hours)].copy()
    removed_hours = tuple(hour for hour in available_hours if hour not in kept_hours)

    updated_filter = PvHourFilter(
        enabled=pv_hour_filter.enabled,
        min_production=pv_hour_filter.min_production,
        min_hour_max_production_ratio=pv_hour_filter.min_hour_max_production_ratio,
        active_hours=pv_hour_filter.active_hours,
        rows_before=rows_before,
        rows_after=len(filtered),
        kept_hours=tuple(kept_hours),
        removed_hours=removed_hours,
    )
    return filtered, updated_filter


def parse_active_hours(active_hours: str) -> list[int]:
    try:
        start_text, end_text = active_hours.split("-", maxsplit=1)
        start_hour = int(start_text)
        end_hour = int(end_text)
    except ValueError as exc:
        raise ValueError("--active-hours must look like 6-18") from exc

    if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
        raise ValueError("--active-hours values must be between 0 and 23")
    if end_hour < start_hour:
        raise ValueError("--active-hours end hour must be greater than or equal to start hour")

    return list(range(start_hour, end_hour + 1))


def make_quantile_labels(values: pd.Series, labels: list[str]) -> pd.Series:
    """Return equal-frequency labels ordered by production."""
    ranks = values.rank(method="first")
    codes = pd.qcut(ranks, q=len(labels), labels=False)
    return codes.map({i: label for i, label in enumerate(labels)})


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def place_images(records: list[ImageRecord], output_dir: Path, copy_mode: str) -> dict[str, dict[str, int]]:
    summaries: dict[str, dict[str, int]] = {}

    for cluster_count, labels in LABELS_BY_CLUSTER_COUNT.items():
        cluster_dir = output_dir / f"{cluster_count}_classes"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        summaries[f"{cluster_count}_classes"] = {}

        for label in labels:
            label_dir = cluster_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            summaries[f"{cluster_count}_classes"][label] = 0

        for record in records:
            label = record.label_5 if cluster_count == 5 else record.label_3
            destination = unique_destination(cluster_dir / label, record.image_path.name)

            if copy_mode != "none":
                put_file(record.image_path, destination, copy_mode)

            summaries[f"{cluster_count}_classes"][label] += 1

    return summaries


def unique_destination(folder: Path, filename: str) -> Path:
    destination = folder / filename
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while True:
        candidate = folder / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def put_file(source: Path, destination: Path, copy_mode: str) -> None:
    if copy_mode == "copy":
        shutil.copy2(source, destination)
    elif copy_mode == "hardlink":
        try:
            destination.hardlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
    elif copy_mode == "symlink":
        try:
            destination.symlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported copy mode: {copy_mode}")


def load_torch_model(model_name: str, weights_name: str):
    import torch
    from torch import nn
    from torchvision import models

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if weights_name == "imagenet" else None
        model = models.resnet18(weights=weights)
        transform = weights.transforms() if weights else default_transform()
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if weights_name == "imagenet" else None
        model = models.resnet50(weights=weights)
        transform = weights.transforms() if weights else default_transform()
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if weights_name == "imagenet" else None
        model = models.vgg16(weights=weights)
        transform = weights.transforms() if weights else default_transform()
        feature_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return torch, model, transform, feature_dim


def default_transform() -> Callable:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class SkyImageDataset:
    def __init__(self, records: list[ImageRecord], transform: Callable) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def resolve_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_features(
    records: list[ImageRecord],
    model_name: str,
    weights_name: str,
    batch_size: int,
    requested_device: str,
):
    torch, model, transform, feature_dim = load_torch_model(model_name, weights_name)
    from torch.utils.data import DataLoader

    device = resolve_device(torch, requested_device)
    model.to(device)

    dataset = SkyImageDataset(records, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features = np.zeros((len(records), feature_dim), dtype=np.float32)

    with torch.inference_mode():
        for batch_number, (images, indices) in enumerate(loader, start=1):
            images = images.to(device)
            output = model(images).detach().cpu().numpy().astype(np.float32)
            features[indices.numpy()] = output
            if batch_number == 1 or batch_number % 20 == 0:
                done = min(batch_number * batch_size, len(records))
                print(f"Extracted features for {done}/{len(records)} images")

    return features, str(device), feature_dim


def records_to_dataframe(records: list[ImageRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [record.row_id for record in records],
            "timestamp": [record.timestamp for record in records],
            "image_path": [record.image_path_source for record in records],
            "resolved_image_path": [str(record.image_path) for record in records],
            "production": [record.production for record in records],
            "label_5": [record.label_5 for record in records],
            "label_3": [record.label_3 for record in records],
        }
    )


def save_outputs(
    records: list[ImageRecord],
    features: np.ndarray,
    output_dir: Path,
    model_name: str,
    weights_name: str,
    device: str,
    feature_dim: int,
    class_counts: dict[str, dict[str, int]],
    save_feature_csv: bool,
    pv_hour_filter: PvHourFilter | None,
) -> None:
    metadata = records_to_dataframe(records)
    metadata_path = output_dir / "assignments.csv"
    metadata.to_csv(metadata_path, index=False)

    features_path = output_dir / f"features_{model_name}.npz"
    np.savez_compressed(
        features_path,
        features=features,
        row_id=metadata["row_id"].to_numpy(),
        image_path=metadata["image_path"].to_numpy(),
        production=metadata["production"].to_numpy(dtype=np.float32),
        label_5=metadata["label_5"].to_numpy(),
        label_3=metadata["label_3"].to_numpy(),
    )

    if save_feature_csv:
        feature_df = pd.DataFrame(features, columns=[f"feature_{i:04d}" for i in range(features.shape[1])])
        pd.concat([metadata, feature_df], axis=1).to_csv(output_dir / f"features_{model_name}.csv", index=False)

    class_feature_files = save_class_feature_outputs(metadata, features, output_dir, model_name)

    summary = {
        "model": model_name,
        "weights": weights_name,
        "device": device,
        "feature_dim": feature_dim,
        "images_used": len(records),
        "features_file": str(features_path),
        "assignments_file": str(metadata_path),
        "class_feature_files": class_feature_files,
        "class_counts": class_counts,
        "production_ranges": production_ranges(metadata),
        "pv_hour_filter": pv_hour_filter_to_dict(pv_hour_filter),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def save_class_feature_outputs(
    metadata: pd.DataFrame,
    features: np.ndarray,
    output_dir: Path,
    model_name: str,
) -> dict[str, dict[str, dict[str, str]]]:
    class_feature_files: dict[str, dict[str, dict[str, str]]] = {}
    feature_root = output_dir / "feature_vectors"

    for cluster_count, labels in LABELS_BY_CLUSTER_COUNT.items():
        cluster_key = f"{cluster_count}_classes"
        cluster_dir = feature_root / cluster_key
        cluster_dir.mkdir(parents=True, exist_ok=True)
        class_feature_files[cluster_key] = {}

        label_column = f"label_{cluster_count}"
        for label in labels:
            mask = metadata[label_column].to_numpy() == label
            class_metadata = metadata.loc[mask].reset_index(drop=True)
            class_features = features[mask]

            npz_path = cluster_dir / f"{label}_features_{model_name}.npz"
            csv_path = cluster_dir / f"{label}_assignments.csv"

            np.savez_compressed(
                npz_path,
                features=class_features,
                row_id=class_metadata["row_id"].to_numpy(),
                image_path=class_metadata["image_path"].to_numpy(),
                production=class_metadata["production"].to_numpy(dtype=np.float32),
                label=class_metadata[label_column].to_numpy(),
                label_5=class_metadata["label_5"].to_numpy(),
                label_3=class_metadata["label_3"].to_numpy(),
            )
            class_metadata.to_csv(csv_path, index=False)

            class_feature_files[cluster_key][label] = {
                "features": str(npz_path),
                "assignments": str(csv_path),
            }

    return class_feature_files


def production_ranges(metadata: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    ranges: dict[str, dict[str, dict[str, float]]] = {}
    for cluster_count in (5, 3):
        column = f"label_{cluster_count}"
        ranges[f"{cluster_count}_classes"] = {}
        for label in LABELS_BY_CLUSTER_COUNT[cluster_count]:
            values = metadata.loc[metadata[column] == label, "production"]
            ranges[f"{cluster_count}_classes"][label] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "count": int(values.size),
            }
    return ranges


def pv_hour_filter_to_dict(pv_hour_filter: PvHourFilter | None) -> dict[str, object] | None:
    if not pv_hour_filter:
        return None
    return {
        "enabled": pv_hour_filter.enabled,
        "min_production": pv_hour_filter.min_production,
        "min_hour_max_production_ratio": pv_hour_filter.min_hour_max_production_ratio,
        "active_hours": pv_hour_filter.active_hours,
        "rows_before": pv_hour_filter.rows_before,
        "rows_after": pv_hour_filter.rows_after,
        "kept_hours_utc": list(pv_hour_filter.kept_hours),
        "removed_hours_utc": list(pv_hour_filter.removed_hours),
    }


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    torch_cache_dir = args.torch_cache_dir or (args.output_dir / "torch_cache")
    os.environ.setdefault("TORCH_HOME", str(torch_cache_dir.resolve()))

    pv_hour_filter = PvHourFilter(
        enabled=not args.no_filter_pv_hours,
        min_production=args.min_production,
        min_hour_max_production_ratio=args.min_hour_max_production_ratio,
        active_hours=args.active_hours,
    )
    records, pv_hour_filter = read_records(args.metadata, args.project_root, pv_hour_filter)
    print(f"Loaded {len(records)} image records with production values.")
    if pv_hour_filter and pv_hour_filter.enabled:
        print(
            "PV hour filter kept UTC hours "
            f"{list(pv_hour_filter.kept_hours)} and removed {list(pv_hour_filter.removed_hours)}."
        )

    class_counts = place_images(records, args.output_dir, args.copy_mode)
    print(f"Created class folders under {args.output_dir}.")

    features, device, feature_dim = extract_features(
        records=records,
        model_name=args.model,
        weights_name=args.weights,
        batch_size=args.batch_size,
        requested_device=args.device,
    )

    save_outputs(
        records=records,
        features=features,
        output_dir=args.output_dir,
        model_name=args.model,
        weights_name=args.weights,
        device=device,
        feature_dim=feature_dim,
        class_counts=class_counts,
        save_feature_csv=args.save_feature_csv,
        pv_hour_filter=pv_hour_filter,
    )
    print(f"Done. Feature matrix shape: {features.shape}")


if __name__ == "__main__":
    main()
