"""Cluster saved CNN embeddings and compare them with PV-production labels."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABELS_BY_CLUSTER_COUNT = {
    5: ["very_low", "low", "medium", "high", "very_high"],
    3: ["low", "medium", "high"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run unsupervised KMeans on ResNet/VGG embeddings to see which visual "
            "clusters the images fall into."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing pv_clusters_* folders.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18", "resnet50", "vgg16"],
        help="Model feature sets to cluster.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="PCA dimensions used before KMeans.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Base path used to resolve relative image paths.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "none"],
        default="copy",
        help="Whether to copy visual-cluster images into folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, dict[str, object]] = {}

    for model_name in args.models:
        model_dir = args.output_root / f"pv_clusters_{model_name}"
        features_path = model_dir / f"features_{model_name}.npz"
        if not features_path.exists():
            print(f"Skipping {model_name}: missing {features_path}")
            continue

        data = np.load(features_path, allow_pickle=True)
        features = data["features"].astype(np.float32)
        metadata = pd.DataFrame(
            {
                "row_id": data["row_id"],
                "image_path": data["image_path"].astype(str),
                "production": data["production"].astype(float),
                "pv_label_5": data["label_5"].astype(str),
                "pv_label_3": data["label_3"].astype(str),
            }
        )

        visual_dir = model_dir / "visual_kmeans_clusters"
        visual_dir.mkdir(parents=True, exist_ok=True)
        summary[model_name] = {}

        for cluster_count, ordered_names in LABELS_BY_CLUSTER_COUNT.items():
            result = run_visual_clustering(
                features=features,
                metadata=metadata,
                cluster_count=cluster_count,
                ordered_names=ordered_names,
                pca_components=args.pca_components,
                random_state=args.random_state,
            )
            task_key = f"{cluster_count}_clusters"
            summary[model_name][task_key] = result["metrics"]

            task_dir = visual_dir / task_key
            task_dir.mkdir(parents=True, exist_ok=True)
            result["assignments"].to_csv(task_dir / "visual_cluster_assignments.csv", index=False)
            result["confusion"].to_csv(task_dir / "confusion_vs_pv_labels.csv")
            with (task_dir / "metrics.json").open("w", encoding="utf-8") as file:
                json.dump(result["metrics"], file, indent=2)
            if args.copy_mode == "copy":
                copy_visual_cluster_images(
                    assignments=result["assignments"],
                    task_dir=task_dir,
                    ordered_names=ordered_names,
                    project_root=args.project_root,
                )

            print(
                f"{model_name} {task_key}: "
                f"ordered_label_accuracy={result['metrics']['ordered_label_accuracy']:.4f}, "
                f"ARI={result['metrics']['adjusted_rand_index']:.4f}, "
                f"NMI={result['metrics']['normalized_mutual_info']:.4f}"
            )

    with (args.output_root / "visual_kmeans_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def run_visual_clustering(
    features: np.ndarray,
    metadata: pd.DataFrame,
    cluster_count: int,
    ordered_names: list[str],
    pca_components: int,
    random_state: int,
) -> dict[str, object]:
    components = min(pca_components, features.shape[1], features.shape[0] - 1)
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=components, random_state=random_state),
        MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=random_state,
            batch_size=1024,
            n_init=20,
        ),
    )

    cluster_ids = pipeline.fit_predict(features)
    assignments = metadata.copy()
    assignments["visual_cluster_id"] = cluster_ids

    cluster_order = (
        assignments.groupby("visual_cluster_id")["production"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    cluster_to_name = {cluster_id: ordered_names[index] for index, cluster_id in enumerate(cluster_order)}
    assignments["visual_cluster_label"] = assignments["visual_cluster_id"].map(cluster_to_name)

    pv_column = f"pv_label_{cluster_count}"
    confusion = pd.DataFrame(
        confusion_matrix(
            assignments[pv_column],
            assignments["visual_cluster_label"],
            labels=ordered_names,
        ),
        index=ordered_names,
        columns=ordered_names,
    )
    cluster_stats = assignments.groupby(["visual_cluster_id", "visual_cluster_label"]).agg(
        count=("production", "size"),
        production_mean=("production", "mean"),
        production_min=("production", "min"),
        production_max=("production", "max"),
    )

    metrics = {
        "ordered_label_accuracy": float(
            accuracy_score(assignments[pv_column], assignments["visual_cluster_label"])
        ),
        "adjusted_rand_index": float(
            adjusted_rand_score(assignments[pv_column], assignments["visual_cluster_id"])
        ),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(assignments[pv_column], assignments["visual_cluster_id"])
        ),
        "cluster_name_mapping": {str(key): value for key, value in cluster_to_name.items()},
        "cluster_stats": {
            f"{cluster_id}_{label}": {
                "count": int(row["count"]),
                "production_mean": float(row["production_mean"]),
                "production_min": float(row["production_min"]),
                "production_max": float(row["production_max"]),
            }
            for (cluster_id, label), row in cluster_stats.iterrows()
        },
    }

    return {
        "assignments": assignments,
        "confusion": confusion,
        "metrics": metrics,
    }


def copy_visual_cluster_images(
    assignments: pd.DataFrame,
    task_dir: Path,
    ordered_names: list[str],
    project_root: Path,
) -> None:
    image_root = task_dir / "images_by_visual_cluster"
    if image_root.exists():
        shutil.rmtree(image_root)
    image_root.mkdir(parents=True, exist_ok=True)

    for label in ordered_names:
        (image_root / label).mkdir(parents=True, exist_ok=True)

    root = project_root.resolve()
    for row in assignments.itertuples(index=False):
        source = Path(str(row.image_path))
        if not source.is_absolute():
            source = root / source
        source = source.resolve()
        if not source.exists():
            continue

        label = str(row.visual_cluster_label)
        destination = unique_destination(image_root / label, source.name)
        shutil.copy2(source, destination)


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


if __name__ == "__main__":
    main()
