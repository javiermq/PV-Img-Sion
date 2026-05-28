"""Evaluate whether CNN feature vectors separate PV production classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple classifier on saved CNN features to test PV class separability."
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
        help="Model names to evaluate.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the stratified split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_results: dict[str, dict[str, object]] = {}

    for model_name in args.models:
        model_dir = args.output_root / f"pv_clusters_{model_name}"
        features_path = model_dir / f"features_{model_name}.npz"
        if not features_path.exists():
            print(f"Skipping {model_name}: missing {features_path}")
            continue

        data = np.load(features_path, allow_pickle=True)
        features = data["features"]
        labels_by_task = {
            "5_classes": data["label_5"].astype(str),
            "3_classes": data["label_3"].astype(str),
        }

        evaluation_dir = model_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        all_results[model_name] = {}

        for task_name, labels in labels_by_task.items():
            result = evaluate_task(features, labels, args.test_size, args.random_state)
            all_results[model_name][task_name] = result["metrics"]

            prefix = evaluation_dir / task_name
            pd.DataFrame(result["confusion_matrix"], index=result["labels"], columns=result["labels"]).to_csv(
                prefix.with_name(f"{task_name}_confusion_matrix.csv")
            )
            with prefix.with_name(f"{task_name}_classification_report.json").open("w", encoding="utf-8") as file:
                json.dump(result["classification_report"], file, indent=2)
            with prefix.with_name(f"{task_name}_metrics.json").open("w", encoding="utf-8") as file:
                json.dump(result["metrics"], file, indent=2)

            print(
                f"{model_name} {task_name}: "
                f"accuracy={result['metrics']['accuracy']:.4f}, "
                f"balanced_accuracy={result['metrics']['balanced_accuracy']:.4f}, "
                f"macro_f1={result['metrics']['macro_f1']:.4f}"
            )

    with (args.output_root / "feature_separability_summary.json").open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)


def evaluate_task(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> dict[str, object]:
    label_order = list(dict.fromkeys(labels))
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    classifier = make_pipeline(
        StandardScaler(),
        RidgeClassifier(class_weight="balanced"),
    )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    report = classification_report(y_test, y_pred, labels=label_order, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=label_order)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "labels": label_order,
    }
    return {
        "labels": label_order,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


if __name__ == "__main__":
    main()
