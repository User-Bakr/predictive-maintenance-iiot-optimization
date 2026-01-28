"""
ACO Project - Experiment Runner (safe skeleton)

- Supports binary and multi-class datasets (two Kaggle datasets)
- Includes the pipeline structure: load -> checks -> preprocess -> train/test -> evaluate
- ACO optimization is an outline only (publication restriction)
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing.data_loader import load_csv, basic_sanity_checks
from preprocessing.preprocess import PreprocessConfig, prepare_xy, enforce_machine_failure_from_components
from models.classifiers import ModelConfig, build_binary_classifiers, build_multiclass_classifiers
from optimization.aco import ACOConfig, optimize_with_aco


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Use macro F1 as a reasonable default (works for multi-class too)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
    }

    # Precision/recall are straightforward for binary; for multi-class use macro averaging
    metrics["precision_macro"] = float(precision_score(y_test, preds, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_test, preds, average="macro", zero_division=0))
    return metrics


def run_task(csv_path: str, target_col: str, task_type: str) -> None:
    df = load_csv(csv_path)
    basic_sanity_checks(df, id_cols=["Product ID"])  # adjust/remove for your dataset

    df = enforce_machine_failure_from_components(df, target_col=target_col)

    # Preprocess -> X/y scaled
    X, y, _ = prepare_xy(df, PreprocessConfig(target_col=target_col, sample_n=None))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Build models based on task
    mcfg = ModelConfig(random_state=42)
    if task_type == "binary":
        models = build_binary_classifiers(mcfg)
    elif task_type == "multiclass":
        models = build_multiclass_classifiers(mcfg)
    else:
        raise ValueError("task_type must be 'binary' or 'multiclass'")

    # Objective function for optimizer (minimize negative accuracy, for example)
    def objective_fn(model) -> float:
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        return -metrics["accuracy"]  # minimize

    # Optimize (outline)
    best_name, best_model, best_score = optimize_with_aco(
        models=models,
        objective_fn=objective_fn,
        cfg=ACOConfig(),
    )

    # Final evaluation for the chosen model
    final_metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test)

    print(f"\n=== Task: {task_type} | Best Model: {best_name} ===")
    print("Best score (objective):", best_score)
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")


def main():
    binary_csv_path = "path/to/binary_dataset.csv"
    multiclass_csv_path = "path/to/multiclass_dataset.csv"

    run_task(binary_csv_path, target_col="Machine failure", task_type="binary")
    run_task(multiclass_csv_path, target_col="Target", task_type="multiclass")


if __name__ == "__main__":
    main()
