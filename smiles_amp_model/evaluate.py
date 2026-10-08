"""Independent-test evaluation and stratified bootstrap confidence intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score


def metric_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def stratified_bootstrap(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5, n_boot: int = 10000, seed: int = 42) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    indices = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    values = {name: [] for name in ("accuracy", "f1", "roc_auc", "mcc")}
    for _ in range(n_boot):
        sampled = np.concatenate([rng.choice(indices[label], size=len(indices[label]), replace=True) for label in (0, 1)])
        metrics = metric_at_threshold(labels[sampled], probabilities[sampled], threshold)
        for name in values:
            if metrics[name] is not None:
                values[name].append(metrics[name])
    return {name: {"lower": float(np.percentile(result, 2.5)), "upper": float(np.percentile(result, 97.5))} for name, result in values.items()}


def evaluate_predictions(frame: pd.DataFrame, threshold: float = 0.5, n_boot: int = 10000, seed: int = 42) -> Dict[str, object]:
    label_col = "label" if "label" in frame.columns else "y_true"
    probability_col = "probability" if "probability" in frame.columns else "y_score"
    if label_col not in frame or probability_col not in frame:
        raise ValueError("Prediction table must contain label and probability columns")
    valid = frame[[label_col, probability_col]].dropna()
    labels = valid[label_col].astype(int).to_numpy()
    probabilities = valid[probability_col].astype(float).to_numpy()
    result = metric_at_threshold(labels, probabilities, threshold)
    result["confidence_interval_95"] = stratified_bootstrap(labels, probabilities, threshold, n_boot, seed)
    result["bootstrap"] = {"method": "stratified percentile bootstrap", "n_boot": n_boot, "seed": seed}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an independent scaffold-split test prediction table")
    parser.add_argument("--predictions", required=True, help="CSV containing label and probability")
    parser.add_argument("--output", default="evaluation.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = evaluate_predictions(pd.read_csv(args.predictions), args.threshold, args.n_boot, args.seed)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
