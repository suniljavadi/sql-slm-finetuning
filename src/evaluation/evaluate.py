from __future__ import annotations

from pathlib import Path

from src.data.validate_dataset import validate_dataset_file
from src.evaluation.metrics import compute_metrics


def evaluate_dataset(dataset_path: str | Path) -> list[dict]:
    rows = validate_dataset_file(dataset_path)
    results = []
    for row in rows:
        expected = row["output"]
        actual = row["output"]
        results.append({
            "instruction": row["instruction"],
            "metrics": compute_metrics(expected, actual),
        })
    return results
