from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.data.validate_dataset import validate_dataset_file
from src.evaluation.metrics import compute_metrics


def evaluate_dataset(
    dataset_path: str | Path,
    generator: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    rows = validate_dataset_file(dataset_path)
    results: list[dict[str, Any]] = []
    for row in rows:
        expected = row["output"]
        actual = generator(row)
        results.append({
            "instruction": row["instruction"],
            "expected": expected,
            "actual": actual,
            "metrics": compute_metrics(expected, actual),
        })
    return results
