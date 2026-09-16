from __future__ import annotations

from typing import Any


def syntax_validity(sql: str) -> bool:
    return bool(sql and isinstance(sql, str) and sql.strip())


def execution_accuracy(expected: Any, actual: Any) -> bool:
    return expected == actual


def compute_metrics(expected: Any, actual: Any) -> dict[str, float | bool]:
    return {
        "syntax_valid": syntax_validity(actual),
        "execution_accuracy": execution_accuracy(expected, actual),
        "exact_match": expected == actual,
    }
