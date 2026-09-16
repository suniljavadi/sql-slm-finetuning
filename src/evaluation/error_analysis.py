from __future__ import annotations

from collections import defaultdict


def summarize_failures(failures: list[dict]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in failures:
        category = item.get("category", "unknown")
        grouped[category].append(item.get("message", "unknown failure"))
    return dict(grouped)
