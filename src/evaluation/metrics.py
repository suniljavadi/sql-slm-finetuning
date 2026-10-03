from __future__ import annotations

import re
from typing import Any

from sqlglot import parse
from sqlglot.errors import ParseError

from src.inference.generate import validate_sql_output


SQL_PREFIX = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().rstrip(";").split()).casefold()


def _is_sql_target(value: Any) -> bool:
    return isinstance(value, str) and SQL_PREFIX.search(value) is not None


def _candidate_output(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    match = SQL_FENCE.search(value)
    return match.group(1).strip() if match else value.strip()


def syntax_validity(sql: str) -> bool:
    if not isinstance(sql, str) or not sql.strip():
        return False
    try:
        candidate = _candidate_output(sql)
        statements = [statement for statement in parse(candidate) if statement is not None]
    except ParseError:
        return False
    return len(statements) == 1


def safety_validity(sql: str) -> bool:
    try:
        validate_sql_output(_candidate_output(sql))
    except (TypeError, ValueError):
        return False
    return True


def compute_metrics(expected: Any, actual: Any) -> dict[str, bool | None]:
    candidate = _candidate_output(actual) if _is_sql_target(expected) else actual
    metrics: dict[str, bool | None] = {
        "non_empty": isinstance(actual, str) and bool(actual.strip()),
        "exact_match": _normalize_text(expected) == _normalize_text(candidate),
        "sql_syntax_valid": None,
        "sql_safe": None,
    }
    if _is_sql_target(expected):
        metrics["sql_syntax_valid"] = syntax_validity(actual)
        metrics["sql_safe"] = safety_validity(actual)
    return metrics
