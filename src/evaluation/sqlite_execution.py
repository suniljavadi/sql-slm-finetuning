from __future__ import annotations

import sqlite3
import time
import re
from collections import Counter
from pathlib import Path
from typing import Any

from sqlglot import exp, parse, parse_one
from sqlglot.errors import SqlglotError

from src.inference.generate import trim_model_continuation, validate_sql_output

SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _read_only_query(sql: str) -> tuple[str, bool]:
    candidate = trim_model_continuation(sql)
    fenced_sql = SQL_FENCE.search(candidate)
    if fenced_sql:
        candidate = fenced_sql.group(1).strip()
    candidate = validate_sql_output(candidate)
    try:
        statements = [statement for statement in parse(candidate, read="sqlite") if statement is not None]
        parsed = parse_one(candidate, read="sqlite")
    except SqlglotError as exc:
        raise ValueError("SQL could not be parsed as SQLite") from exc
    if len(statements) != 1 or not isinstance(parsed, exp.Query):
        raise ValueError("Only one read-only SQL query is allowed")
    return candidate, bool(parsed.args.get("order"))


def _canonical_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, bytes):
        return value.hex()
    return value


def execute_read_only_query(
    database_path: str | Path,
    sql: str,
    timeout_seconds: float = 3.0,
    max_rows: int = 10000,
) -> list[tuple[Any, ...]]:
    if timeout_seconds <= 0 or max_rows < 1:
        raise ValueError("Timeout and row limit must be positive")
    query, _ = _read_only_query(sql)
    uri = f"{Path(database_path).resolve().as_uri()}?mode=ro"
    deadline = time.monotonic() + timeout_seconds
    connection = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        cursor = connection.execute(query)
        if cursor.description is None:
            raise ValueError("SQL did not return a result set")
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError(f"Query returned more than the {max_rows}-row limit")
        return [tuple(_canonical_value(value) for value in row) for row in rows]
    finally:
        connection.close()


def execution_matches(
    database_path: str | Path,
    expected_sql: str,
    actual_sql: str,
    timeout_seconds: float = 3.0,
    max_rows: int = 10000,
) -> bool:
    expected_query, expected_ordered = _read_only_query(expected_sql)
    actual_query, _ = _read_only_query(actual_sql)
    expected_rows = execute_read_only_query(database_path, expected_query, timeout_seconds, max_rows)
    actual_rows = execute_read_only_query(database_path, actual_query, timeout_seconds, max_rows)
    if expected_ordered:
        return expected_rows == actual_rows
    return Counter(expected_rows) == Counter(actual_rows)