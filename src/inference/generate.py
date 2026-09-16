from __future__ import annotations

import re

SAFE_SQL_PATTERN = re.compile(r"\b(SELECT|WITH|SHOW|DESCRIBE|EXPLAIN)\b", re.IGNORECASE)


def validate_user_input(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input cannot be empty.")
    if len(cleaned) > 4000:
        raise ValueError("Input is too long.")
    return cleaned


def validate_sql_output(sql_text: str) -> str:
    if not isinstance(sql_text, str):
        raise TypeError("Generated SQL output must be a string.")
    cleaned = sql_text.strip()
    if not cleaned:
        raise ValueError("Generated SQL is empty.")
    if not SAFE_SQL_PATTERN.search(cleaned):
        raise ValueError("Generated SQL is unsafe or unsupported for this assistant.")
    if any(token in cleaned.upper() for token in ["DROP ", "ALTER ", "DELETE ", "UPDATE ", "INSERT "]):
        raise ValueError("Unsafe SQL action detected.")
    return cleaned


def generate_sql(prompt: str, model_response: str | None = None) -> str:
    user_input = validate_user_input(prompt)
    if model_response is None:
        model_response = f"SELECT * FROM example_table WHERE description LIKE '%{user_input}%';"
    return validate_sql_output(model_response)
