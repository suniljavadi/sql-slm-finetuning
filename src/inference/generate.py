from __future__ import annotations

import re
from collections.abc import Callable

SAFE_SQL_PATTERN = re.compile(r"\b(SELECT|WITH|SHOW|DESCRIBE|EXPLAIN)\b", re.IGNORECASE)
CONTINUATION_MARKER = re.compile(r"(?:Human:\s*)?###\s*Instruction:", re.IGNORECASE)


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


def trim_model_continuation(sql_text: str) -> str:
    match = CONTINUATION_MARKER.search(sql_text)
    return sql_text[:match.start()].strip() if match else sql_text.strip()


def generate_sql(
    prompt: str,
    model_response: str | None = None,
    generator: Callable[[str, str], str] | None = None,
    schema: str = "",
) -> str:
    user_input = validate_user_input(prompt)
    if not isinstance(schema, str):
        raise TypeError("Schema must be a string.")
    cleaned_schema = schema.strip()
    if len(cleaned_schema) > 4000:
        raise ValueError("Schema is too long.")
    if model_response is None:
        if generator is None:
            model_response = f"SELECT * FROM example_table WHERE description LIKE '%{user_input}%';"
        else:
            model_response = trim_model_continuation(generator(user_input, cleaned_schema))
    return validate_sql_output(model_response)
