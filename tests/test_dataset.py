import json
from pathlib import Path

from src.data.prepare_sql_create_context import prepare_splits
from src.data.validate_dataset import validate_dataset_file


def test_valid_dataset_file(tmp_path: Path):
    rows = [
        {
            "instruction": "Return all customers from Hyderabad.",
            "input": "Show customers in Hyderabad.",
            "schema": "customers(id, name, city)",
            "output": "SELECT * FROM customers WHERE city = 'Hyderabad';",
        },
        {
            "instruction": "Count rows by city.",
            "input": "How many customers per city?",
            "schema": "customers(id, city)",
            "output": "SELECT city, COUNT(*) FROM customers GROUP BY city;",
        },
    ]
    file_path = tmp_path / "dataset.jsonl"
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    parsed = validate_dataset_file(file_path)
    assert len(parsed) == 2


def test_missing_fields_rejected(tmp_path: Path):
    file_path = tmp_path / "bad.jsonl"
    file_path.write_text('{"instruction": "hi", "input": "x"}\n', encoding="utf-8")

    try:
        validate_dataset_file(file_path)
        raise AssertionError("Expected validation to fail")
    except ValueError:
        pass


def test_duplicate_rows_rejected(tmp_path: Path):
    file_path = tmp_path / "dup.jsonl"
    row = {
        "instruction": "Return all customers.",
        "input": "List all customers.",
        "schema": "customers(id, name)",
        "output": "SELECT * FROM customers;",
    }
    file_path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    try:
        validate_dataset_file(file_path)
        raise AssertionError("Expected duplicate validation to fail")
    except ValueError:
        pass


def test_prepare_splits_filters_mutations_deduplicates_and_is_deterministic():
    source_rows = [
        {
            "question": f"Show customer {index}",
            "context": "CREATE TABLE customers (id INT, name TEXT)",
            "answer": f"SELECT name FROM customers WHERE id = {index}",
        }
        for index in range(10)
    ]
    source_rows.extend(
        [
            {
                "question": "Remove all customers",
                "context": "CREATE TABLE customers (id INT)",
                "answer": "DELETE FROM customers",
            },
            source_rows[0],
        ]
    )

    first = prepare_splits(source_rows, max_examples=10, seed=19)
    second = prepare_splits(source_rows, max_examples=10, seed=19)

    assert first == second
    assert sum(map(len, first.values())) == 10
    assert all("DELETE" not in row["output"].upper() for rows in first.values() for row in rows)
    assert all(first[name] for name in ("train", "validation", "test"))
