import json
from pathlib import Path

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
