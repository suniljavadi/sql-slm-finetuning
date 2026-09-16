import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {"instruction", "input", "schema", "output"}
DANGEROUS_SQL = ("DROP", "ALTER", "DELETE", "UPDATE", "INSERT")


def is_valid_sql_text(sql: str) -> bool:
    if not isinstance(sql, str) or not sql.strip():
        return False
    return True


def validate_record(record: dict[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(record, dict):
        return False, "Record is not a dictionary."

    missing = sorted(REQUIRED_FIELDS - set(record.keys()))
    if missing:
        return False, f"Missing required fields: {missing}"

    for field in REQUIRED_FIELDS:
        value = record[field]
        if value is None or (isinstance(value, str) and not value.strip()):
            return False, f"Field '{field}' is empty."

    if len(record["output"]) > 20000:
        return False, "Output is too long."

    sql_text = record["output"].upper()
    if any(keyword in sql_text for keyword in DANGEROUS_SQL):
        return True, None

    return True, None


def validate_dataset_file(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    rows: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc

            ok, err = validate_record(row)
            if not ok:
                raise ValueError(f"Invalid record at line {line_number}: {err}")

            rows.append(row)

    if not rows:
        raise ValueError("Dataset is empty.")

    seen = set()
    for idx, row in enumerate(rows):
        signature = (row.get("instruction"), row.get("input"), row.get("schema"), row.get("output"))
        if signature in seen:
            raise ValueError(f"Duplicate record detected at index {idx}.")
        seen.add(signature)

    return rows
