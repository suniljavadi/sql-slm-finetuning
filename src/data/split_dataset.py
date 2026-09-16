import json
from pathlib import Path


def split_dataset(rows: list[dict], train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, output_dir: str | Path = "./data") -> dict[str, list[dict]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Train/validation/test ratios must sum to 1.0.")

    total = len(rows)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, subset in {
        "train.jsonl": train_rows,
        "val.jsonl": val_rows,
        "test.jsonl": test_rows,
    }.items():
        file_path = output_path / name
        with file_path.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"train": train_rows, "validation": val_rows, "test": test_rows}
