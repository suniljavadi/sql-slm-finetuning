from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from sqlglot import exp, parse
from sqlglot.errors import ParseError


INSTRUCTION = "Generate a read-only SQL query using the request and database schema. Return only SQL."


def _convert_row(row: dict[str, Any]) -> dict[str, str] | None:
    question = row.get("question")
    schema = row.get("context")
    answer = row.get("answer")
    if not all(isinstance(value, str) and value.strip() for value in (question, schema, answer)):
        return None
    if len(question) > 2000 or len(schema) > 4000 or len(answer) > 8000:
        return None

    try:
        statements = [statement for statement in parse(answer) if statement is not None]
    except ParseError:
        return None
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return None

    return {
        "instruction": INSTRUCTION,
        "input": question.strip(),
        "schema": schema.strip(),
        "output": answer.strip(),
    }


def prepare_splits(
    source_rows: Iterable[dict[str, Any]],
    max_examples: int = 1000,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    if max_examples < 4:
        raise ValueError("max_examples must be at least 4 for non-empty train, validation, and test splits")

    candidates = list(source_rows)
    random.Random(seed).shuffle(candidates)
    examples: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in candidates:
        example = _convert_row(row)
        if example is None:
            continue
        signature = (
            example["input"].casefold(),
            example["schema"].casefold(),
            example["output"].casefold(),
        )
        if signature in seen:
            continue
        seen.add(signature)
        examples.append(example)
        if len(examples) == max_examples:
            break

    if len(examples) < 4:
        raise ValueError(f"Only {len(examples)} valid unique read-only examples were found")

    train_end = min(len(examples) - 2, max(1, int(len(examples) * 0.8)))
    validation_end = min(len(examples) - 1, train_end + max(1, int(len(examples) * 0.1)))
    return {
        "train": examples[:train_end],
        "validation": examples[train_end:validation_end],
        "test": examples[validation_end:],
    }


def write_splits(splits: dict[str, list[dict[str, str]]], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        filename = "val.jsonl" if split_name == "validation" else f"{split_name}.jsonl"
        with (output_path / filename).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a filtered SQL-Create-Context sample")
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./artifacts/sql-create-context-1000")
    arguments = parser.parse_args()

    from datasets import load_dataset

    source_rows = load_dataset("b-mc2/sql-create-context", split="train")
    splits = prepare_splits(source_rows, arguments.max_examples, arguments.seed)
    write_splits(splits, arguments.output_dir)
    print(json.dumps({name: len(rows) for name, rows in splits.items()}, indent=2))
    print(f"Wrote splits to {Path(arguments.output_dir)}")


if __name__ == "__main__":
    main()