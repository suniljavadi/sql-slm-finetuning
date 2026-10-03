from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from src.evaluation.metrics import compute_metrics
from src.evaluation.model_evaluation import _generate_with_adapter_state, _load_adapter_model, _summary
from src.evaluation.sqlite_execution import execution_matches
from src.inference.generate import trim_model_continuation


def select_bird_examples(
    rows: list[dict[str, Any]],
    max_examples: int = 30,
    seed: int = 42,
    database_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    allowed = set(database_ids) if database_ids else None
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        database_id = row.get("db_id")
        if isinstance(database_id, str) and (allowed is None or database_id in allowed):
            groups[database_id].append(row)

    generator = random.Random(seed)
    for group in groups.values():
        generator.shuffle(group)

    selected: list[dict[str, Any]] = []
    database_order = sorted(groups)
    while len(selected) < max_examples:
        added = False
        for database_id in database_order:
            if groups[database_id]:
                selected.append(groups[database_id].pop())
                added = True
                if len(selected) == max_examples:
                    break
        if not added:
            break
    if not selected:
        raise ValueError("No BIRD examples matched the requested database IDs")
    return selected


def read_sqlite_schema(database_path: str | Path) -> str:
    uri = f"{Path(database_path).resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        definitions: list[str] = []
        for (table_name,) in tables:
            quoted_table = table_name.replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{quoted_table}")').fetchall()
            column_definitions = []
            for column in columns:
                column_name = column[1].replace('"', '""')
                column_type = column[2] or ""
                column_definitions.append(f'"{column_name}" {column_type}'.rstrip())
            definitions.append(f'CREATE TABLE "{quoted_table}" ({", ".join(column_definitions)})')
        if not definitions:
            raise ValueError(f"No user tables found in SQLite database: {database_path}")
        return "; ".join(definitions)
    finally:
        connection.close()


def evaluate_bird_examples(
    rows: list[dict[str, Any]],
    database_dir: str | Path,
    generator: Callable[[dict[str, Any]], str],
    max_examples: int = 30,
    seed: int = 42,
    database_ids: list[str] | None = None,
    timeout_seconds: float = 3.0,
    max_rows: int = 10000,
) -> dict[str, Any]:
    selected = select_bird_examples(rows, max_examples, seed, database_ids)
    results: list[dict[str, Any]] = []
    root = Path(database_dir)
    for row in selected:
        database_id = row["db_id"]
        database_path = root / database_id / f"{database_id}.sqlite"
        evidence = row.get("evidence") or ""
        prompt = row["question"].strip()
        if evidence:
            prompt = f"{prompt}\nEvidence: {evidence}"
        record = {
            "instruction": "Generate a read-only SQL query. Return only the SQL query.",
            "input": prompt,
            "schema": read_sqlite_schema(database_path),
            "output": row["SQL"],
        }
        actual = generator(record)
        metrics = compute_metrics(record["output"], actual)
        try:
            metrics["execution_correct"] = execution_matches(
                database_path,
                record["output"],
                trim_model_continuation(actual),
                timeout_seconds=timeout_seconds,
                max_rows=max_rows,
            )
        except (sqlite3.Error, TypeError, ValueError):
            metrics["execution_correct"] = False
        results.append({
            "db_id": database_id,
            "question_id": row.get("question_id"),
            "expected": record["output"],
            "actual": actual,
            "metrics": metrics,
        })

    summary = _summary([{"metrics": result["metrics"]} for result in results])
    summary["execution_accuracy"] = sum(
        result["metrics"]["execution_correct"] for result in results
    ) / len(results)
    return {"examples": len(results), "summary": summary, "results": results}


def evaluate_bird_base_and_adapter(
    dataset_file: str | Path,
    database_dir: str | Path,
    model_name: str,
    adapter_dir: str | Path,
    max_examples: int = 30,
    seed: int = 42,
    database_ids: list[str] | None = None,
    max_new_tokens: int = 128,
    use_4bit: bool = True,
) -> dict[str, Any]:
    with Path(dataset_file).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    records = select_bird_examples(rows, max_examples, seed, database_ids)
    model, tokenizer = _load_adapter_model(model_name, adapter_dir, use_4bit)

    def generate(record: dict[str, Any], disable_adapter: bool) -> str:
        return _generate_with_adapter_state(
            model, tokenizer, record, max_new_tokens, disable_adapter
        )

    base = evaluate_bird_examples(
        records,
        database_dir,
        lambda record: generate(record, True),
        max_examples=len(records),
        seed=seed,
        timeout_seconds=3.0,
    )
    adapter = evaluate_bird_examples(
        records,
        database_dir,
        lambda record: generate(record, False),
        max_examples=len(records),
        seed=seed,
        timeout_seconds=3.0,
    )
    return {"base": base, "adapter": adapter}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BIRD Mini-Dev with SQLite execution accuracy")
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--database-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--database-ids", nargs="+")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-4bit", action="store_true")
    arguments = parser.parse_args()
    result = evaluate_bird_base_and_adapter(
        arguments.dataset_file,
        arguments.database_dir,
        arguments.model,
        arguments.adapter_dir,
        max_examples=arguments.max_examples,
        seed=arguments.seed,
        database_ids=arguments.database_ids,
        max_new_tokens=arguments.max_new_tokens,
        use_4bit=not arguments.no_4bit,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()