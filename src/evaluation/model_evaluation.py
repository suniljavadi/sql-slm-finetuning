from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.validate_dataset import validate_dataset_file
from src.evaluation.evaluate import evaluate_dataset
from src.inference.model import generate_completion as _generate
from src.inference.model import load_adapter_model as _load_adapter_model


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    sql_results = [result["metrics"] for result in results if result["metrics"]["sql_syntax_valid"] is not None]
    return {
        "examples": len(results),
        "exact_match_rate": sum(result["metrics"]["exact_match"] for result in results) / len(results),
        "sql_examples": len(sql_results),
        "sql_syntax_valid_rate": (
            sum(metric["sql_syntax_valid"] for metric in sql_results) / len(sql_results)
            if sql_results else None
        ),
        "sql_safety_pass_rate": (
            sum(metric["sql_safe"] for metric in sql_results) / len(sql_results)
            if sql_results else None
        ),
    }


def evaluate_base_and_adapter(
    test_file: str | Path,
    model_name: str,
    adapter_dir: str | Path,
    max_new_tokens: int = 128,
    use_4bit: bool = True,
) -> dict[str, Any]:
    records = validate_dataset_file(test_file)
    model, tokenizer = _load_adapter_model(model_name, adapter_dir, use_4bit)

    base_results = evaluate_dataset(
        test_file,
        lambda record: _generate_with_adapter_state(
            model, tokenizer, record, max_new_tokens, disable_adapter=True
        ),
    )
    adapter_results = evaluate_dataset(
        test_file,
        lambda record: _generate_with_adapter_state(
            model, tokenizer, record, max_new_tokens, disable_adapter=False
        ),
    )
    return {
        "model": model_name,
        "adapter_dir": str(adapter_dir),
        "test_examples": len(records),
        "base": _summary(base_results),
        "adapter": _summary(adapter_results),
        "examples": [
            {"expected": record["output"], "base": base["actual"], "adapter": tuned["actual"]}
            for record, base, tuned in zip(records, base_results, adapter_results, strict=True)
        ],
    }


def _generate_with_adapter_state(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    max_new_tokens: int,
    disable_adapter: bool,
) -> str:
    if disable_adapter:
        with model.disable_adapter():
            return _generate(model, tokenizer, record, max_new_tokens)
    return _generate(model, tokenizer, record, max_new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved SQL LoRA adapter against its base model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-dir", default="./artifacts/checkpoints")
    parser.add_argument("--test-file", default="./data/test.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-4bit", action="store_true", help="Load BF16/base precision instead of QLoRA weights")
    arguments = parser.parse_args()
    result = evaluate_base_and_adapter(
        arguments.test_file,
        arguments.model,
        arguments.adapter_dir,
        arguments.max_new_tokens,
        use_4bit=not arguments.no_4bit,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()