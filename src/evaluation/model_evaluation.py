from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.data.validate_dataset import validate_dataset_file
from src.evaluation.evaluate import evaluate_dataset
from src.training.data import format_prompt


def _load_adapter_model(model_name: str, adapter_dir: str | Path, use_4bit: bool):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit evaluation requires an available CUDA or ROCm GPU")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def _generate(model: Any, tokenizer: Any, record: dict[str, Any], max_new_tokens: int):
    prompt = format_prompt(record)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    device = model.get_input_embeddings().weight.device
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


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