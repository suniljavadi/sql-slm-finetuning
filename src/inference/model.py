from __future__ import annotations

from pathlib import Path
from typing import Any

from src.training.data import format_prompt


def load_adapter_model(model_name: str, adapter_dir: str | Path, use_4bit: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit inference requires an available CUDA or ROCm GPU")
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


def generate_completion(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    max_new_tokens: int,
):
    import torch

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


def make_sql_generator(
    model_name: str,
    adapter_dir: str | Path,
    use_4bit: bool = True,
    max_new_tokens: int = 128,
):
    model, tokenizer = load_adapter_model(model_name, adapter_dir, use_4bit)

    def generate(prompt: str, schema: str = "") -> str:
        record = {
            "instruction": "Generate a read-only SQL query. Return only the SQL query.",
            "input": prompt,
            "schema": schema,
        }
        return generate_completion(model, tokenizer, record, max_new_tokens)

    return generate