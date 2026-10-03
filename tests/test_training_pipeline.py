import json
from pathlib import Path

import pytest
import torch

from src.training.data import CausalLMCollator, SQLInstructionDataset, format_prompt
from src.training.config import TrainingConfig
from src.training.train import (
    _apply_warmup_arguments,
    _training_arguments,
    build_training_summary,
    detect_training_backend,
    run_training,
)


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=1024):
        token_ids = list(range(1, len(text.split()) + 1))
        if add_special_tokens:
            token_ids.insert(0, 98)
        return {"input_ids": token_ids[:max_length]}


def test_sql_instruction_dataset_masks_prompt_and_appends_eos():
    record = {
        "instruction": "Generate SQL.",
        "input": "Count rows.",
        "schema": "items(id)",
        "output": "SELECT COUNT(*) FROM items",
    }
    example = SQLInstructionDataset([record], FakeTokenizer(), max_seq_length=64)[0]

    prompt_length = len(format_prompt(record).split()) + 1
    assert example["labels"][:prompt_length] == [-100] * prompt_length
    assert example["input_ids"][-1] == FakeTokenizer.eos_token_id
    assert example["labels"][-1] == FakeTokenizer.eos_token_id
    assert example["attention_mask"] == [1] * len(example["input_ids"])


def test_causal_lm_collator_pads_inputs_and_masks_padding_labels():
    tokenizer = FakeTokenizer()
    records = [
        {"instruction": "SQL", "input": "a", "schema": "t(x)", "output": "SELECT x"},
        {"instruction": "SQL", "input": "a longer request", "schema": "t(x)", "output": "SELECT x FROM t"},
    ]
    dataset = SQLInstructionDataset(records, tokenizer, max_seq_length=64)
    batch = CausalLMCollator(tokenizer.pad_token_id)([dataset[0], dataset[1]])

    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["labels"].shape
    padding_mask = batch["attention_mask"] == 0
    assert torch.all(batch["input_ids"][padding_mask] == tokenizer.pad_token_id)
    assert torch.all(batch["labels"][padding_mask] == -100)


def test_detect_training_backend_returns_known_backend():
    backend = detect_training_backend()
    assert backend in {"cpu", "cuda", "rocm"}


def test_build_training_summary_includes_backend_and_model():
    summary = build_training_summary(model_name="Qwen/Qwen2.5-3B-Instruct")
    assert "model_name" in summary
    assert "backend" in summary
    assert summary["model_name"] == "Qwen/Qwen2.5-3B-Instruct"


def test_detect_training_backend_prefers_rocm_when_cuda_api_is_available(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert detect_training_backend() == "rocm"


def test_rocm_training_summary_reports_device_count(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    summary = build_training_summary("test-model", backend="rocm")

    assert summary["device_count"] == 1


def _write_training_config(tmp_path: Path) -> Path:
    record = {
        "instruction": "Generate SQL.",
        "input": "Count rows.",
        "schema": "items(id)",
        "output": "SELECT COUNT(*) FROM items",
    }
    train_file = tmp_path / "train.jsonl"
    validation_file = tmp_path / "validation.jsonl"
    train_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    validation_record = {**record, "input": "List items."}
    validation_file.write_text(json.dumps(validation_record) + "\n", encoding="utf-8")
    config = TrainingConfig(
        train_file=str(train_file),
        validation_file=str(validation_file),
        output_dir=str(tmp_path / "adapter"),
    )
    import yaml

    config_file = tmp_path / "training.yaml"
    config_file.write_text(yaml.safe_dump(config.__dict__), encoding="utf-8")
    return config_file


def test_training_dry_run_validates_splits_without_creating_artifacts(tmp_path, monkeypatch):
    config_file = _write_training_config(tmp_path)
    monkeypatch.setenv("SKIP_TRAINING", "1")

    result = run_training(config_file)

    assert result["status"] == "dry_run"
    assert result["train_examples"] == 1
    assert result["validation_examples"] == 1
    assert not (tmp_path / "adapter").exists()


def test_qlora_training_fails_clearly_without_gpu(tmp_path, monkeypatch):
    config_file = _write_training_config(tmp_path)
    monkeypatch.delenv("SKIP_TRAINING", raising=False)
    monkeypatch.setattr("src.training.train.detect_training_backend", lambda: "cpu")

    with pytest.raises(RuntimeError, match="no CUDA/ROCm GPU"):
        run_training(config_file)

    assert not (tmp_path / "adapter").exists()


def test_training_arguments_translate_warmup_ratio_for_new_transformers(monkeypatch):
    config = TrainingConfig(precision="fp32", warmup_ratio=0.05)
    arguments = {}

    _apply_warmup_arguments(arguments, config, train_examples=19, supported_parameters={"warmup_steps"})

    assert arguments["warmup_steps"] == 1
    assert "warmup_ratio" not in arguments


def test_training_arguments_keep_warmup_ratio_for_older_transformers():
    config = TrainingConfig(precision="fp32", warmup_ratio=0.05)
    arguments = {}

    _apply_warmup_arguments(arguments, config, train_examples=19, supported_parameters={"warmup_ratio"})

    assert arguments == {"warmup_ratio": 0.05}
