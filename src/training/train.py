import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from src.data.validate_dataset import validate_dataset_file
from src.training.config import TrainingConfig
from src.training.data import CausalLMCollator, SQLInstructionDataset
from src.utils.logging import get_logger
from src.utils.seed import set_seed


logger = get_logger("training")


def _apply_warmup_arguments(
    arguments: dict[str, Any],
    config: TrainingConfig,
    train_examples: int,
    supported_parameters,
) -> None:
    if "warmup_ratio" in supported_parameters:
        arguments["warmup_ratio"] = config.warmup_ratio
        return
    batches_per_epoch = math.ceil(train_examples / config.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    total_training_steps = steps_per_epoch * config.epochs
    arguments["warmup_steps"] = math.ceil(total_training_steps * config.warmup_ratio)


def detect_training_backend() -> str:
    if getattr(torch.version, "hip", None) is not None:
        return "rocm"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_training_summary(model_name: str, backend: str | None = None) -> dict[str, Any]:
    backend_name = backend or detect_training_backend()
    return {
        "model_name": model_name,
        "backend": backend_name,
        "status": "configured",
        "device_count": (
            torch.cuda.device_count()
            if backend_name in {"cuda", "rocm"} and torch.cuda.is_available()
            else 0
        ),
    }


def _training_arguments(config: TrainingConfig, backend: str, train_examples: int):
    from transformers import TrainingArguments

    supported = inspect.signature(TrainingArguments).parameters
    strategy_name = "eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"
    use_accelerator = backend in {"cuda", "rocm"}
    if config.precision == "bf16" and use_accelerator and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The accelerator does not report bfloat16 support; set precision to fp16")

    arguments = {
        "output_dir": config.output_dir,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "save_strategy": config.save_strategy,
        "report_to": [],
        "remove_unused_columns": False,
        "gradient_checkpointing": config.gradient_checkpointing,
        "dataloader_pin_memory": use_accelerator,
        "bf16": use_accelerator and config.precision == "bf16",
        "fp16": use_accelerator and config.precision == "fp16",
    }
    arguments[strategy_name] = config.eval_strategy
    _apply_warmup_arguments(arguments, config, train_examples, supported)
    return TrainingArguments(**arguments)


def _load_lora_model(config: TrainingConfig, backend: str, tokenizer):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if config.use_4bit and backend not in {"cuda", "rocm"}:
        raise RuntimeError("4-bit LoRA training requires an available CUDA or ROCm GPU")

    if config.precision == "bf16" and backend in {"cuda", "rocm"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The accelerator does not report bfloat16 support; set precision to fp16")
        compute_dtype = torch.bfloat16
    elif config.precision == "fp16" and backend in {"cuda", "rocm"}:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    model_kwargs: dict[str, Any] = {"torch_dtype": compute_dtype}
    if config.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    if config.use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    adapter_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, adapter_config)


def run_training(config_path: str | Path = "./configs/training.yaml") -> dict[str, Any]:
    config = TrainingConfig.from_yaml(config_path)
    set_seed(config.seed)
    backend = detect_training_backend()
    summary = build_training_summary(config.model_name, backend)
    train_records = validate_dataset_file(config.train_file)
    validation_records = validate_dataset_file(config.validation_file)

    logger.info("Using model: %s", config.model_name)
    logger.info("Detected backend: %s", backend)
    logger.info("Validated train=%s validation=%s records", len(train_records), len(validation_records))
    logger.info(
        "LoRA configuration: rank=%s alpha=%s dropout=%s target=%s",
        config.lora_r,
        config.lora_alpha,
        config.lora_dropout,
        config.target_modules,
    )

    if os.getenv("SKIP_TRAINING", "0") == "1":
        logger.info("Dry run complete; no model was downloaded and no artifacts were written")
        return {
            **summary,
            "status": "dry_run",
            "train_examples": len(train_records),
            "validation_examples": len(validation_records),
            "config": config.__dict__,
        }

    if config.use_4bit and backend not in {"cuda", "rocm"}:
        raise RuntimeError("This configuration requests 4-bit training, but no CUDA/ROCm GPU is available")

    from transformers import AutoTokenizer, Trainer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("The selected tokenizer must provide an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = SQLInstructionDataset(train_records, tokenizer, config.max_seq_length)
    validation_dataset = SQLInstructionDataset(validation_records, tokenizer, config.max_seq_length)
    model = _load_lora_model(config, backend, tokenizer)
    model.print_trainable_parameters()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": _training_arguments(config, backend, len(train_dataset)),
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": CausalLMCollator(tokenizer.pad_token_id),
    }
    trainer_parameters = inspect.signature(Trainer).parameters
    tokenizer_parameter = "processing_class" if "processing_class" in trainer_parameters else "tokenizer"
    trainer_kwargs[tokenizer_parameter] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()
    evaluation_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return {
        **summary,
        "status": "trained",
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "output_dir": str(output_dir),
        "train_metrics": train_result.metrics,
        "evaluation_metrics": evaluation_metrics,
        "config": config.__dict__,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune the SQL SLM with LoRA/QLoRA")
    parser.add_argument("--config", default="./configs/training.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and data without downloading a model",
    )
    arguments = parser.parse_args()
    if arguments.dry_run:
        os.environ["SKIP_TRAINING"] = "1"
    print(json.dumps(run_training(arguments.config), indent=2, default=str))
