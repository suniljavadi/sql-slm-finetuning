import os
from pathlib import Path

import torch

from src.training.config import TrainingConfig
from src.utils.logging import get_logger
from src.utils.seed import set_seed


logger = get_logger("training")


def detect_training_backend() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return "rocm"
    return "cpu"


def build_training_summary(model_name: str, backend: str | None = None) -> dict:
    backend_name = backend or detect_training_backend()
    return {
        "model_name": model_name,
        "backend": backend_name,
        "status": "configured",
        "device_count": torch.cuda.device_count() if backend_name == "cuda" else 0,
    }


def run_training(config_path: str | Path = "./configs/training.yaml") -> dict:
    config = TrainingConfig.from_yaml(config_path)
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = detect_training_backend()
    logger.info("Using model: %s", config.model_name)
    logger.info("Output dir: %s", output_dir)
    logger.info("Detected backend: %s", backend)
    logger.info("LoRA configuration: rank=%s alpha=%s dropout=%s target=%s", config.lora_r, config.lora_alpha, config.lora_dropout, config.target_modules)

    summary = build_training_summary(config.model_name, backend)

    if os.getenv("SKIP_TRAINING", "0") == "1":
        logger.warning("Training skipped due to SKIP_TRAINING=1")
        return {"status": "skipped", "config": config.__dict__, **summary}

    return {"status": "configured", "config": config.__dict__, **summary}
