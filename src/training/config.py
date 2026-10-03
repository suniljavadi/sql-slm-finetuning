from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    train_file: str = "./data/train.jsonl"
    validation_file: str = "./data/val.jsonl"
    output_dir: str = "./artifacts/checkpoints"
    max_seq_length: int = 1024
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    epochs: int = 1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    precision: str = "bf16"
    use_4bit: bool = True
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_seq_length < 3:
            raise ValueError("max_seq_length must be at least 3")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("Batch size and gradient accumulation must be positive")
        if self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("Epochs and learning rate must be positive")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if self.precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be one of: bf16, fp16, fp32")
        if self.lora_r < 1 or self.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("At least one LoRA target module is required")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls(**payload)
