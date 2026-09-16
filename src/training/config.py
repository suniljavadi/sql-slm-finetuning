from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
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
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 100
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path):
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls(**payload)
