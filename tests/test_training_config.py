from src.training.config import TrainingConfig


def test_training_config_loads_defaults():
    config = TrainingConfig()
    assert config.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert config.lora_r == 16
    assert config.seed == 42
