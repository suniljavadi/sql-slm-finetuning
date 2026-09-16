from src.training.train import detect_training_backend, build_training_summary


def test_detect_training_backend_returns_known_backend():
    backend = detect_training_backend()
    assert backend in {"cpu", "cuda", "rocm"}


def test_build_training_summary_includes_backend_and_model():
    summary = build_training_summary(model_name="Qwen/Qwen2.5-3B-Instruct")
    assert "model_name" in summary
    assert "backend" in summary
    assert summary["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
