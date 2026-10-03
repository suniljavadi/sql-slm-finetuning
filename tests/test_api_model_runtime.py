from fastapi.testclient import TestClient

import api.main as api_main
import src.inference.model as model_runtime


def test_model_generator_formats_prompt_and_uses_loaded_adapter(monkeypatch):
    loaded = []
    generated = []
    model = object()
    tokenizer = object()

    def fake_load_adapter_model(model_name, adapter_dir, use_4bit):
        loaded.append((model_name, adapter_dir, use_4bit))
        return model, tokenizer

    def fake_generate_completion(active_model, active_tokenizer, record, max_new_tokens):
        generated.append((active_model, active_tokenizer, record, max_new_tokens))
        return "SELECT name FROM customers;"

    monkeypatch.setattr(model_runtime, "load_adapter_model", fake_load_adapter_model)
    monkeypatch.setattr(model_runtime, "generate_completion", fake_generate_completion)

    generator = model_runtime.make_sql_generator("base-model", "adapter", True, 64)
    result = generator("List customer names")

    assert loaded == [("base-model", "adapter", True)]
    assert result == "SELECT name FROM customers;"
    assert generated == [
        (
            model,
            tokenizer,
            {
                "instruction": "Generate a read-only SQL query. Return only the SQL query.",
                "input": "List customer names",
                "schema": "",
            },
            64,
        )
    ]


def test_api_loads_configured_generator_once_at_startup(monkeypatch):
    loaded = []
    prompts = []

    def fake_make_sql_generator(**kwargs):
        loaded.append(kwargs)
        return lambda prompt: prompts.append(prompt) or "SELECT name FROM customers;"

    monkeypatch.setattr(api_main, "make_sql_generator", fake_make_sql_generator)
    monkeypatch.setenv("SQL_MODEL_ADAPTER_DIR", "artifacts/test-adapter")
    monkeypatch.setenv("SQL_MODEL_NAME", "test-base")
    monkeypatch.setenv("SQL_MODEL_USE_4BIT", "false")
    monkeypatch.setenv("SQL_MODEL_MAX_NEW_TOKENS", "64")

    with TestClient(api_main.app) as client:
        response = client.post("/generate", json={"prompt": "List customer names"})

    assert response.status_code == 200
    assert response.json()["sql"] == "SELECT name FROM customers;"
    assert prompts == ["List customer names"]
    assert loaded == [
        {
            "model_name": "test-base",
            "adapter_dir": "artifacts/test-adapter",
            "use_4bit": False,
            "max_new_tokens": 64,
        }
    ]