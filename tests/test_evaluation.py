import json
from pathlib import Path

from src.evaluation.evaluate import evaluate_dataset
from src.evaluation.metrics import compute_metrics
from src.evaluation.model_evaluation import _summary


def test_sql_metrics_distinguish_syntax_safety_and_exact_match():
    metrics = compute_metrics("SELECT id FROM customers;", "DELETE FROM customers;")

    assert metrics["exact_match"] is False
    assert metrics["sql_syntax_valid"] is True
    assert metrics["sql_safe"] is False


def test_sql_metrics_use_first_fenced_query_before_extra_continuation():
    actual = "```sql\nSELECT id FROM customers;\n```\n### Instruction:\nnext example"

    metrics = compute_metrics("SELECT id FROM customers;", actual)

    assert metrics["exact_match"] is True
    assert metrics["sql_syntax_valid"] is True
    assert metrics["sql_safe"] is True


def test_explanation_output_does_not_claim_sql_syntax_metrics():
    explanation = "This query counts all customer rows."

    metrics = compute_metrics(explanation, explanation)

    assert metrics["exact_match"] is True
    assert metrics["sql_syntax_valid"] is None
    assert metrics["sql_safe"] is None


def test_evaluator_scores_generator_output_instead_of_expected_output(tmp_path: Path):
    dataset_path = tmp_path / "test.jsonl"
    row = {
        "instruction": "Generate SQL.",
        "input": "List customer names.",
        "schema": "customers(id, name)",
        "output": "SELECT name FROM customers;",
    }
    dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    results = evaluate_dataset(dataset_path, lambda _: "SELECT id FROM customers;")

    assert results[0]["expected"] == row["output"]
    assert results[0]["actual"] == "SELECT id FROM customers;"
    assert results[0]["metrics"]["exact_match"] is False


def test_model_evaluation_summary_aggregates_metric_rows():
    results = [
        {"metrics": {"exact_match": True, "sql_syntax_valid": True, "sql_safe": True}},
        {"metrics": {"exact_match": False, "sql_syntax_valid": False, "sql_safe": False}},
    ]

    summary = _summary(results)

    assert summary["exact_match_rate"] == 0.5
    assert summary["sql_syntax_valid_rate"] == 0.5
    assert summary["sql_safety_pass_rate"] == 0.5