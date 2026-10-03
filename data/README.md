# Data

This directory holds the synthetic SQL/data-engineering examples used for baseline, fine-tuning, and evaluation.

## Files

- raw/: raw generated or downloaded source data
- processed/: cleaned and formatted records used in training
- evaluation/: holdout evaluation datasets

## Data Format

Each record is a JSON object with:

- instruction
- input
- schema
- output

Example:

```json
{
  "instruction": "Generate a SQL query to find the top 5 customers by revenue.",
  "input": "Show the top 5 customers by total revenue.",
  "schema": "customers(id, name, revenue)",
  "output": "SELECT name, revenue FROM customers ORDER BY revenue DESC LIMIT 5;"
}
```

## Validation Rules

- fields cannot be empty
- duplicate examples are rejected
- invalid SQL is flagged
- schema must be non-empty
- train/test overlap is prevented
- output too long or malformed is rejected

## Status

This folder contains reproducible example data for software validation. It is not production enterprise data.

The checked-in JSONL splits currently contain 19 training, 4 validation, and 5 test examples (28 total). The small size is suitable for validating the data and training pipeline only, not for measuring production model quality.

## Optional Schema-Conditioned Fine-Tuning Data

`src/data/prepare_sql_create_context.py` downloads a seeded sample from [SQL-Create-Context](https://huggingface.co/datasets/b-mc2/sql-create-context), filters invalid, duplicate, and non-query SQL examples, then writes deterministic train/validation/test splits under ignored `artifacts/`. The source dataset is labeled CC-BY-4.0 and was built from [WikiSQL](https://huggingface.co/datasets/wikisql) and [Spider](https://huggingface.co/datasets/spider); retain attribution to b-mc2 and cite the upstream datasets when using derived data. The dataset itself is not checked into this repository.

Prepare 1,000 examples and train one QLoRA epoch with:

```bash
python -m src.data.prepare_sql_create_context --max-examples 1000
python -m src.training.train --config configs/training.sql-create-context.yaml
python -m src.evaluation.model_evaluation --test-file artifacts/sql-create-context-1000/test.jsonl --adapter-dir artifacts/sql-create-context-qlora/checkpoints
```

This larger run is an experiment, not a quality claim. Compare exact match, SQL syntax/safety, and execution-based results before describing it as an improvement.
