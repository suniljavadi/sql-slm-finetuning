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
