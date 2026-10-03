# Project Evidence

## Project Scope

- Model: Qwen2.5-3B-Instruct
- Method: QLoRA NF4 with LoRA rank 16
- Data: 28 synthetic baseline examples plus a filtered 1,000-row sample from SQL-Create-Context; no proprietary data
- Public dataset provenance: SQL-Create-Context is labeled CC-BY-4.0 and derived from WikiSQL and Spider; attribution is documented in `data/README.md`
- Purpose: verify the pipeline and run a preliminary quality experiment, not claim production readiness

## Dataset

- Total examples: 28
- Train: 19
- Validation: 4
- Test: 5
- Split: 70% / 15% / 15%, verified by dataset tests

The checked-in synthetic dataset is intentionally tiny and used only for the initial smoke run.

The separate schema-conditioned dataset experiment sampled 1,000 valid, unique read-only queries from SQL-Create-Context using seed 42. It was shuffled and split 800/100/100 before fine-tuning; exact duplicates were removed across the whole sample. The source dataset combines WikiSQL and Spider-derived examples. The held-out split is random within this derived dataset and is not an independent benchmark.

## AMD Runtime

- Provider: AMD Developer Cloud GPU Droplet
- Accelerator: 1x AMD Instinct MI300X VF, 192 GB VRAM
- Region: ATL1
- Container: PyTorch marketplace image
- Runtime observed in container: Python 3.12.3, PyTorch 2.12.0+rocm7.14.0, HIP 7.14.60850, bitsandbytes 0.50.2
- `torch.cuda.is_available()`: true; device count: 1
- bitsandbytes NF4 linear-layer GPU smoke test: passed

## Actual Training Run

Status: COMPLETED as a short pipeline smoke run.

- Backend: ROCm
- Epochs: 1
- Optimizer steps: 2
- Batch size: 2
- Gradient accumulation: 8
- Train examples: 19
- Validation examples: 4
- Train loss: 1.4969
- Validation loss: 1.7522
- Train runtime: 9.63 seconds
- Trainable adapter parameters: 7,372,800 (0.2383% of 3,093,311,488 total parameters)
- Adapter saved on Droplet at `/shared-docker/sql-slm-finetuning/artifacts/checkpoints`
- Adapter and tokenizer downloaded locally to ignored `artifacts/amd-mi300x-qlora/`

This run proves the ROCm QLoRA pipeline executed; it is not a meaningful convergence or model-quality claim.

## Schema-Conditioned Dataset Experiment

Status: COMPLETED as a preliminary quality experiment.

- Source: `b-mc2/sql-create-context`, labeled CC-BY-4.0
- Filtered sample: 1,000 rows; train 800, validation 100, test 100
- Filtering: requires question/schema/answer, SQLGlot-parsable single read-only query, bounded field lengths, and unique question/schema/answer
- Split seed: 42
- Backend: ROCm on the same MI300X VF
- Epochs: 1; optimizer steps: 50
- Batch size: 2; gradient accumulation: 8
- Train loss: 0.3254
- Validation loss: 0.0874
- Runtime: 123.1 seconds
- Adapter saved on Droplet at `/shared-docker/sql-slm-finetuning-api-demo/artifacts/sql-create-context-qlora/checkpoints`
- Adapter copied locally to ignored `artifacts/sql-create-context-qlora/checkpoints/`
- Evaluation report saved on Droplet and copied locally to ignored `artifacts/sql-create-context-qlora/evaluation.json`

These loss values document the run; held-out generation metrics below are the more relevant quality check.

### Paired 100-Row Evaluation

The base and adapter were evaluated on the same 100-row random held-out split. Scoring trims model continuation at the next training prompt and scores a fenced SQL block when present.

| Measure | Base | Adapter |
| --- | ---: | ---: |
| Normalized exact match | 0/100 | 61/100 |
| SQL parse-valid | 27/100 | 100/100 |
| Project SQL safety pass | 100/100 | 100/100 |

This is a promising within-dataset result, not an independent benchmark. Random row splitting may leave related schemas or query patterns across splits. Exact match does not establish semantic correctness, and no queries were executed against databases. The safety metric checks the project’s read-only validator; it is not a database authorization boundary.

### Transfer Check on Original Synthetic Holdout

After the larger fine-tune, the new adapter was evaluated again on the original five-row synthetic test split:

| Measure | Base | New adapter |
| --- | ---: | ---: |
| Exact match | 0/5 | 0/5 |
| SQL parse-valid | 4/4 | 4/4 |
| Project SQL safety pass | 4/4 | 4/4 |

This small check found no exact-match transfer to the original synthetic examples. Together with the 100-row result, it suggests the model may be learning dataset-specific patterns; independent and execution-based evaluation is still needed.

## Held-Out Evaluation

The base model and fine-tuned adapter were both generated against the same five test records.

| Measure | Base | Adapter |
| --- | ---: | ---: |
| Exact match | 0/5 | 0/5 |
| SQL targets with parse-valid first fenced query | 4/4 | 4/4 |
| SQL targets passing the project safety filter | 4/4 | 4/4 |

The adapter showed no exact-match gain in this smoke experiment. The test set is too small for generalization conclusions. The generator also emitted prompt-like continuation after the SQL block; evaluation scores only the first fenced SQL candidate. Semantic correctness and database execution were not measured.

## Software Verification

- Dataset validation and split tests: passed
- Public dataset filtering, deduplication, deterministic split, and minimum non-empty split tests: passed
- Training-data prompt masking and padding tests: passed
- ROCm backend precedence and device-count tests: passed
- Dry-run: passed for both the baseline (19 train / 4 validation) and public-data (800 train / 100 validation) configs without downloading weights or writing model artifacts
- CPU + 4-bit guard: passed; refuses to claim GPU training on CPU
- Base-vs-adapter evaluation metrics: tested with mocked predictions and exercised on 100 AMD-held-out examples

## Failure and Fix

The first AMD attempt loaded the base model and LoRA weights but stopped before training because Transformers 5.18 removed the `warmup_ratio` argument. The trainer now maps the configured ratio to `warmup_steps` for newer Transformers while retaining `warmup_ratio` for versions that support it. The subsequent AMD run completed.

## Not Verified

- Generalization beyond the SQL-Create-Context-derived random split
- Exact-match transfer to the original synthetic holdout
- Semantic correctness or SQL execution accuracy
- Multi-epoch training or a larger, independently sourced evaluation benchmark
- Production deployment, serving performance, or cost optimization

## Final Claim Status

- Verified: AMD MI300X ROCm smoke run and separate 800-example QLoRA experiment with saved adapters
- Preliminary evidence: adapter improved normalized exact match from 0/100 to 61/100 on a random SQL-Create-Context-derived held-out split
- Verified: local dry-run, training-data handling, backend detection, and evaluation code tests
- Not demonstrated: independent benchmark generalization, execution-based accuracy, or production readiness
