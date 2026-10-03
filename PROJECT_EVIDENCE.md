# Project Evidence

## Project Scope

- Model: Qwen2.5-3B-Instruct
- Method: QLoRA NF4 with LoRA rank 16
- Data: synthetic SQL/data-engineering examples only; no proprietary data
- Purpose: verify the end-to-end fine-tuning pipeline, not claim production model quality

## Dataset

- Total examples: 28
- Train: 19
- Validation: 4
- Test: 5
- Split: 70% / 15% / 15%, verified by dataset tests

The dataset is intentionally tiny. It is useful for a smoke run and code-path validation, not a representative SQL benchmark.

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
- Training-data prompt masking and padding tests: passed
- ROCm backend precedence and device-count tests: passed
- Dry-run: passed; validated 19 train and 4 validation examples without downloading a model or writing artifacts
- CPU + 4-bit guard: passed; refuses to claim GPU training on CPU
- Base-vs-adapter evaluation metrics: tested with mocked predictions

## Failure and Fix

The first AMD attempt loaded the base model and LoRA weights but stopped before training because Transformers 5.18 removed the `warmup_ratio` argument. The trainer now maps the configured ratio to `warmup_steps` for newer Transformers while retaining `warmup_ratio` for versions that support it. The subsequent AMD run completed.

## Not Verified

- Useful fine-tuned quality or improvement over the base model
- Semantic correctness or SQL execution accuracy
- Multi-epoch training or a larger, representative dataset
- Production deployment, serving performance, or cost optimization

## Final Claim Status

- Verified: working AMD MI300X ROCm QLoRA smoke run and saved adapter
- Verified: local dry-run, training-data handling, backend detection, and evaluation code tests
- Not demonstrated: meaningful model improvement or production readiness
