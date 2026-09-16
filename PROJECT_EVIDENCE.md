# Project Evidence

## Model Selected

- Model: Qwen2.5-3B-Instruct
- Status: SELECTED FOR DESIGN AND PIPELINE, not claimed as fully fine-tuned in this environment
- Reason: small, quantizable, instruction-tuned, practical for QLoRA and interview defense

## Dataset Source

- Source: synthetic SQL/data-engineering examples created for the project
- Status: synthetic only, not enterprise proprietary data
- Purpose: controlled SQL generation and safety evaluation

## Dataset Size

- Total examples: 30 synthetic examples in the current repository seed set
- Split: 70% train, 15% validation, 15% test
- Verified by the split script and tests

## Train/Validation/Test Split

- Train: 21 examples
- Validation: 4 examples
- Test: 5 examples

Status: VERIFIED by split script output and dataset tests.

## Training Configuration

- Model: Qwen2.5-3B-Instruct
- Method: LoRA / QLoRA intended
- Key settings: rank, alpha, dropout, target modules configured in [src/training/config.py](src/training/config.py)
- Status: IMPLEMENTED, not executed end-to-end in this environment

## Actual Training Run

Status: NOT RUN in this environment.

Evidence: no GPU-backed fine-tune was executed. This repository includes a real training entry point, but the actual training run is not claimed to have completed.

## Actual Evaluation Results

- Dataset validation tests: PASSED
- API tests: PASSED
- Inference tests: PASSED
- Base-model SQL evaluation: NOT MEASURED
- Fine-tuned evaluation: NOT RUN

## Actual Failures

Observed during implementation:

- missing library compatibility assumptions were resolved by keeping the code compatible with a minimal Python stack
- the environment is not set up for a real GPU fine-tuning run, so end-to-end LoRA training remains proposed

## Fixes

- added a dry-run training mode to avoid fake training claims
- added data validation and SQL safety checks
- kept evaluation and API code independent of model training

## Inference Tests

Status: VERIFIED by tests that call the generator and safety validation.

## API Tests

Status: VERIFIED by FastAPI client tests.

## Docker Verification

Status: NOT VERIFIED end-to-end because Docker was not run in this workspace.

## Deployment Status

Status: PROPOSED / local reproducible setup only.

## Limitations

- No actual model training was executed on a GPU
- No production deployment was performed
- Evaluation values are conservative and evidence-based only

## Final Claim Status

- Verified: dataset validation, config loading, API behavior, inference safety tests
- Partially verified: training pipeline structure and code integration
- Proposed: GPU fine-tuning, deployment, cloud inference, production monitoring
