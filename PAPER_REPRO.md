# LUT-LLM Paper Reproduction

This file tracks the paper-targeted reproduction path for LUT-LLM, separate from the earlier PQ+LUT feasibility experiments in `RESULTS.md`.

## Target Paper Setup

Paper: LUT-LLM: Efficient Language Model Inference with Memory-based Computations on FPGAs, arXiv v2, 2026-03-22.

Official artifact checked at `LUT-FPGA/LUT-LLM` commit `9ee2259d312f9b1119a398d8ff7703154260a417`. The public artifact contains FPGA/HLS code, a Qwen 3 1.7B hardware model, and latency/resource scripts. It does not provide the full PyTorch QAT/STE training code, fused lookup/reduce training kernels, GPTVQ scripts, or exact evaluation harness used for Table III.

The paper's algorithm setup is:

- Model: Qwen 3 1.7B.
- Activation VQ: `subdim=2`, `Ka=64`.
- Weight VQ: `Kw=16`, with INT8 quantized lookup tables.
- Training: KMeans initialization, QAT with STE and custom fused forward/backward kernels.
- Final conversion: reconstruct weights from trained lookup tables, apply GPTVQ, then precompute activation-codebook x weight-codebook LUTs.
- Evaluation: GLUE, SQuAD v2, and MMLU-Pro.

## Paper Table III Reference

| Method | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | SQuADv2 | MMLU-Pro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 87.6 | 86.5 | 92.9 | 91.2 | 80.9 | 93.7 | 72.8 | 33.1 |
| RTN INT8 | 86.7 | 80.2 | 88.0 | 89.3 | 70.4 | 87.4 | 62.0 | 23.6 |
| SmoothQuant | 87.0 | 85.3 | 91.7 | 89.6 | 79.1 | 91.2 | 71.3 | 31.7 |
| SpinQuant | 87.3 | 83.3 | 91.8 | 89.5 | 80.2 | 91.5 | 72.0 | 28.0 |
| LUT-LLM + Act. Quant. | 87.0 | 84.1 | 91.9 | 90.7 | 78.3 | 91.2 | 70.3 | 31.8 |
| + INT8 LUT | 86.9 | 83.8 | 91.7 | 90.8 | 76.9 | 90.7 | 69.8 | 31.3 |
| + Weight Quant. | 86.9 | 82.8 | 90.4 | 89.5 | 76.5 | 90.6 | 69.7 | 30.8 |

## Implemented Reproduction Code

New files:

- `run_paper_eval.py`: Qwen causal-LM evaluator for GLUE, SQuAD v2, and MMLU-Pro.
- `run_lutllm_qat.py`: simplified LUT-LLM-style activation QAT with STE, followed by activation-weight LUT conversion.
- `pq_lut_lm/paper_eval.py`: prompt-based GLUE/MMLU-Pro log-likelihood scoring and SQuADv2 short generation/F1.
- `pq_lut_lm/activation_quant.py`: trainable activation VQ wrapper with STE.

Important limitation: this is not yet a byte-identical reproduction of the paper. The missing pieces are the paper's custom fused QAT kernels, adjustable-gradient STE details, GPTVQ implementation, and exact benchmark harness/prompts. The current code is a transparent PyTorch reproduction scaffold that runs the same model family and datasets but not the undisclosed training/eval stack.

## scai7 Runs

### Prompt Evaluator Baselines

These runs use 128 validation/test examples per task. They are not directly comparable to Table III because the paper's exact evaluation harness is not public in the artifact, and this repo uses simple prompt-based scoring/generation.

| Run | Model | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | SQuADv2 F1 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_baseline_qwen3_1p7b_128` | `Qwen/Qwen3-1.7B` | 42.2 | 70.3 | 46.1 | 28.1 | 52.3 | 77.3 | 12.2 | 23.4 |
| `paper_baseline_qwen3_1p7b_base_128` | `Qwen/Qwen3-1.7B-Base` | 41.4 | 74.2 | 74.2 | 56.2 | 52.3 | 57.0 | 36.0 | 28.9 |
| Paper FP16 | Qwen 3 1.7B | 87.6 | 86.5 | 92.9 | 91.2 | 80.9 | 93.7 | 72.8 | 33.1 |

The baseline mismatch is large on GLUE and SQuAD, so the exact paper numbers cannot be reproduced by this prompt evaluator alone.

### Simplified Full-Layer QAT Smoke

Run: `paper_lutllm_qwen3_1p7b_all_qat20`

Config:

- Model: `Qwen/Qwen3-1.7B`.
- Quantized linears: all 196 transformer block linears matching `(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$`.
- Activation VQ: `subdim=2`, `Ka=64`, Chebyshev assignment.
- Weight VQ: `Kw=16`, `weight_group_size=256`.
- LUT: INT8 min/max quantized values.
- QAT: 20 STE steps on WikiText train text, sequence length 64.
- Evaluation: 16 examples per GLUE task and MMLU-Pro; SQuAD skipped for this run because final LUT evaluation is slow in unfused PyTorch.

| Stage | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline, same 16 rows | 25.0 | 62.5 | 37.5 | 37.5 | 62.5 | 87.5 | 43.8 |
| + Act. Quant., simplified STE | 25.0 | 37.5 | 62.5 | 62.5 | 62.5 | 56.2 | 6.2 |
| + Weight Quant., final LUT | 43.8 | 37.5 | 62.5 | 56.2 | 62.5 | 50.0 | 6.2 |

Hardware aggregate for final LUT:

| Quantized Linears | Compact INT8 LUT | Weight Codes Packed | Lookups / Token | Act Code Bits / Token | Theoretical Expanded LUT FP16 |
|---:|---:|---:|---:|---:|---:|
| 196 | 2,688.0 MiB | 336.0 MiB | 704,643,072 | 1,548,288 | 86,016.0 MiB |

This confirms the paper-shaped codebook and lookup scale for Qwen 3 1.7B, but the accuracy is not paper-level. The main reason is that the run uses a simplified PyTorch STE path with 20 steps and no GPTVQ, while the paper reports an optimized QAT recipe costing about 10 A100 GPU-hours.

## Commands

Baseline:

```bash
python3 run_paper_eval.py \
  --model-id Qwen/Qwen3-1.7B \
  --output-dir results/paper_baseline_qwen3_1p7b_128 \
  --paper-samples 128
```

Simplified QAT:

```bash
python3 run_lutllm_qat.py \
  --model-id Qwen/Qwen3-1.7B \
  --output-dir results/paper_lutllm_qwen3_1p7b_all_qat20 \
  --paper-samples 16 \
  --skip-squad \
  --seq-len 64 \
  --train-tokens 2048 \
  --calib-tokens 512 \
  --calib-batches 4 \
  --calib-vectors-per-layer 128 \
  --train-steps 20 \
  --kmeans-iters 1 \
  --sample-limit 128 \
  --eval-baseline \
  --eval-act-quant \
  --eval-final-lut
```

Stronger paper-supervised QAT mode:

```bash
python3 run_lutllm_qat.py \
  --model-id Qwen/Qwen3-1.7B \
  --output-dir results/paper_lutllm_qwen3_1p7b_all_paperqat100_affine \
  --train-source paper \
  --task-train-samples 32 \
  --paper-samples 32 \
  --skip-squad \
  --seq-len 128 \
  --calib-batches 8 \
  --calib-vectors-per-layer 256 \
  --train-steps 100 \
  --lr 3e-4 \
  --kmeans-iters 1 \
  --sample-limit 256 \
  --output-correction affine \
  --eval-baseline \
  --eval-act-quant \
  --eval-final-lut
```

The `--train-source paper` mode trains activation codebooks on supervised GLUE/SQuAD/MMLU-Pro-style prompt+answer examples instead of WikiText continuation loss. `--output-correction affine` fits a post-hoc per-output affine correction during final LUT conversion; this is not GPTVQ, but it is a useful lightweight approximation for reducing final layer-output error.
