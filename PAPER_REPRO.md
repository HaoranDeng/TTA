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
- `run_act_lut_fit.py`: layerwise direct activation-LUT fitting, least-squares reconstruction of dense weights from trained tables, and final activation-weight VQ.
- `pq_lut_lm/paper_eval.py`: prompt-based GLUE/MMLU-Pro log-likelihood scoring and SQuADv2 short generation/F1.
- `pq_lut_lm/activation_quant.py`: trainable activation VQ wrapper with STE, direct activation-LUT modules, and LUT-to-weight reconstruction.

Important limitation: this is not yet a byte-identical reproduction of the paper. The missing pieces are the paper's custom fused QAT kernels, adjustable-gradient STE details, GPTVQ implementation, and exact benchmark harness/prompts. The current code is a transparent PyTorch reproduction scaffold that runs the same model family and datasets but not the undisclosed training/eval stack.

## scai7 Runs

### Prompt Evaluator Baselines

These runs use 128 validation/test examples per task. They are not directly comparable to Table III because the paper's exact evaluation harness is not public in the artifact, and this repo uses simple prompt-based scoring/generation.

Updated protocol search: the original plain prompts were the main cause of the huge FP16 mismatch on MMLU-Pro and several GLUE tasks. Larger sweeps on scai7 show that the raw public checkpoints still do not match the paper FP16 row. The closest public-checkpoint protocol found so far is `Qwen/Qwen3-1.7B-Base` with instruction few-shot prompts, but the best 512-example GLUE average is still `82.35` versus the paper's `88.80`.

Current baseline-alignment status:

| Run | Protocol | Samples | GLUE Avg | Gap vs Paper FP16 GLUE | MMLU-Pro | Gap vs Paper FP16 MMLU |
|---|---|---:|---:|---:|---:|---:|
| Paper FP16 | customized Qwen 3 1.7B | full paper eval | 88.80 | 0.00 | 33.10 | 0.00 |
| `lutllm_base_instruction_g8_all196_shufcalib_ka64_calib1024_k5_init_actonly_ppl256` FP16 | internal instruction 8-shot GLUE, 0-shot MMLU | 256/task | 82.88 | -5.92 | 29.69 | -3.41 |
| `lutllm_base_instruction_g8_all196_shufcalib_steqat1000_int8_actonly_ppl128` FP16 | internal instruction 8-shot GLUE, 0-shot MMLU | 128/task | 83.07 | -5.73 | 33.59 | +0.49 |
| `baseline_prompt_grid_qwen3_1p7b_base_512_more_shots/instruction_g8_m0_plain` | internal instruction 8-shot GLUE, 0-shot MMLU | 512/task | 82.35 | -6.45 | 28.52 | -4.58 |
| `baseline_prompt_grid_qwen3_1p7b_base_512_more_shots/instruction_g16_m8_plain` | internal instruction 16-shot GLUE, 8-shot MMLU | 512/task | 81.52 | -7.28 | 30.27 | -2.83 |
| `lmeval_qwen3_1p7b_base_glue6_limit1024` | standard EleutherAI `lm_eval` GLUE prompts | 1024/task limit | 71.63 | -17.17 | - | - |
| `lmeval_qwen3_1p7b_glue6_limit1024` | standard `lm_eval`, non-Base public checkpoint | 1024/task limit | 62.01 | -26.79 | - | - |
| `lmeval_taskadapt_glue1024_500_glue6_limit1024` | simple GLUE-only task adaptation, then standard `lm_eval` | 1024/task limit | 67.97 | -20.83 | - | - |

Interpretation: standard public `lm_eval` prompts do not explain the paper's FP16 row; they make GLUE substantially worse. Simple GLUE-only continued training for 500 updates also failed to move toward the paper. The remaining FP16 mismatch is therefore likely due to the paper's customized checkpoint and/or an undisclosed task/evaluation protocol. All quantized all-196-linear results below remain useful diagnostics, but they are not a faithful reproduction until this FP16 row is matched.

Latest all-196 diagnostic gap:

| Stage | GLUE Avg | Paper Target | Gap | MMLU-Pro | Paper Target | Gap | WikiText PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline | 82.88 | 88.80 | -5.92 | 29.69 | 33.10 | -3.41 | 16.45 |
| Act Quant, `Ka=64`, no QAT | 61.07 | 87.20 | -26.13 | 6.25 | 31.80 | -25.55 | 332.60 |
| centers-only STE Act Quant, 1000 steps | 70.96 | 87.20 | -16.24 | 7.81 | 31.80 | -23.99 | 335.62 |
| `subdim=4, Ka=64` centers-only STE Act Quant, 1000 steps | 46.09 | 87.20 | -41.11 | 7.81 | 31.80 | -23.99 | 19,947.91 |
| `subdim=2, Ka=128` centers-only STE Act Quant, 1000 steps | 68.75 | 87.20 | -18.45 | 7.81 | 31.80 | -23.99 | 217.62 |

Prompt grid, 64 examples/task, SQuAD skipped:

| Run | Model / Prompt | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_prompt_grid_qwen3_1p7b_64/simple_g0_m0_plain` | `Qwen3-1.7B`, simple | 32.8 | 67.2 | 53.1 | 34.4 | 50.0 | 85.9 | 20.3 |
| `baseline_prompt_grid_qwen3_1p7b_64/instruction_g3_m3_chat` | `Qwen3-1.7B`, instruction, 3-shot, chat | 51.6 | 70.3 | 62.5 | 67.2 | 81.2 | 96.9 | 29.7 |
| `baseline_prompt_grid_qwen3_1p7b_base_64/instruction_g0_m0_plain` | `Qwen3-1.7B-Base`, instruction | 82.8 | 67.2 | 81.2 | 84.4 | 78.1 | 87.5 | 39.1 |
| `baseline_prompt_grid_qwen3_1p7b_base_64/instruction_g3_m3_plain` | `Qwen3-1.7B-Base`, instruction, 3-shot | 85.9 | 71.9 | 82.8 | 85.9 | 73.4 | 96.9 | 31.2 |

Best 128-example public-checkpoint baseline:

| Run | Model / Prompt | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | SQuADv2 F1 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_qwen3_1p7b_base_instruction_128` | `Qwen/Qwen3-1.7B-Base`, instruction | 78.9 | 68.0 | 78.1 | 85.2 | 80.5 | 86.7 | 36.0 | 33.6 |
| Paper FP16 | customized Qwen 3 1.7B | 87.6 | 86.5 | 92.9 | 91.2 | 80.9 | 93.7 | 72.8 | 33.1 |

| Run | Model | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | SQuADv2 F1 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_baseline_qwen3_1p7b_128` | `Qwen/Qwen3-1.7B` | 42.2 | 70.3 | 46.1 | 28.1 | 52.3 | 77.3 | 12.2 | 23.4 |
| `paper_baseline_qwen3_1p7b_base_128` | `Qwen/Qwen3-1.7B-Base` | 41.4 | 74.2 | 74.2 | 56.2 | 52.3 | 57.0 | 36.0 | 28.9 |
| `paper_baseline_qwen3_1p7b_chat_128` | `Qwen/Qwen3-1.7B`, chat template | 29.7 | 68.0 | 56.2 | 51.6 | 52.3 | 81.2 | 33.9 | 10.9 |
| Paper FP16 | Qwen 3 1.7B | 87.6 | 86.5 | 92.9 | 91.2 | 80.9 | 93.7 | 72.8 | 33.1 |

The baseline mismatch is large on GLUE and SQuAD, so the exact paper numbers cannot be reproduced by this prompt evaluator alone. Applying Qwen3's chat template improves SQuAD, QNLI, QQP, and SST-2 relative to the raw prompt on this small sample, but it hurts MMLU-Pro and still does not approach the paper's FP16 baseline. This points to a missing evaluation or task-adaptation protocol rather than a simple prompt-format issue.

Note: after commit `7c78c60`, supervised GLUE calibration/training batches use the GLUE train split. Earlier paper-supervised diagnostic runs were useful for debugging the quantizer but should not be treated as strict held-out evaluation if they were produced before this fix.

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

### Paper-Supervised QAT Runs

The initial 20-step smoke used WikiText continuation loss, which is not aligned with Table III tasks. The follow-up runs use `--train-source paper`, which trains on supervised prompt+gold-answer examples from GLUE, SQuADv2, and MMLU-Pro-style data. They also enable `--output-correction affine` during final LUT conversion as a lightweight approximation to reduce final layer-output error. This is still not the paper's GPTVQ implementation.

Run: `paper_lutllm_qwen3_1p7b_all_paperqat100_affine`

Config:

- Model: `Qwen/Qwen3-1.7B`.
- Quantized linears: all 196 target transformer block linears.
- QAT: 100 supervised steps, 32 training examples per paper task, sequence length 128.
- Evaluation: 32 examples per GLUE task and MMLU-Pro; SQuAD skipped because final LUT evaluation is slow in the unfused PyTorch prototype.

| Stage | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline, same 32 rows | 28.1 | 62.5 | 56.2 | 31.2 | 56.2 | 87.5 | 28.1 |
| + Act. Quant., paper-supervised STE | 37.5 | 62.5 | 56.2 | 34.4 | 56.2 | 62.5 | 6.2 |
| + Weight Quant., final LUT | 40.6 | 37.5 | 40.6 | 68.8 | 56.2 | 43.8 | 6.2 |

Hardware aggregate for final LUT:

| Quantized Linears | Compact INT8 LUT | Weight Codes Packed | Lookups / Token | Act Code Bits / Token | Theoretical Expanded LUT FP16 |
|---:|---:|---:|---:|---:|---:|
| 196 | 2,688.0 MiB | 336.0 MiB | 704,643,072 | 1,548,288 | 86,016.0 MiB |

Run: `paper_lutllm_qwen3_1p7b_7linear_paperqat500_affine`

Config:

- Model: `Qwen/Qwen3-1.7B`.
- Quantized linears: first 7 target linears only.
- QAT: 500 supervised steps, 128 training examples per paper task, sequence length 256.
- Evaluation: 64 examples per GLUE task, SQuADv2, and MMLU-Pro.

| Stage | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | SQuADv2 F1 | MMLU-Pro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline, same 64 rows | 32.8 | 67.2 | 53.1 | 34.4 | 50.0 | 85.9 | 10.5 | 20.3 |
| + Act. Quant., paper-supervised STE | 48.4 | 67.2 | 75.0 | 71.9 | 50.0 | 75.0 | 9.8 | 14.1 |
| + Weight Quant., final LUT | 48.4 | 67.2 | 53.1 | 46.9 | 50.0 | 75.0 | 4.1 | 14.1 |

Hardware aggregate for final LUT:

| Quantized Linears | Compact INT8 LUT | Weight Codes Packed | Lookups / Token |
|---:|---:|---:|---:|
| 7 | 96.0 MiB | 12.0 MiB | 25,165,824 |

The 7-linear run shows that supervised task loss can train the activation centroids: training loss dropped from `12.945` to `0.130`, and Act Quant improved several prompt-eval metrics on the small sample. The full-layer run still does not reproduce paper accuracy; likely missing factors are exact evaluation harness, full QAT duration and kernel behavior, adjustable-gradient STE details, and GPTVQ.

Run: `paper_lutllm_qwen3_1p7b_all_paperqat1000_actonly`

Config:

- Model: `Qwen/Qwen3-1.7B`.
- Quantized linears: all 196 target transformer block linears.
- QAT: 1000 supervised steps, 256 training examples per paper task, sequence length 256.
- Evaluation: 128 examples per GLUE task and MMLU-Pro; SQuAD skipped.
- Final LUT conversion skipped to isolate the paper's `+ Act. Quant.` stage.

| Stage | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline, same 128 rows | 42.2 | 70.3 | 46.1 | 28.1 | 52.3 | 77.3 | 23.4 |
| + Act. Quant., paper-supervised STE | 35.9 | 68.0 | 46.1 | 32.0 | 52.3 | 53.9 | 13.3 |

The 1000-step full-layer activation-only run reduced supervised training loss from `7.541` to `4.772`, but it still did not approach the paper's `+ Act. Quant.` accuracy. This strengthens the conclusion that reproducing Table III requires missing pieces from the paper implementation: exact evaluation/task-adaptation setup, full QAT recipe, adjustable-gradient STE details, and GPTVQ/fused-kernel behavior.

### Corrected Base+Instruction All-196-Linear Runs

After the prompt grid above, the formal public-checkpoint reproduction path is `Qwen/Qwen3-1.7B-Base` with instruction prompts, quantizing all 196 transformer-block linear layers matching `q/k/v/o/gate/up/down_proj`. These runs preserve the paper-shaped codebooks: `subdim=2`, `Ka=64`, `Kw=16`, Chebyshev activation assignment, `weight_group_size=256`, and INT8 compact final LUTs. Supervised calibration/training uses GLUE train split after commit `7c78c60`.

256 examples/task, 8-shot GLUE instruction prompt, 0-shot MMLU-Pro, SQuAD skipped, all 196 target linears:

| Run | Stage | WikiText PPL | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lutllm_base_instruction_g8_all196_shufcalib_ka64_calib1024_k5_init_actonly_ppl256` | FP16 baseline | 16.45 | 81.6 | 74.2 | 82.8 | 84.4 | 79.7 | 94.5 | 29.7 |
| same | Act Quant, `Ka=64`, no QAT | 332.60 | 50.4 | 70.3 | 60.5 | 68.4 | 62.9 | 53.9 | 6.2 |

This is the closest current public-checkpoint reproduction protocol before QAT. It still fails the paper reproduction target: same-run FP16 is `-5.92` GLUE points below paper FP16, and no-QAT activation quantization is `-26.13` GLUE points below the paper `+ Act. Quant.` row. The quantization drop is `-21.81` GLUE points here, while the paper reports roughly `-1.63` from FP16 to `+ Act. Quant.`.

128 examples/task, 8-shot GLUE instruction prompt, 0-shot MMLU-Pro, SQuAD skipped, all 196 target linears:

| Run | Stage | WikiText PPL | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lutllm_base_instruction_g8_all196_shufcalib_steqat1000_int8_actonly_ppl128` | FP16 baseline | 16.45 | 81.2 | 75.0 | 82.0 | 87.5 | 79.7 | 93.0 | 33.6 |
| same | centers-only STE Act Quant, 1000 steps | 335.62 | 68.0 | 75.0 | 62.5 | 64.1 | 72.7 | 83.6 | 7.8 |

This QAT run trained `33,030,144` activation-center parameters for 1000 supervised steps and quantized all 196 transformer-block linears. It improves GLUE over no-QAT activation quantization, but it still does not reproduce the paper: GLUE is `70.96` versus paper `+ Act. Quant.` `87.20` (`-16.24`), and MMLU-Pro is `7.81` versus `31.80` (`-23.99`). The same-run quantization drop is `-12.11` GLUE points and `-25.78` MMLU-Pro points, while the paper's drop from FP16 to `+ Act. Quant.` is about `-1.63` GLUE and `-1.30` MMLU-Pro.

64 examples/task, 8-shot GLUE instruction prompt, 0-shot MMLU-Pro, SQuAD skipped, all 196 target linears, testing the official artifact's `vec_len=4`-style branch:

| Run | Stage | WikiText PPL | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lutllm_base_instruction_g8_all196_subdim4_ka64_steqat1000_actonly_ppl64` | FP16 baseline | 16.45 | 84.4 | 73.4 | 81.2 | 87.5 | 76.6 | 96.9 | 39.1 |
| same | `subdim=4, Ka=64` centers-only STE Act Quant, 1000 steps | 19,947.91 | 29.7 | 40.6 | 29.7 | 67.2 | 50.0 | 59.4 | 7.8 |

This branch halves lookup count and expanded Act-LUT size (`352,321,536` lookups/token and `43,008.0` MiB expanded FP16 Act-LUT), but accuracy collapses harder than `subdim=2`. It therefore does not explain the paper's Table III accuracy under the current reproduction scaffold.

64 examples/task, 8-shot GLUE instruction prompt, 0-shot MMLU-Pro, SQuAD skipped, all 196 target linears, testing a larger activation codebook:

| Run | Stage | WikiText PPL | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lutllm_base_instruction_g8_all196_ka128_steqat1000_actonly_ppl64` | FP16 baseline | 16.45 | 84.4 | 73.4 | 81.2 | 87.5 | 76.6 | 96.9 | 39.1 |
| same | `subdim=2, Ka=128` centers-only STE Act Quant, 1000 steps | 217.62 | 43.8 | 76.6 | 62.5 | 71.9 | 64.1 | 93.8 | 7.8 |

Increasing `Ka` from 64 to 128 improves WikiText PPL relative to the 128-sample `Ka=64` QAT run (`217.62` vs `335.62`) but does not improve the paper-targeted accuracy gap. GLUE is `68.75`, still `-18.45` below the paper `+ Act. Quant.` row, and MMLU-Pro remains `7.81`, `-23.99` below the paper. The hardware cost doubles activation centers to `66,060,288`, doubles centroid-distance vectors to `33,030,144` per token, and increases expanded Act-LUT FP16 size to `172,032.0` MiB while lookup count stays `704,643,072` per token.

64 examples/task, SQuAD skipped, all 196 target linears:

| Run | Stage | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `lutllm_base_instruction_all196_batched_traincalib_steqat1000_int8_64_actonly` | FP16 baseline | 82.8 | 67.2 | 81.2 | 84.4 | 78.1 | 87.5 | 39.1 |
| same | simplified STE Act Quant | 37.5 | 68.8 | 51.6 | 46.9 | 51.6 | 60.9 | 9.4 |

16 examples/task, SQuAD skipped, all 196 target linears:

| Run | Stage | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `lutllm_base_instruction_all196_batched_traincalib_actlutfit10_int8_final16_v4` | FP16 baseline | 87.5 | 56.2 | 75.0 | 75.0 | 87.5 | 81.2 | 62.5 |
| same | reconstructed final LUT | 31.2 | 25.0 | 68.8 | 50.0 | 62.5 | 50.0 | 6.2 |

Hardware aggregate for the all-196 final LUT:

| Quantized Linears | Compact INT8 LUT | Weight Codes Packed | Lookups / Token | Act Code Bits / Token | Expanded Act-LUT FP16 Intermediate |
|---:|---:|---:|---:|---:|---:|
| 196 | 2,688.0 MiB | 336.0 MiB | 704,643,072 | 1,548,288 | 86,016.0 MiB |

Interpretation: once all 196 linears are quantized, the current simplified STE/QAT recipe and inferred final-LUT reconstruction are far below the paper. This is now a more meaningful failure mode than the earlier partial-layer runs: the missing pieces are likely the paper's customized checkpoint/task adaptation, exact STE gradient recipe, fused training implementation, and GPTVQ details.

Implementation note: full all-196 runs exposed Python prototype bottlenecks, so commits `ead46f9`, `28fe1be`, `0434a8b`, and `2c813e1` add batched activation k-means, batched LUT-to-weight reconstruction, calibration-input reuse during conversion, and batched final weight VQ. These are speed fixes for the reproduction scaffold; they do not change the intended all-layer quantization scope.

### Shuffled Calibration and PPL Runs

Commit `3708f19` shuffles paper-supervised training/calibration batches before taking the calibration prefix. This fixes a reproduction artifact in earlier `traincalib` runs: because examples were concatenated task-by-task, the first calibration batches were mostly MNLI. The following runs supersede the earlier unshuffled all-196 act-only rows for calibration-order analysis.

64 examples/task, SQuAD skipped, `Qwen/Qwen3-1.7B-Base`, instruction prompt, all 196 block linears unless noted:

| Run | Stage | WikiText PPL | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lutllm_base_instruction_all196_shufcalib_ka64_calib1024_k5_init_actonly_ppl64` | FP16 baseline | 16.4 | 82.8 | 67.2 | 81.2 | 84.4 | 78.1 | 87.5 | 39.1 |
| same | Act Quant, `Ka=64`, no QAT | 332.6 | 39.1 | 71.9 | 67.2 | 65.6 | 54.7 | 62.5 | 1.6 |
| `lutllm_base_instruction_all196_shufcalib_ka256_calib512_k3_init_actonly_ppl64` | Act Quant, `Ka=256`, no QAT | 250.9 | 50.0 | 73.4 | 67.2 | 65.6 | 59.4 | 65.6 | 3.1 |
| `lutllm_base_instruction_all196_shufcalib_steqat1000_int8_64_actonly_ppl64` | simplified STE Act Quant, 1000 steps | 413.4 | 51.6 | 68.8 | 62.5 | 75.0 | 60.9 | 70.3 | 9.4 |

16 examples/task, diagnostic run including `lm_head` so every `nn.Linear` is quantized:

| Run | Scope | Stage | WikiText PPL | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lutllm_base_instruction_all197_includelmhead_shufcalib_ka64_calib512_k3_actonly_ppl16` | 197 linears incl. `lm_head` | FP16 baseline | 15.6 | 87.5 | 56.2 | 75.0 | 75.0 | 87.5 | 81.2 | 62.5 |
| same | 197 linears incl. `lm_head` | Act Quant, `Ka=64`, no QAT | 403.4 | 18.8 | 50.0 | 50.0 | 62.5 | 56.2 | 62.5 | 6.2 |

Hardware scale for the shuffled-calibration runs:

| Run | Quantized Linears | Activation Centers | Expanded Act-LUT FP16 | Lookups / Token | Act Code Bits / Token | Centroid Distance Vectors / Token |
|---|---:|---:|---:|---:|---:|---:|
| all196 `Ka=64` | 196 | 33,030,144 | 86,016.0 MiB | 704,643,072 | 1,548,288 | 16,515,072 |
| all196 `Ka=256` | 196 | 132,120,576 | 344,064.0 MiB | 704,643,072 | 2,064,384 | 66,060,288 |
| all196 `subdim=4, Ka=64` | 196 | 33,030,144 | 43,008.0 MiB | 352,321,536 | 774,144 | 8,257,536 |
| all197 `Ka=64`, incl. `lm_head` | 197 | 33,161,216 | 105,008.0 MiB | 860,225,536 | 1,554,432 | 16,580,608 |

Interpretation: shuffling calibration removes the task-order artifact, and increasing the activation codebook from `Ka=64` to `Ka=256` improves PPL and several GLUE metrics slightly. It still does not approach the paper's `+ Act. Quant.` row. The current centers-only STE QAT can improve some classification metrics, but it worsens WikiText PPL and remains far from paper accuracy. This points back to the missing paper pieces: trainable lookup-table values or fused lookup/reduce QAT, adjustable-gradient STE, GPTVQ, and possibly the customized checkpoint/task adaptation.

### Additional All-196 Reproduction Attempts

These runs continue from the corrected shuffled-calibration setup. All rows quantize all 196 transformer-block linears unless explicitly noted.

| Run | Stage | Samples | WikiText PPL | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lutllm_base_instruction_all196_wikitext_steqat1000_int8_64_actonly_ppl64` | WikiText STE Act Quant, 1000 steps | 64 | 104.5 | 37.5 | 34.4 | 50.0 | 64.1 | 51.6 | 56.2 | 12.5 |
| `lutllm_base_instruction_all196_wikitext_steqat3000_lr1e4_int8_64_actonly_ppl64` | WikiText STE Act Quant, 3000 steps, lr=1e-4 | 64 | 101.3 | 40.6 | 34.4 | 46.9 | 62.5 | 50.0 | 50.0 | 10.9 |
| `lutllm_base_instruction_all196_wikitext_steqat5000_int8_64_actonly_ppl64` | WikiText STE Act Quant, 5000 steps, lr=3e-4 | 64 | 749.3 | 34.4 | 32.8 | 46.9 | 65.6 | 50.0 | 51.6 | 9.4 |
| `lutllm_base_instruction_all196_wikitext_softhard_steqat1000_temp05_actonly_ppl64` | WikiText soft-hard STE, temp=0.5, 1000 steps | 64 | 467.1 | 31.2 | 32.8 | 46.9 | 67.2 | 50.0 | 56.2 | 10.9 |
| `lutllm_base_instruction_all196_wikitext_hard_steqat1000_ingrad0_actonly_ppl64` | WikiText hard STE, input-gradient scale=0, 1000 steps | 64 | 499.7 | 34.4 | 32.8 | 48.4 | 65.6 | 50.0 | 51.6 | 14.1 |
| `lutllm_base_instruction_all196_shufcalib_subdim4_ka64_calib1024_k5_init_actonly_ppl64` | Act Quant, `subdim=4`, `Ka=64`, no QAT | 64 | 3,356.9 | 32.8 | 57.8 | 48.4 | 31.2 | 50.0 | 46.9 | 9.4 |
| `lutllm_base_instruction_all196_shufcalib_denseqat300_centers3e4_dense1e5_actonly_ppl32` | centers + dense linear QAT, 300 steps | 32 | 472.2 | 40.6 | 75.0 | 53.1 | 68.8 | 65.6 | 78.1 | 3.1 |
| `lutllm_base_instruction_all196_shufcalib_steqat500_int8_final32_ppl` | Act Quant before final LUT | 32 | 464.3 | 50.0 | 68.8 | 65.6 | 62.5 | 65.6 | 81.2 | 15.6 |
| same | compact INT8 final LUT with local k-means weight VQ | 32 | 39,716.5 | 31.2 | 40.6 | 28.1 | 59.4 | 56.2 | 37.5 | 3.1 |
| `lutllm_base_instruction_all196_shufcalib_steqat500_int8_final16_affine_ppl` | compact INT8 final LUT + per-output affine correction | 16 | 10,768.6 | 31.2 | 37.5 | 62.5 | 62.5 | 62.5 | 37.5 | 6.2 |
| `lutllm_base_instruction_all196_shufcalib_steqat500_int8_final16_reassign1_ppl` | compact INT8 final LUT + output-aware weight-code reassignment, 1 pass | 16 | 98,591.5 | 25.0 | 37.5 | 62.5 | 62.5 | 62.5 | 50.0 | 12.5 |
| `lutllm_base_instruction_all196_shufcalib_steqat500_int8_final16_centerrefine1_ppl` | compact INT8 final LUT + LS weight-center refinement | 16 | 945,281.6 | 25.0 | 37.5 | 62.5 | 62.5 | 62.5 | 31.2 | 0.0 |
| `lutllm_base_instruction_all196_shufcalib_steqat500_int8_final16_centerrefine1_blend01_ppl` | compact INT8 final LUT + damped LS center refinement, blend=0.1 | 16 | 945,281.6 | 25.0 | 37.5 | 62.5 | 62.5 | 62.5 | 31.2 | 0.0 |
| `lutllm_base_instruction_all196_shufcalib_actlutfit50_actonly16_ppl` | direct expanded Act-LUT fit, 50 local steps | 16 | 608.5 | 25.0 | 81.2 | 68.8 | 62.5 | 68.8 | 50.0 | 0.0 |

Findings:

- WikiText LM-loss QAT improves PPL relative to task-supervised centers-only QAT, but the best PPL is still about `101`, far from the FP16 `16.4`, and GLUE/MMLU-Pro collapse remains.
- Longer centers-only QAT is not monotonic: `5000` steps at `3e-4` degraded PPL to `749`.
- Soft-hard activation STE and an input-gradient scale of `0` were tested as inferred adjustable-gradient variants; both were worse than the hard STE WikiText QAT baseline.
- `subdim=4` was tested because the official artifact's setting file mentions `vec_len=4`. It halves lookups and the expanded activation-LUT size but gives much worse PPL, so it does not explain the paper gap in this scaffold.
- Naively unfreezing dense linear weights during QAT did not help; it improved a few small-sample GLUE cells but damaged PPL and MMLU-Pro.
- The compact final LUT path is currently dominated by local k-means weight VQ error: PPL jumps to about `39,716`.
- A per-output affine correction reduces the diagnostic final-LUT PPL to about `10,769`, but GLUE/MMLU-Pro accuracy remains near random.
- The first implemented output-aware weight-code reassignment pass (`--weight-code-reassign-iters 1`) did not help: with 128 calibration vectors per layer it reached PPL about `98,591`, although MMLU-Pro moved from `6.2` to `12.5` on a very small 16-row diagnostic.
- Least-squares weight-center refinement, both undamped and with `blend=0.1`, overfit or destabilized the compact LUT and reached PPL about `945,282`.
- Direct local expanded Act-LUT fitting is also insufficient, so simply training lookup-table values layerwise against dense linear outputs does not reproduce the paper's end-to-end QAT.

Next implementation target: replace the current one-pass code reassignment with a closer GPTQ/GPTVQ-style weight quantizer that updates codes and/or centers under a second-order or blockwise reconstruction objective. The one-pass diagnostic shows that simple coordinate reassignment is not enough.

### Historical 7-Linear Debug Runs

The following runs quantize only the first 7 target linears. They are retained for debugging/profiling history only and should not be interpreted as formal LUT-LLM reproduction results.

64 examples/task, SQuAD skipped:

| Run | Stage | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `lutllm_base_instruction_7linear_traincalib_actlutfit50_int8_64` | FP16 baseline | 82.8 | 67.2 | 81.2 | 84.4 | 78.1 | 87.5 | 39.1 |
| same | direct Act LUT | 89.1 | 62.5 | 81.2 | 75.0 | 79.7 | 84.4 | 32.8 |
| same | reconstructed final LUT | 67.2 | 73.4 | 68.8 | 54.7 | 54.7 | 62.5 | 21.9 |
| `lutllm_base_instruction_7linear_traincalib_steqat300_int8_64` | simplified STE Act Quant | 70.3 | 62.5 | 79.7 | 79.7 | 85.9 | 90.6 | 28.1 |
| same | final LUT | 45.3 | 57.8 | 78.1 | 68.8 | 70.3 | 85.9 | 21.9 |

128 examples/task with SQuAD, act-only direct LUT:

| Run | Stage | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | SQuADv2 F1 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_qwen3_1p7b_base_instruction_128` | FP16 baseline | 78.9 | 68.0 | 78.1 | 85.2 | 80.5 | 86.7 | 36.0 | 33.6 |
| `lutllm_base_instruction_7linear_traincalib_actlutfit50_int8_128_actonly` | direct Act LUT | 83.6 | 64.1 | 82.8 | 78.1 | 82.8 | 82.0 | 37.7 | 26.6 |

Hardware aggregate for these 7-linear runs:

| Stage | Compact LUT / Expanded Table | Weight Codes Packed | Lookups / Token | Act Code Bits / Token |
|---|---:|---:|---:|---:|
| direct Act LUT | 3,072.0 MiB FP16 expanded table | 0.0 MiB | 25,165,824 | 55,296 |
| final activation-weight LUT | 96.0 MiB compact INT8 LUT | 12.0 MiB | 25,165,824 | 55,296 |

Interpretation: once the baseline protocol is corrected, direct activation-LUT fitting can roughly preserve some GLUE tasks and gives MMLU-Pro in the same range as the paper's `+ Act. Quant.` row on the 64-sample run. The final weight-VQ conversion is still much worse than the paper's final row, which points specifically to the missing GPTVQ and trained-LUT reconstruction details rather than just to prompt formatting.

### Direct Activation-LUT Fitting Inference

The paper says the final flow reconstructs weights from trained lookup tables before applying GPTVQ. The public artifact does not include that training code, but the HLS path confirms the hardware-facing shape: 64 activation centroids, 16 weight centroids, 4-bit packed final LUT entries in the `*_final_v2` kernels, 4-bit weight indices, and scale/zero dequantization before emitting FP32 streams.

To test that inferred flow, `run_act_lut_fit.py` fits each activation lookup table directly against the dense linear output, reconstructs dense weights with a least-squares solve `pinv(activation_centers) @ trained_table`, and then applies the existing activation-weight VQ path. This is still not GPTVQ or the paper's end-to-end fused STE training; it is a diagnostic reproduction attempt for the missing "trained LUT -> reconstructed weights" step.

7-linear runs, 64 rows/task, SQuAD skipped:

| Run | Stage | LUT Bits | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_lutllm_qwen3_1p7b_7linear_actlutfit5_int4_fast` | direct Act LUT | 16-bit expanded | 35.9 | 67.2 | 53.1 | 35.9 | 50.0 | 73.4 | 12.5 |
| `paper_lutllm_qwen3_1p7b_7linear_actlutfit5_int4_fast` | reconstructed final LUT | 4-bit | 34.4 | 32.8 | 46.9 | 64.1 | 50.0 | 53.1 | 7.8 |
| `paper_lutllm_qwen3_1p7b_7linear_actlutfit5_int8_final_fast` | reconstructed final LUT | 8-bit | 32.8 | 37.5 | 45.3 | 59.4 | 50.0 | 57.8 | 9.4 |

Full-layer runs, 16 rows/task, SQuAD skipped:

| Run | Stage | Quantized Linears | LUT Bits | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_lutllm_qwen3_1p7b_all_actlutfit5_int4_fast16` | direct Act LUT | 196 | 16-bit expanded | 25.0 | 62.5 | 50.0 | 43.8 | 62.5 | 50.0 | 6.2 |
| `paper_lutllm_qwen3_1p7b_all_actlutfit5_int4_expanded_final16_fast` | reconstructed final LUT | 196 | 4-bit | 25.0 | 62.5 | 31.2 | 37.5 | 62.5 | 50.0 | 6.2 |

Hardware aggregate for the full-layer reconstructed final LUT:

| Quantized Linears | Compact 4-bit LUT | Weight Codes Packed | Lookups / Token | Act Code Bits / Token | Expanded LUT FP16 Used For PyTorch Eval |
|---:|---:|---:|---:|---:|---:|
| 196 | 1,344.0 MiB | 336.0 MiB | 704,643,072 | 1,548,288 | 86,016.0 MiB |

This experiment did not reproduce the paper's `+ Act. Quant.` or final LUT accuracy. It also shows that 4-bit versus 8-bit final LUT precision is not the dominant issue in this scaffold: the 7-linear INT8 final run only improves a few points relative to INT4 and remains far from the paper. The likely missing pieces remain end-to-end QAT over the whole model, the exact adjustable-gradient STE/fused lookup kernels, GPTVQ rather than local k-means weight VQ, and the paper's evaluation/task-adaptation protocol.

## Commands

Baseline:

```bash
python3 run_paper_eval.py \
  --model-id Qwen/Qwen3-1.7B-Base \
  --output-dir results/paper_baseline_qwen3_1p7b_128 \
  --paper-samples 128
```

Simplified QAT:

```bash
python3 run_lutllm_qat.py \
  --model-id Qwen/Qwen3-1.7B-Base \
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
  --model-id Qwen/Qwen3-1.7B-Base \
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

The `--train-source paper` mode trains activation codebooks on supervised GLUE/SQuAD/MMLU-Pro-style prompt+answer examples instead of WikiText continuation loss. `--output-correction affine` fits a post-hoc per-output affine correction during final LUT conversion; this is not GPTVQ, but it is a useful lightweight approximation for reducing final layer-output error. `--weight-code-reassign-iters 1` enables the experimental output-aware final weight-code reassignment diagnostic. `--weight-center-refine-iters` enables the LS centroid-refinement diagnostic. `--act-train-mode {hard,soft,soft_hard}` and `--act-ste-input-scale` test inferred activation-STE variants.

Direct activation-LUT fitting with reconstructed final LUT:

```bash
python3 run_act_lut_fit.py \
  --model-id Qwen/Qwen3-1.7B-Base \
  --output-dir results/paper_lutllm_qwen3_1p7b_all_actlutfit5_int4_expanded_final16_fast \
  --calib-source paper \
  --task-calib-samples 64 \
  --paper-samples 16 \
  --skip-squad \
  --seq-len 256 \
  --calib-batches 16 \
  --calib-vectors-per-layer 256 \
  --fit-steps 5 \
  --fit-lr 1e-2 \
  --fit-batch-size 128 \
  --kmeans-iters 1 \
  --sample-limit 256 \
  --lut-quant-bits 4 \
  --lut-storage expanded \
  --eval-final-lut
```

The `--lut-storage expanded` flag is for faster PyTorch evaluation only. Hardware estimates still report the compact 4-bit base LUT and packed weight-code sizes.

Prompt-style baseline check:

```bash
python3 run_paper_eval.py \
  --model-id Qwen/Qwen3-1.7B \
  --output-dir results/paper_baseline_qwen3_1p7b_chat_128 \
  --paper-samples 128 \
  --prompt-style chat
```
