# LUT-LLM Paper Reproduction

This file tracks the paper-targeted reproduction path for LUT-LLM, separate from the earlier PQ+LUT feasibility experiments in `RESULTS.md`.

## Target Paper Setup

Paper: LUT-LLM: Efficient Language Model Inference with Memory-based Computations on FPGAs, arXiv v2, 2026-03-22.

Official artifact checked at `LUT-FPGA/LUT-LLM` commit `9ee2259d312f9b1119a398d8ff7703154260a417`. The public artifact contains FPGA/HLS code, a Qwen 3 1.7B hardware model, and latency/resource scripts. It does not provide the full PyTorch QAT/STE training code, fused lookup/reduce training kernels, GPTVQ scripts, or exact evaluation harness used for Table III.

The paper's algorithm setup is:

- Model: Qwen 3 1.7B.
- Public quantization configuration from the arXiv v2 source: `G=512`, vector length `v=2`, activation codebook size `c_a=64`, weight codebook size `c_w=16`, with INT8 quantized lookup tables.
- Activation VQ in this repo: `subdim=2`, `Ka=64` for strict paper-configuration attempts.
- Weight VQ in this repo: `weight_group_size=512`, `Kw=16` for final-LUT attempts.
- Training: KMeans initialization, QAT with STE and custom fused forward/backward kernels.
- Additional training details from the paper text: the authors report continuing training from Qwen 3 1.7B with FineWeb 512-token sequences and then WikiQA for 3 epochs, plus the LUT-DLA-style reconstruction loss ratio `0.1`.
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
- `run_w8a8_quant.py`: RTN/W8A8 fake-quantization runner with dynamic/static activation scales and SmoothQuant-style smoothing.
- `probe_squad_prompts.py`: FP16/BF16 SQuAD v2 prompt probe used to test whether the SQuAD gap is a simple prompt/generation-length issue.
- `pq_lut_lm/paper_eval.py`: prompt-based GLUE/MMLU-Pro log-likelihood scoring and SQuADv2 short generation/F1.
- `pq_lut_lm/activation_quant.py`: trainable activation VQ wrapper with STE, direct activation-LUT modules, and LUT-to-weight reconstruction.
- `pq_lut_lm/w8a8_quant.py`: all-target-linear W8A8 wrapper used to test RTN/SmoothQuant-style activation quantization.

Important limitation: this is not yet a byte-identical reproduction of the paper. The missing pieces are the paper's custom fused QAT kernels, adjustable-gradient STE details, GPTVQ implementation, and exact benchmark harness/prompts. The current code is a transparent PyTorch reproduction scaffold that runs the same model family and datasets but not the undisclosed training/eval stack.

Current implementation additions for the first-stage reproduction:

- `--train-include-squad` separates train/calibration SQuAD inclusion from SQuAD evaluation skipping.
- `SCORE_COMPLETION_BATCH_SIZE=1` allows memory-safe GLUE/MMLU-Pro completion scoring for activation-VQ models.
- `--reconstruction-loss-ratio` adds a dense-output reconstruction MSE term during STE QAT. This approximates the paper's stated reconstruction-loss ratio `0.1`.
- `--task-loss-ratio` allows reconstruction-only ablations without changing the default task-loss behavior.
- `--weight-decay` now records and controls AdamW weight decay explicitly. Dense+centroid QAT can be run with `0.0` or with the earlier implicit PyTorch default `0.01`.
- `--train-source lutllm_paper` approximates the paper's FineWeb 512-token pretrain plus WikiQA finetune data mix.
- Supervised prompt/completion encoding now preserves completion labels under left truncation. This matters for long SQuAD/WikiQA prompts: the old path could mask nearly all answer tokens after truncation, weakening task-adaptation and QAT supervision.

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

SQuAD v2 FP16/BF16 prompt probe on `Qwen/Qwen3-1.7B-Base`, 64 validation examples:

| Prompt Probe | Best F1 | Gap vs Paper FP16 SQuAD |
|---|---:|---:|
| Current instruction prompt, max 24 new tokens | 31.86 | -40.94 |
| Best simple probe (`short_span`, max 16/24/32) | 34.41 | -38.39 |
| Best simple probe (`qa_only`, max 16/24/32) | 34.40 | -38.40 |

Interpretation: the SQuAD gap is not explained by a short generation length or a simple prompt variant. The public checkpoint/protocol is far below the paper FP16 SQuAD row before any quantization is applied.

Layerwise trained activation-LUT diagnostic:

| Run | Samples | GLUE Avg | Gap vs Paper Act GLUE | MMLU-Pro | Gap vs Paper Act MMLU | SQuADv2 F1 | Gap vs Paper Act SQuAD | WikiText PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `actlutfit_centers_all196_ca64_fit50_cv2048_lr3e3_clr3e4_temp05_squad32_ppl64` | 32/task | 51.56 | -35.64 | 6.25 | -25.55 | 28.12 | -42.17 | 98,629.29 |

This run fits expanded activation-LUT values and activation centers locally for all 196 target linears. It is a closer diagnostic for the paper phrase "trained lookup tables" than fixed-center LUT fitting, but it performs substantially worse than the STE activation-QAT path. The missing detail is therefore not just local LUT-value training; the effective version likely depends on end-to-end fused QAT, the exact checkpoint/data, and/or GPTVQ reconstruction.

Latest all-196 diagnostic gap:

| Stage | GLUE Avg | Paper Target | Gap | MMLU-Pro | Paper Target | Gap | WikiText PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-label `Ka=128`, seq512, reconstruction 0.1 + dense LR `1e-7`, 1500 steps | 79.43 | 87.20 | -7.77 | 7.81 | 31.80 | -23.99 | 66.77 |
| Fixed-label `Ka=256`, centers-only, task 0.3 + reconstruction 1.0, LR `1e-5`, 1500 steps | 78.39 | 87.20 | -8.81 | 12.50 | 31.80 | -19.30 | 64.89 |
| Fixed-label `Ka=256`, centers-only, reconstruction 0.1, LR `1e-4`, 2000 steps | 78.12 | 87.20 | -9.08 | 12.50 | 31.80 | -19.30 | 38.71 |
| `Ka=256`, dense+centroids, `weight_decay=0.01`, task 0.1 + reconstruction 1.0, 2000 steps | 77.60 | 87.20 | -9.60 | 15.62 | 31.80 | -16.18 | 64.78 |
| Fixed-label `Ka=64`, seq512, reconstruction 0.1 + dense LR `1e-7`, 2000 steps | 76.56 | 87.20 | -10.64 | 9.38 | 31.80 | -22.43 | 92.80 |
| Fixed-label `Ka=256`, centers-only, task 0.1 + reconstruction 1.0, LR `1e-5`, 1500 steps | 76.04 | 87.20 | -11.16 | 21.88 | 31.80 | -9.93 | 66.96 |
| Fixed-label `Ka=256`, reconstruction 0.1 + dense LR `1e-6`, 1000-step control | 74.22 | 87.20 | -12.98 | 15.62 | 31.80 | -16.18 | 89.38 |
| Fixed-label `Ka=64`, 8192 calib vectors/layer, KMeans 10, centers-only, 2000 steps | 73.70 | 87.20 | -13.50 | 10.94 | 31.80 | -20.86 | 144.20 |
| Strict public paper config `c_a=64`, dense+centroids, `weight_decay=0`, task 1.0 + reconstruction 0.1, 3000 steps | 72.66 | 87.20 | -14.54 | 9.38 | 31.80 | -22.43 | 81.44 |
| Fixed-label `Ka=128`, dense LR `1e-7`, task 0.3 + reconstruction 1.0, 1500 steps | 72.92 | 87.20 | -14.28 | 12.50 | 31.80 | -19.30 | 83.41 |
| Fixed-label `Ka=64`, centers-only, reconstruction 0.1, LR `5e-5`, 4000 steps | 72.66 | 87.20 | -14.54 | 10.94 | 31.80 | -20.86 | 93.14 |
| `Ka=256`, reconstruction 0.1 + dense weights, 1000 steps | 71.09 | 87.20 | -16.11 | 23.44 | 31.80 | -8.36 | 107.73 |
| Strict public paper config `c_a=64`, dense+centroids, `weight_decay=0`, task 0.3 + reconstruction 1.0, 3000 steps | 69.01 | 87.20 | -18.19 | 14.06 | 31.80 | -17.74 | 92.32 |
| Fixed-label `Ka=64`, reconstruction 1.0, centers-only LR `3e-5`, 2000 steps | 68.49 | 87.20 | -18.71 | 7.81 | 31.80 | -23.99 | 157.09 |
| Fixed-label `Ka=64`, centers-only, task 0.1 + reconstruction 1.0, LR `1e-5`, 2000 steps | 67.45 | 87.20 | -19.75 | 15.62 | 31.80 | -16.18 | 93.21 |
| Fixed-label `Ka=64`, reconstruction-only, centers-only LR `3e-5`, 1500 steps | 61.46 | 87.20 | -25.74 | 7.81 | 31.80 | -23.99 | 152.58 |
| FineWeb/WikiQA approximation, `Ka=64`, centers-only, reconstruction 0.1, 3000 steps | 53.39 | 87.20 | -33.81 | 9.38 | 31.80 | -22.43 | 397.58 |
| Paper-like first-step FP16, SQuAD included | 83.33 | 88.80 | -5.47 | 39.06 | 33.10 | +5.96 | 16.45 |
| Paper-like centers-only STE Act Quant, SQuAD included | 64.06 | 87.20 | -23.14 | 12.50 | 31.80 | -19.30 | 207.19 |
| Paper-like reconstruction 0.1 + dense-weight STE Act Quant, SQuAD included | 71.88 | 87.20 | -15.33 | 18.75 | 31.80 | -13.05 | 242.19 |
| FineWeb/WikiQA reconstruction 0.1 + dense-weight STE Act Quant | 54.43 | 87.20 | -32.77 | 12.50 | 31.80 | -19.30 | 109.93 |
| FP16 baseline | 82.88 | 88.80 | -5.92 | 29.69 | 33.10 | -3.41 | 16.45 |
| Act Quant, `Ka=64`, no QAT | 61.07 | 87.20 | -26.13 | 6.25 | 31.80 | -25.55 | 332.60 |
| centers-only STE Act Quant, 1000 steps | 70.96 | 87.20 | -16.24 | 7.81 | 31.80 | -23.99 | 335.62 |
| `subdim=4, Ka=64` centers-only STE Act Quant, 1000 steps | 46.09 | 87.20 | -41.11 | 7.81 | 31.80 | -23.99 | 19,947.91 |
| `subdim=2, Ka=128` centers-only STE Act Quant, 1000 steps | 68.75 | 87.20 | -18.45 | 7.81 | 31.80 | -23.99 | 217.62 |

July 4 continuation summary, including SQuAD gaps to the paper's first-stage target:

| Run | Stage | GLUE Avg | Gap vs Paper Act GLUE | MMLU-Pro | Gap vs Paper Act MMLU | SQuADv2 F1 | Gap vs Paper Act SQuAD | WikiText PPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Paper `+ Act. Quant.` target | target | 87.20 | 0.00 | 31.80 | 0.00 | 70.30 | 0.00 | - |
| `lutllm_base_instruction_g8_all196_fixedlabels_seq512_ka128_recon01_dense1e7_steqat1500_actonly_squad64_ppl64` | fixed-label, `Ka=128`, recon 0.1 + dense LR `1e-7`, 1500 steps | 79.43 | -7.77 | 7.81 | -23.99 | 31.96 | -38.34 | 66.77 |
| `lutllm_base_instruction_g8_all196_fixedlabels_seq512_ka256_recon1_task03_centersonly_lr1e5_steqat1500_squad64_ppl64` | fixed-label, `Ka=256`, centers-only, task 0.3 + recon 1.0, LR `1e-5` | 78.39 | -8.81 | 12.50 | -19.30 | 27.27 | -43.03 | 64.89 |
| `lutllm_base_instruction_g8_all196_fixedlabels_seq512_ka256_centersonly_lr1e4_recon01_steqat2000_squad64_ppl64` | fixed-label, `Ka=256`, centers-only, recon 0.1, LR `1e-4` | 78.12 | -9.08 | 12.50 | -19.30 | 33.76 | -36.54 | 38.71 |
| `lutllm_bigcode_all196_ka256_dense_wd001_lr1e5_dense1e7_recon1_task01_steqat2000_squad64_ppl64` | `Ka=256`, dense+centroids, `weight_decay=0.01`, task 0.1 + recon 1.0 | 77.60 | -9.60 | 15.62 | -16.18 | 31.46 | -38.84 | 64.78 |
| `lutllm_base_instruction_g8_all196_fixedlabels_seq512_ka64_recon01_dense1e7_steqat2000_actonly_squad64_ppl64` | fixed-label, `Ka=64`, recon 0.1 + dense LR `1e-7`, 2000 steps | 76.56 | -10.64 | 9.38 | -22.43 | 35.52 | -34.78 | 92.80 |
| `lutllm_base_instruction_g8_all196_fixedlabels_seq512_ka256_recon1_task01_centersonly_lr1e5_steqat1500_squad64_ppl64` | fixed-label, `Ka=256`, centers-only, task 0.1 + recon 1.0, LR `1e-5` | 76.04 | -11.16 | 21.88 | -9.93 | 31.98 | -38.32 | 66.96 |
| `lutllm_base_instruction_g8_all196_fixedlabels_ka256_recon01_dense_steqat1000_actonly_squad64_ppl64_control` | fixed-label control, `Ka=256`, recon 0.1 + dense LR `1e-6`, 1000 steps | 74.22 | -12.98 | 15.62 | -16.18 | 28.52 | -41.78 | 89.38 |
| `lutllm_base_instruction_g8_all196_ka64_calib8192_k10_recon01_centersonly_steqat2000_actonly_squad64_ppl64_fixedlabels` | fixed-label, 8192 calib vectors/layer, KMeans 10, centers-only | 73.70 | -13.50 | 10.94 | -20.86 | 32.75 | -37.55 | 144.20 |
| `lutllm_paperexact_all196_ca64_dense_wd0_lr3e5_dense3e7_recon01_task1_steqat3000_squad64_ppl64` | strict public paper config `c_a=64`, dense+centroids, `weight_decay=0`, task 1.0 + recon 0.1 | 72.66 | -14.54 | 9.38 | -22.43 | 33.39 | -36.91 | 81.44 |
| `lutllm_base_instruction_g8_all196_fixedlabels_seq512_ka128_balancedrecon1_task03_dense1e7_steqat1500_squad64_ppl64` | fixed-label, `Ka=128`, dense LR `1e-7`, task 0.3 + recon 1.0 | 72.92 | -14.28 | 12.50 | -19.30 | 23.59 | -46.71 | 83.41 |
| `lutllm_base_instruction_g8_all196_fixedlabels_seq512_ka64_centersonly_lr5e5_recon01_steqat4000_squad64_ppl64` | fixed-label, `Ka=64`, centers-only, recon 0.1, LR `5e-5`, 4000 steps | 72.66 | -14.54 | 10.94 | -20.86 | 35.00 | -35.30 | 93.14 |
| `lutllm_base_instruction_g8_all196_ka256_recon01_dense_steqat1000_actonly_squad64_ppl64` | `Ka=256`, recon 0.1 + dense LR `1e-6`, 1000 steps | 71.09 | -16.11 | 23.44 | -8.36 | 33.18 | -37.12 | 107.73 |
| `lutllm_paperexact_all196_ca64_dense_wd0_lr1e5_dense1e7_recon1_task03_steqat3000_squad64_ppl64` | strict public paper config `c_a=64`, dense+centroids, `weight_decay=0`, task 0.3 + recon 1.0 | 69.01 | -18.19 | 14.06 | -17.74 | 5.94 | -64.36 | 92.32 |
| `lutllm_base_instruction_g8_all196_fixedlabels_ka64_calib8192_k10_recon1_centersonly_lr3e5_steqat2000_actonly_squad64_ppl64` | fixed-label, recon 1.0, centers-only LR `3e-5` | 68.49 | -18.71 | 7.81 | -23.99 | 33.90 | -36.40 | 157.09 |
| `lutllm_base_instruction_g8_all196_fixedlabels_seq512_ka64_recon1_task01_centersonly_lr1e5_steqat2000_squad64_ppl64` | fixed-label, `Ka=64`, centers-only, task 0.1 + recon 1.0, LR `1e-5` | 67.45 | -19.75 | 15.62 | -16.18 | 4.05 | -66.25 | 93.21 |
| `lutllm_base_instruction_g8_all196_cheb_softhard_t05_recon01_dense_steqat1000_actonly_squad64_ppl64_retry` | soft-hard STE temp 0.5, recon 0.1 + dense | 64.84 | -22.36 | 12.50 | -19.30 | 37.80 | -32.50 | 297.71 |
| `lutllm_base_instruction_g8_all196_fixedlabels_ka64_calib8192_k10_recononly_centersonly_lr3e5_steqat1500_actonly_squad64_ppl64` | fixed-label, reconstruction-only, centers-only LR `3e-5` | 61.46 | -25.74 | 7.81 | -23.99 | 3.51 | -66.79 | 152.58 |
| `lutllm_lutllmpaper_all196_seq512_ka64_centersonly_lr1e4_recon01_steqat3000_squad64_ppl64_bigdata` | FineWeb/WikiQA approximation, `Ka=64`, centers-only, recon 0.1 | 53.39 | -33.81 | 9.38 | -22.43 | 6.99 | -63.31 | 397.58 |

Interpretation: the fixed-label truncation repair plus dense QAT improves GLUE to `79.43` with `Ka=128`, the best all-196 first-step GLUE result so far. Increasing the codebook to `Ka=256` improves PPL substantially (`38.71`) and a conservative `Ka=256` loss mix recovers MMLU-Pro to `21.88`, but neither setting recovers SQuAD or reaches the paper's first-step GLUE/MMLU targets. The newest `Ka=256` dense+centroid run does not improve that tradeoff, and the two newest strict public-paper-configuration `c_a=64` dense+centroid runs also fail to close the gap. The larger FineWeb/WikiQA approximation damages all downstream metrics, so the missing paper data/checkpoint recipe is not reproduced by the small public-data proxy. The result is still not a reproduction of Table III; it is a constrained reverse-engineering attempt around missing QAT/evaluation details.

Task-adapted checkpoint diagnostic:

| Run | Stage | GLUE Avg | Gap vs Paper Target | MMLU-Pro | Gap vs Paper Target | SQuADv2 F1 | Gap vs Paper Target | WikiText PPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `eval_taskadapt_fixedlabels_qwen3_1p7b_base_instruction_paperall_1000_g8_squad64` | FP16 after fixed-label 1000-update task adaptation | 84.64 | -4.16 vs FP16 | 37.50 | +4.40 vs FP16 | 32.81 | -39.99 vs FP16 | - |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_centersonly_lr5e6_recon01_task1_steqat3000_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, LR `5e-6`, task 1.0 + recon 0.1, 3000 steps | 83.59 | -3.61 vs Act | 12.50 | -19.30 vs Act | 31.63 | -38.67 vs Act | 56.87 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_centersonly_lr5e6_recon1_task03_steqat3000_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, LR `5e-6`, task 0.3 + recon 1.0, 3000 steps | 81.77 | -5.43 vs Act | 14.06 | -17.74 vs Act | 30.43 | -39.87 vs Act | 68.36 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_dense_lr5e6_dense1e8_recon01_task1_steqat3000_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, center LR `5e-6`, dense LR `1e-8`, task 1.0 + recon 0.1, 3000 steps | 81.77 | -5.43 vs Act | 14.06 | -17.74 vs Act | 33.33 | -36.97 vs Act | 56.53 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_dense_lr1e8_recon1_task03_steqat2000_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, dense LR `1e-8`, task 0.3 + recon 1.0, 2000 steps | 80.73 | -6.47 vs Act | 20.31 | -11.49 vs Act | 30.13 | -40.17 vs Act | 46.17 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka512_km5_sample4096_centersonly_lr1e5_recon1_task01_steqat2000_squad64_ppl64` | `Ka=512`, KMeans 5/sample 4096, task 0.1 + recon 1.0, 2000 steps | 79.95 | -7.25 vs Act | 15.62 | -16.18 vs Act | 32.17 | -38.13 vs Act | 35.06 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_dense_lr1e7_recon1_task03_steqat2000_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, dense LR `1e-7`, task 0.3 + recon 1.0, 2000 steps | 79.95 | -7.25 vs Act | 10.94 | -20.86 vs Act | 32.02 | -38.28 vs Act | 81.91 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_centersonly_lr1e5_recon1_task03_steqat2000_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, task 0.3 + recon 1.0, 2000 steps | 79.69 | -7.51 vs Act | 12.50 | -19.30 vs Act | 29.85 | -40.45 vs Act | 56.05 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_dense_lr5e6_dense1e8_recon1_task03_steqat3000_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, center LR `5e-6`, dense LR `1e-8`, task 0.3 + recon 1.0, 3000 steps | 79.69 | -7.51 vs Act | 10.94 | -20.86 vs Act | 33.00 | -37.30 vs Act | 68.30 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka512_km5_sample4096_centersonly_lr5e6_recon1_task03_steqat3000_squad64_ppl64` | `Ka=512`, KMeans 5/sample 4096, LR `5e-6`, task 0.3 + recon 1.0, 3000 steps | 79.69 | -7.51 vs Act | 10.94 | -20.86 vs Act | 33.80 | -36.50 vs Act | 51.59 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka512_km5_sample4096_centersonly_lr5e6_recon1_task01_steqat3000_squad64_ppl64` | `Ka=512`, KMeans 5/sample 4096, LR `5e-6`, task 0.1 + recon 1.0, 3000 steps | 78.65 | -8.55 vs Act | 15.62 | -16.18 vs Act | 33.59 | -36.71 vs Act | 52.23 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_samples2048_centersonly_lr5e6_recon01_task1_steqat3000_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, 2048 train samples/task, LR `5e-6`, task 1.0 + recon 0.1, 3000 steps | 78.39 | -8.81 vs Act | 14.06 | -17.74 vs Act | 34.56 | -35.74 vs Act | 60.70 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_centersonly_lr1e5_recon1_task03_steqat1500_repeat_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, task 0.3 + recon 1.0, 1500 steps | 77.86 | -9.34 vs Act | 12.50 | -19.30 vs Act | 30.13 | -40.17 vs Act | 66.16 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka512_km5_sample4096_centersonly_lr1e5_recon1_task03_steqat2000_squad64_ppl64` | `Ka=512`, KMeans 5/sample 4096, task 0.3 + recon 1.0, 2000 steps | 77.86 | -9.34 vs Act | 18.75 | -13.05 vs Act | 35.63 | -34.67 vs Act | 52.57 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_centersonly_lr1e5_recon1_task01_steqat2000_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, task 0.1 + recon 1.0, 2000 steps | 77.60 | -9.60 vs Act | 14.06 | -17.74 vs Act | 31.12 | -39.18 vs Act | 67.58 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_task03_recon1_steqat1500_squad64_ppl64` | `Ka=256`, centers-only, task 0.3 + recon 1.0 | 77.34 | -9.86 vs Act | 25.00 | -6.80 vs Act | 33.73 | -36.57 vs Act | 54.94 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka128_dense_wd001_lr1e4_dense1e7_recon01_task1_steqat1500_squad64_ppl64` | `Ka=128`, dense+centroids, task 1.0 + recon 0.1 | 77.08 | -10.12 vs Act | 17.19 | -14.61 vs Act | 28.87 | -41.43 vs Act | 67.32 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_centersonly_lr3e6_recon01_task1_steqat4000_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, LR `3e-6`, task 1.0 + recon 0.1, 4000 steps | 76.56 | -10.64 vs Act | 10.94 | -20.86 vs Act | 33.21 | -37.09 vs Act | 55.89 |
| `lutllm_taskadapt_fixedlabels1000_all196_ca64_km3_sample2048_centersonly_lr5e5_task1_steqat2000_squad64_ppl64` | strict `c_a=64`, KMeans 3, sample limit 2048, centers-only task loss | 74.22 | -12.98 vs Act | 10.94 | -20.86 vs Act | 24.11 | -46.19 vs Act | 130.17 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km2_sample1024_centersonly_lr5e5_task03_recon1_steqat2000_squad64_ppl64` | `Ka=256`, KMeans 2, sample limit 1024, task 0.3 + recon 1.0 | 71.61 | -15.59 vs Act | 10.94 | -20.86 vs Act | 33.73 | -36.57 vs Act | 89.26 |
| `lutllm_taskadapt_fixedlabels1000_all196_ca64_km5_sample4096_centersonly_lr1e5_recon1_task03_steqat3000_squad64_ppl64` | strict `c_a=64`, KMeans 5/sample 4096, task 0.3 + recon 1.0, 3000 steps | 70.83 | -16.37 vs Act | 7.81 | -23.99 vs Act | 4.57 | -65.73 vs Act | 105.57 |
| `lutllm_taskadapt_fixedlabels1000_all196_subdim1_ka256_centersonly_lr1e5_recon1_task03_steqat1500_squad64_ppl64` | `subdim=1`, `Ka=256`, centers-only, task 0.3 + recon 1.0 | 66.93 | -20.27 vs Act | 14.06 | -17.74 vs Act | 30.65 | -39.65 vs Act | 77.18 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_lr1e4_recon01_task1_steqat1500_squad64_ppl64` | `Ka=256`, centers-only, task 1.0 + recon 0.1, LR `1e-4` | 66.15 | -21.05 vs Act | 12.50 | -19.30 vs Act | 32.46 | -37.84 vs Act | 47.25 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_centersonly_lr1e5_recon1_task03_steqat3000_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, task 0.3 + recon 1.0, 3000 steps | 65.36 | -21.84 vs Act | 9.38 | -22.43 vs Act | 31.61 | -38.69 vs Act | 47.75 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_lr1e5_recon1_task03_steqat1000_squad64_ppl64` | `Ka=256`, default KMeans 1/sample 256, task 0.3 + recon 1.0, 1000 steps | 64.58 | -22.62 vs Act | 10.94 | -20.86 vs Act | 17.80 | -52.50 vs Act | 261.02 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_centersonly_lr1e5_recon1_task01_steqat3000_squad64_ppl64` | `Ka=256`, KMeans 5/sample 4096, task 0.1 + recon 1.0, 3000 steps | 64.32 | -22.88 vs Act | 12.50 | -19.30 vs Act | 31.57 | -38.73 vs Act | 39.26 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_initonly_squad64_ppl64` | `Ka=256`, KMeans init only, no QAT steps | 63.80 | -23.40 vs Act | 6.25 | -25.55 vs Act | 1.56 | -68.74 vs Act | 220.15 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_l2_centersonly_lr1e5_recon1_task03_steqat2000_squad64_ppl64` | `Ka=256`, L2 assignment, task 0.3 + recon 1.0 | 60.42 | -26.78 vs Act | 10.94 | -20.86 vs Act | 28.57 | -41.73 vs Act | 173.79 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_dense_wd0_lr1e5_dense1e8_recon1_task03_steqat2000_squad64_ppl64` | `Ka=256`, dense+centroids, dense LR `1e-8`, task 0.3 + recon 1.0 | 60.42 | -26.78 vs Act | 7.81 | -23.99 vs Act | 26.49 | -43.81 vs Act | 223.25 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_lr3e5_recon1_task03_steqat3000_squad64_ppl64` | `Ka=256`, centers-only, task 0.3 + recon 1.0, LR `3e-5`, 3000 steps | 58.85 | -28.35 vs Act | 7.81 | -23.99 vs Act | 7.64 | -62.66 vs Act | 229.21 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_recononly_lr1e5_steqat2000_squad64_ppl64` | `Ka=256`, centers-only, reconstruction-only | 57.29 | -29.91 vs Act | 6.25 | -25.55 vs Act | 24.13 | -46.17 vs Act | 248.89 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_steinput01_lr1e5_recon1_task03_steqat2000_squad64_ppl64` | `Ka=256`, STE input gradient scale 0.1 | 56.77 | -30.43 vs Act | 10.94 | -20.86 vs Act | 1.56 | -68.74 vs Act | 265.69 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_lr1e5_recon1_task03_steqat2000_squad64_ppl64_repeat` | `Ka=256`, default KMeans 1/sample 256, task 0.3 + recon 1.0, 2000 steps | 56.77 | -30.43 vs Act | 6.25 | -25.55 vs Act | 27.79 | -42.51 vs Act | 178.75 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_lr1e5_recon1_task03_steqat500_squad64_ppl64` | `Ka=256`, default KMeans 1/sample 256, task 0.3 + recon 1.0, 500 steps | 56.77 | -30.43 vs Act | 6.25 | -25.55 vs Act | 21.56 | -48.74 vs Act | 187.80 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_lr5e6_recon1_task03_steqat4000_squad64_ppl64` | `Ka=256`, centers-only, task 0.3 + recon 1.0, LR `5e-6`, 4000 steps | 58.59 | -28.61 vs Act | 9.38 | -22.43 vs Act | 24.51 | -45.79 vs Act | 253.49 |

Interpretation: fixed-label task adaptation improves the FP16 GLUE/MMLU starting point and reduces the baseline-alignment problem for those two metrics, but it still leaves SQuAD far below the paper FP16 row. Quantizing all 196 target linears from that checkpoint now drops GLUE by about 1 point in the best task-adapted run. Init-only activation VQ is much worse, so the simplified QAT helps. The best branch so far uses strong KMeans initialization, lower center LR (`5e-6`), and the paper-like task/reconstruction ratio 1.0/0.1, reaching GLUE 83.59, still 3.61 points below the paper `+ Act. Quant.` row. July 5-7 follow-up runs did not improve this: KMeans10, 8192 calibration, SQuAD `No Answer` label repair, SQuAD-weighted baseline adaptation, a lower-LR conservation run, and inferred adjustable STE gradients all remained below the best row. Larger `Ka=512` codebooks improve SQuAD/MMLU in some old-checkpoint runs but do not recover GLUE; a non-chunked `Ka=512` run from the improved checkpoint OOMed during long-context SQuAD eval, while a chunked 32-sample diagnostic completed and still left GLUE `-7.51` below the paper target. Strict `c_a=64`, L2 assignment, reconstruction-only training, smaller STE input gradients, `subdim=1`, higher LR, lower LR `3e-6`, and longer 4000-step training also fail. Full-layer `soft_hard` assignment with `Ka=256` and `Ka=64` OOMs on a 140GB H200 in this unfused PyTorch implementation.

July 5 continuation, all runs quantize all 196 transformer-block target linears:

| Run | Stage | GLUE Avg | Gap vs Paper Target | MMLU-Pro | Gap vs Paper Target | SQuADv2 F1 | Gap vs Paper Target | WikiText PPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `eval_taskadapt_noanswer_squadrepeat4_qwen3_1p7b_base_instruction_paperall_4000_lr5e7_g8_squad64` | FP16, SQuAD `No Answer` + SQuAD-repeat task adaptation | 85.16 | -3.64 vs FP16 | 37.50 | +4.40 vs FP16 | 39.06 | -33.74 vs FP16 | - |
| `eval_taskadapt_noanswer_continue_qwen3_1p7b_base_instruction_paperall_3000_lr1e6_g8_squad64` | FP16, continued No Answer task adaptation | 84.64 | -4.16 vs FP16 | 37.50 | +4.40 vs FP16 | 39.06 | -33.74 vs FP16 | - |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample8192_cv8192_centersonly_lr5e6_recon01_task1_steqat3000_squad64_ppl64` | old checkpoint, KMeans 5/sample 8192/calib 8192 | 79.95 | -7.25 vs Act | 9.38 | -22.43 vs Act | 31.96 | -38.34 vs Act | 42.31 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km10_sample4096_centersonly_lr5e6_recon01_task1_steqat3000_squad64_ppl64` | old checkpoint, KMeans 10/sample 4096 | 79.69 | -7.51 vs Act | 14.06 | -17.74 vs Act | 33.28 | -37.02 vs Act | 45.33 |
| `lutllm_taskadapt_noanswer_trainsquad_all196_ka256_km5_sample4096_centersonly_lr5e6_recon01_task1_steqat3000_squad64_ppl64` | old checkpoint, No Answer fix + SQuAD in QAT | 79.17 | -8.03 vs Act | 6.25 | -25.55 vs Act | 31.44 | -38.86 vs Act | 46.27 |
| `lutllm_taskadapt_noanswer_squadrepeat4_4000_all196_ka256_km5_sample4096_centersonly_lr1e6_recon01_task1_steqat1500_squad64_ppl64` | SQuAD-repeat checkpoint, LR `1e-6`, 1500 steps | 78.91 | -8.29 vs Act | 14.06 | -17.74 vs Act | 31.94 | -38.36 vs Act | 47.77 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_seed789_centersonly_lr5e6_recon01_task1_steqat3000_squad64_ppl64` | old checkpoint, seed 789 | 78.65 | -8.55 vs Act | 21.88 | -9.93 vs Act | 32.30 | -38.00 vs Act | 64.90 |
| `lutllm_taskadapt_noanswer_continue3000_all196_ka256_km5_sample4096_centersonly_lr5e6_recon01_task1_steqat3000_squad64_ppl64` | 3000-update No Answer checkpoint, `Ka=256` | 78.65 | -8.55 vs Act | 10.94 | -20.86 vs Act | 34.90 | -35.40 vs Act | 38.95 |
| `lutllm_taskadapt_noanswer_continue3000_all196_ka512_km5_sample4096_chunk16m_centersonly_lr1e5_recon1_task03_steqat2000_squad32_ppl64` | 3000-update No Answer checkpoint, `Ka=512`, chunked distance eval, 32/task | 79.69 | -7.51 vs Act | 25.00 | -6.80 vs Act | 38.54 | -31.76 vs Act | 51.60 |
| `lutllm_taskadapt_noanswer_squadrepeat4_4000_all196_ka256_km5_sample4096_centersonly_lr5e6_recon01_task1_steqat3000_squad64_ppl64` | SQuAD-repeat checkpoint, LR `5e-6`, 3000 steps | 78.39 | -8.81 vs Act | 4.69 | -27.11 vs Act | 32.17 | -38.13 vs Act | 39.20 |
| `lutllm_taskadapt_adjgrad_in05to1_center025to1_all196_ka256_km5_sample4096_centersonly_lr5e6_recon01_task1_steqat3000_squad64_ppl64` | inferred adjustable STE gradient schedule | 77.08 | -10.12 vs Act | 14.06 | -17.74 vs Act | 30.39 | -39.91 vs Act | 47.31 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km5_sample4096_seed456_centersonly_lr5e6_recon01_task1_steqat3000_squad64_ppl64` | old checkpoint, seed 456 | 75.52 | -11.68 vs Act | 10.94 | -20.86 vs Act | 34.06 | -36.24 vs Act | 79.33 |
| `lutllm_taskadapt_noanswer_continue3000_all196_ka512_km5_sample4096_centersonly_lr1e5_recon1_task03_steqat2000_squad64_ppl64` | 3000-update No Answer checkpoint, `Ka=512` | OOM | SQuAD eval OOM | OOM | SQuAD eval OOM | OOM | SQuAD eval OOM | - |

Hardware note: for `Ka=256`, all-196 first-stage activation VQ stores 132,120,576 activation-center values and implies 180,388,626,432 expanded FP16 Act-LUT entries (344,064 MiB), with 704,643,072 lookups/token. `Ka=512` doubles centers and expanded LUT entries (688,128 MiB FP16 equivalent) while lookups/token stay unchanged. The new chunked activation-distance path limits the temporary `[tokens, M, Ka]` distance tensor and lets `Ka=512` finish 32/task SQuAD/MMLU eval; it is slower, but closer to the kind of fused/chunked implementation the paper likely used.

### Latest First-Step Act. Quant. Attempt

Runs:

- `lutllm_base_instruction_g8_all196_paperlike_squad_ka64_steqat1000_actonly_ppl64_chunk1`
- `lutllm_base_instruction_g8_all196_paperlike_squad_ka64_recon01_dense_steqat1000_actonly_ppl64_chunk1`
- `lutllm_base_instruction_g8_all196_lutllmpaperdata_ka64_recon01_dense_steqat1000_actonly_ppl64_chunk1`

Purpose: reproduce the paper's first LUT-LLM stage, `+ Act. Quant.`, as directly as the public information allows. These runs use `Qwen/Qwen3-1.7B-Base`, all 196 transformer-block target linears, `subdim=2`, `Ka=64`, Chebyshev assignment, 1000 STE steps, instruction prompts, 8-shot GLUE, 0-shot MMLU-Pro, 64 rows/task, and SQuAD included. The first two train on the repo's paper-task supervised mix. The third switches to the new `lutllm_paper` training source, a small FineWeb/WikiQA approximation of the paper's described training data.

Quality:

| Stage | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | GLUE Avg | SQuADv2 F1 | MMLU-Pro | WikiText PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper FP16 target | 87.6 | 86.5 | 92.9 | 91.2 | 80.9 | 93.7 | 88.80 | 72.8 | 33.1 | - |
| Same-run FP16 | 84.4 | 73.4 | 81.2 | 87.5 | 76.6 | 96.9 | 83.33 | 31.9 | 39.1 | 16.45 |
| Paper `+ Act. Quant.` target | 87.0 | 84.1 | 91.9 | 90.7 | 78.3 | 91.2 | 87.20 | 70.3 | 31.8 | - |
| Same-run `+ Act. Quant.` | 57.8 | 71.9 | 60.9 | 51.6 | 53.1 | 89.1 | 64.06 | 25.7 | 12.5 | 207.19 |
| Same-run `+ Act. Quant.` with reconstruction 0.1 + dense weights | 54.7 | 75.0 | 70.3 | 68.8 | 75.0 | 87.5 | 71.88 | 32.7 | 18.8 | 242.19 |
| FineWeb/WikiQA `+ Act. Quant.` with reconstruction 0.1 + dense weights | 32.8 | 67.2 | 57.8 | 34.4 | 50.0 | 84.4 | 54.43 | 3.4 | 12.5 | 109.93 |

Gap:

| Stage | GLUE Gap | SQuADv2 Gap | MMLU-Pro Gap |
|---|---:|---:|---:|
| Same-run FP16 vs paper FP16 | -5.47 | -40.94 | +5.96 |
| Same-run `+ Act. Quant.` vs paper `+ Act. Quant.` | -23.14 | -44.59 | -19.30 |
| Reconstruction 0.1 + dense weights vs paper `+ Act. Quant.` | -15.33 | -37.56 | -13.05 |
| FineWeb/WikiQA reconstruction 0.1 + dense weights vs paper `+ Act. Quant.` | -32.77 | -66.93 | -19.30 |

Hardware scale for this first-stage activation VQ:

| Quantized Linears | Activation Centers | Expanded Act-LUT FP16 | Expanded Entries | Lookups / Token | Act Code Bits / Token | Centroid Distance Vectors / Token |
|---:|---:|---:|---:|---:|---:|---:|
| 196 | 33,030,144 | 86,016.0 MiB | 45,097,156,608 | 704,643,072 | 1,548,288 | 16,515,072 |

Interpretation: adding SQuAD to the formal all-196 first-step run did not recover the paper row. Adding the paper's stated reconstruction-loss idea helps GLUE, SQuAD, and MMLU-Pro on the task-supervised mix, but the result is still far from the paper and worsens WikiText PPL. Switching to the small FineWeb/WikiQA approximation improves PPL relative to the task-supervised reconstruction run but damages the downstream benchmark metrics. The current public-checkpoint FP16 SQuAD baseline is itself far below the paper. This points to the missing paper training stack: long FineWeb+WikiQA QAT, the exact adjustable-gradient STE/fused lookup kernels, trained lookup-table reconstruction, and the customized checkpoint/evaluation harness.

### Simple RTN Weight-Only Baselines

These runs start from the simplest quantization path before adding LUT-LLM-specific activation VQ or final lookup tables. They use `Qwen/Qwen3-1.7B-Base`, instruction prompts, 8-shot GLUE, 0-shot MMLU-Pro, 256 rows/task, SQuAD skipped, and 4096-token WikiText PPL. Every RTN row quantizes all 196 transformer-block target linears matching `q/k/v/o/gate/up/down_proj`; `lm_head` is excluded. The PyTorch module stores dequantized weights and runs dense linears, so this is an accuracy sanity check, not an optimized INT8 kernel benchmark.

Quality:

| Run | Stage | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | GLUE Avg | MMLU-Pro | WikiText PPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper FP16 | target | 87.6 | 86.5 | 92.9 | 91.2 | 80.9 | 93.7 | 88.80 | 33.10 | - |
| Paper RTN INT8 | target | 86.7 | 80.2 | 88.0 | 89.3 | 70.4 | 87.4 | 83.67 | 23.60 | - |
| `rtn_int8_perchannel_qwen3_1p7b_base_instruction_g8_all196_ppl256` | FP16 baseline | 81.6 | 74.2 | 82.8 | 84.4 | 79.7 | 94.5 | 82.88 | 29.69 | 16.45 |
| same | RTN INT8 per-channel | 83.6 | 76.2 | 81.6 | 84.4 | 78.1 | 94.5 | 83.07 | 30.08 | 16.59 |
| `rtn_int8_group128_qwen3_1p7b_base_instruction_g8_all196_ppl256` | FP16 baseline | 81.6 | 74.2 | 82.8 | 84.4 | 79.7 | 94.5 | 82.88 | 29.69 | 16.45 |
| same | RTN INT8 group128 | 82.8 | 75.0 | 82.4 | 84.4 | 80.9 | 94.5 | 83.33 | 28.91 | 16.53 |
| `rtn_int8_pertensor_qwen3_1p7b_base_instruction_g8_all196_ppl256` | FP16 baseline | 81.6 | 74.2 | 82.8 | 84.4 | 79.7 | 94.5 | 82.88 | 29.69 | 16.45 |
| same | RTN INT8 per-tensor | 84.0 | 74.6 | 80.1 | 82.0 | 80.1 | 94.1 | 82.49 | 30.08 | 18.06 |

Gap to the paper:

| Stage | GLUE Avg | Paper Target | Gap | Quant Drop vs Same FP16 | MMLU-Pro | Paper Target | Gap | PPL Delta vs Same FP16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Same-run FP16 | 82.88 | Paper FP16 88.80 | -5.92 | - | 29.69 | Paper FP16 33.10 | -3.41 | - |
| RTN INT8 per-channel | 83.07 | Paper RTN 83.67 | -0.60 | +0.19 | 30.08 | Paper RTN 23.60 | +6.48 | +0.14 |
| RTN INT8 group128 | 83.33 | Paper RTN 83.67 | -0.34 | +0.45 | 28.91 | Paper RTN 23.60 | +5.31 | +0.08 |
| RTN INT8 per-tensor | 82.49 | Paper RTN 83.67 | -1.18 | -0.39 | 30.08 | Paper RTN 23.60 | +6.48 | +1.61 |

Hardware and quantization scale:

| Method | Quantized Linears | Codebook / Levels | Packed Weight Payload | Scale Count | FP16 Scale Storage | LUT Lookups / Token | Dense MAC / Token | Weighted Weight MSE |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| RTN INT8 per-channel | 196 | no codebook; 256 integer levels per output-channel scale | 1,344.0 MiB | 573,440 | 1.094 MiB | 0 | 1,409,286,144 | 1.169e-7 |
| RTN INT8 group128 | 196 | no codebook; 256 integer levels per 128-weight group scale | 1,344.0 MiB | 11,010,048 | 21.000 MiB | 0 | 1,409,286,144 | 5.833e-8 |
| RTN INT8 per-tensor | 196 | no codebook; 256 integer levels per tensor scale | 1,344.0 MiB | 196 | 0.000374 MiB | 0 | 1,409,286,144 | 2.360e-6 |

Interpretation: these basic RTN runs do not explain the paper's RTN INT8 row. The same public-checkpoint FP16 baseline is still `-5.92` GLUE points below paper FP16, so this is not a faithful Table III reproduction. More importantly, within the same protocol, weight-only INT8 RTN barely changes accuracy: per-channel and group128 slightly improve GLUE within sampling noise, and per-tensor only drops `0.39` GLUE points. The paper's RTN row drops about `5.13` GLUE points and `9.50` MMLU-Pro points from its FP16 baseline, so their RTN is likely not equivalent to this weight-only dense-dequantized RTN. The remaining candidates are activation quantization, a harsher scaling implementation, a different checkpoint/eval protocol, or some combination of these.

### W8A8 Activation-Quantization Baselines

These runs extend RTN from weight-only to W8A8 fake quantization. They still quantize all 196 transformer-block target linears and use dense PyTorch matmuls after quant-dequant, so the timing is not an INT8-kernel benchmark. Unless noted, calibration uses 32 shuffled paper-style supervised batches with 2048 captured vectors/layer. SmoothQuant-style rows use 8 batches and 1024 vectors/layer to iterate quickly after fixing the smoothing implementation.

Quality:

| Run | Method | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | GLUE Avg | MMLU-Pro | WikiText PPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper RTN INT8 | target | 86.7 | 80.2 | 88.0 | 89.3 | 70.4 | 87.4 | 83.67 | 23.60 | - |
| Paper SmoothQuant | target | 87.0 | 85.3 | 91.7 | 89.6 | 79.1 | 91.2 | 87.32 | 31.70 | - |
| Same-run FP16 | public Base checkpoint | 81.6 | 74.2 | 82.8 | 84.4 | 79.7 | 94.5 | 82.88 | 29.69 | 16.45 |
| `w8a8_dynamic_pertoken_wperchannel_qwen3_1p7b_base_instruction_g8_all196_ppl256` | W8A8 dynamic per-token activation | 81.6 | 72.7 | 80.1 | 83.6 | 79.3 | 94.1 | 81.90 | 29.30 | 17.30 |
| `w8a8_static_perfeature_wperchannel_qwen3_1p7b_base_instruction_g8_all196_ppl256` | W8A8 static per-feature activation | 78.5 | 73.0 | 80.9 | 80.9 | 78.1 | 89.5 | 80.14 | 23.44 | 31.90 |
| `w8a8_static_pertensor_wperchannel_qwen3_1p7b_base_instruction_g8_all196_ppl256` | W8A8 static per-tensor activation | 38.7 | 30.5 | 52.3 | 65.2 | 52.0 | 75.4 | 52.34 | 12.11 | 31.50 |
| `w8a8_smooth_a05_static_pertensor_wperchannel_qwen3_1p7b_base_instruction_g8_all196_ppl256_calib8` | SmoothQuant-style W8A8, `alpha=0.5` | 70.3 | 73.0 | 77.3 | 81.2 | 76.2 | 93.0 | 78.52 | 24.61 | 19.28 |
| `w8a8_smooth_a07_static_pertensor_wperchannel_qwen3_1p7b_base_instruction_g8_all196_ppl256_calib8` | SmoothQuant-style W8A8, `alpha=0.7` | 79.7 | 76.6 | 79.3 | 77.7 | 80.5 | 93.0 | 81.12 | 23.83 | 21.63 |

Gap to the paper:

| Stage | GLUE Avg | Gap vs Paper RTN | Gap vs Paper SmoothQuant | Drop vs Same FP16 | MMLU-Pro | Gap vs Paper RTN | Gap vs Paper SmoothQuant | PPL Delta vs Same FP16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W8A8 dynamic per-token | 81.90 | -1.77 | -5.42 | -0.98 | 29.30 | +5.70 | -2.40 | +0.85 |
| W8A8 static per-feature | 80.14 | -3.53 | -7.18 | -2.74 | 23.44 | -0.16 | -8.26 | +15.45 |
| W8A8 static per-tensor | 52.34 | -31.33 | -34.98 | -30.54 | 12.11 | -11.49 | -19.59 | +15.05 |
| SmoothQuant-style `alpha=0.5` | 78.52 | -5.15 | -8.80 | -4.36 | 24.61 | +1.01 | -7.09 | +2.84 |
| SmoothQuant-style `alpha=0.7` | 81.12 | -2.55 | -6.20 | -1.76 | 23.83 | +0.23 | -7.87 | +5.18 |

Hardware and quantization scale:

| Method | Quantized Linears | Weight Payload | Activation Bits / Token | Activation Scale Storage | Dynamic Act Scales / Token | Smooth Scales | LUT Lookups / Token | INT8 MAC / Token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W8A8 dynamic per-token | 196 | 1,344.0 MiB | 4,128,768 | 0 | 196 | 0 | 0 | 1,409,286,144 |
| W8A8 static per-feature | 196 | 1,344.0 MiB | 4,128,768 | 0.984 MiB | 0 | 0 | 0 | 1,409,286,144 |
| W8A8 static per-tensor | 196 | 1,344.0 MiB | 4,128,768 | 0.000374 MiB | 0 | 0 | 0 | 1,409,286,144 |
| SmoothQuant-style `alpha=0.5` | 196 | 1,344.0 MiB | 4,128,768 | 0.000374 MiB | 0 | 516,096 | 0 | 1,409,286,144 |
| SmoothQuant-style `alpha=0.7` | 196 | 1,344.0 MiB | 4,128,768 | 0.000374 MiB | 0 | 516,096 | 0 | 1,409,286,144 |

Interpretation: activation quantization is the first reproduction branch that creates paper-like RTN degradation. Dynamic per-token W8A8 is still too gentle, while static per-tensor W8A8 is far too harsh. Static per-feature W8A8 nearly matches the paper RTN MMLU-Pro (`23.44` vs `23.60`) but undershoots GLUE by `3.53` points. SmoothQuant-style smoothing recovers much of the naive per-tensor collapse; `alpha=0.7` is currently the best GLUE tradeoff among these W8A8 rows (`81.12`, `-2.55` vs paper RTN), but it is still far below the paper SmoothQuant GLUE target (`87.32`) and below the paper FP16 baseline. This again points to the remaining FP16/evaluation/checkpoint mismatch plus incomplete SmoothQuant details.

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
