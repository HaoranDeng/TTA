# PQ+LUT LLM Evaluation

Minimal experiments for applying product-quantized lookup-table linear layers to real causal language models.

For the paper-targeted LUT-LLM reproduction path, see `PAPER_REPRO.md`. That path uses Qwen 3 1.7B, GLUE, SQuAD v2, MMLU-Pro, and a simplified STE activation-QAT implementation.

The goal is not to provide a fast GPU implementation. The PyTorch implementation is deliberately simple so it can answer early research questions:

- What happens to language-model quality when selected `nn.Linear` layers are replaced by PQ+LUT approximations?
- How many LUT entries, activation codes, weight codes, table lookups, and additions would an FPGA design need?
- How does a baseline model compare with the PQ+LUT model on simple metrics such as WikiText perplexity and zero-shot MMLU accuracy?

## Current Results

Detailed logs and JSON artifacts are under `results/`. The main writeups are:

- `PAPER_REPRO.md`: LUT-LLM paper reproduction path for Qwen 3 1.7B.
- `RESULTS.md`: earlier PTQ PQ+LUT feasibility runs on Qwen2.5 1.5B and 7B.

### LUT-LLM Paper Reproduction Status

The official `LUT-FPGA/LUT-LLM` artifact contains FPGA/HLS code and performance modeling, but it does not include the full PyTorch QAT/STE training code, fused lookup/reduce training kernels, GPTVQ scripts, or exact evaluation harness used for the paper's Table III. This repo implements a transparent PyTorch reproduction scaffold with the same model family and datasets, but it is not yet a byte-identical reproduction of the paper.

Paper Table III reference for Qwen 3 1.7B:

| Method | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | SQuADv2 | MMLU-Pro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper FP16 | 87.6 | 86.5 | 92.9 | 91.2 | 80.9 | 93.7 | 72.8 | 33.1 |
| Paper LUT-LLM final | 86.9 | 82.8 | 90.4 | 89.5 | 76.5 | 90.6 | 69.7 | 30.8 |

Most recent scai7 reproduction finding:

- The public paper artifact does not expose the exact benchmark harness or the customized checkpoint. The paper text says the FPGA prototype uses a customized Qwen 3 1.7B model and describes continuing training on FineWeb and WikiQA.
- The arXiv v2 source explicitly states the public quantization configuration: `G=512`, vector length `v=2`, weight codebook size `c_w=16`, activation codebook size `c_a=64`, and INT8 lookup tables. The newest strict first-stage attempts below use that `c_a=64, v=2, G=512` setup where applicable.
- FP16 is not yet aligned to the paper. The closest public-checkpoint protocol found so far is `Qwen/Qwen3-1.7B-Base` with instruction few-shot prompts, but it remains below the paper FP16 row on GLUE and far below it on this repo's current SQuAD generation protocol.
- Earlier formal first-step attempt: `lutllm_base_instruction_g8_all196_paperlike_squad_ka64_steqat1000_actonly_ppl64_chunk1` uses all 196 transformer-block linears, `subdim=2`, `Ka=64`, Chebyshev activation assignment, instruction prompts, 8-shot GLUE, 0-shot MMLU-Pro, 64 rows/task, and SQuAD included. Same-run FP16 is GLUE `83.33`, MMLU-Pro `39.06`, SQuAD F1 `31.86`, WikiText PPL `16.45`. The `+ Act. Quant.` result is GLUE `64.06`, MMLU-Pro `12.50`, SQuAD F1 `25.71`, WikiText PPL `207.19`.
- Gap to the paper on that earlier first-step run: FP16 is `-5.47` GLUE, `+5.96` MMLU-Pro, and `-40.94` SQuAD F1 versus paper FP16. `+ Act. Quant.` is `-23.14` GLUE, `-19.30` MMLU-Pro, and `-44.59` SQuAD F1 versus the paper `+ Act. Quant.` row.
- The first reconstruction-loss attempt helped but did not close the gap. `lutllm_base_instruction_g8_all196_paperlike_squad_ka64_recon01_dense_steqat1000_actonly_ppl64_chunk1` trains activation centers plus all target dense linear weights with `reconstruction_loss_ratio=0.1`. It reaches GLUE `71.88`, MMLU-Pro `18.75`, SQuAD F1 `32.74`, WikiText PPL `242.19`, leaving gaps of `-15.33`, `-13.05`, and `-37.56` versus paper `+ Act. Quant.`.
- The FineWeb/WikiQA pilot `lutllm_base_instruction_g8_all196_lutllmpaperdata_ka64_recon01_dense_steqat1000_actonly_ppl64_chunk1` improves PPL relative to the task-supervised reconstruction run (`109.93` vs `242.19`) but hurts downstream quality: GLUE `54.43`, MMLU-Pro `12.50`, SQuAD F1 `3.37`. A small 1000-step approximation of the paper's training data is therefore not enough.
- A 1000-step centers-only STE-QAT run on the same all-196 scope improves Act Quant GLUE to `70.96`, but this is still `-16.24` below the paper `+ Act. Quant.` row. MMLU-Pro is `7.81`, still `-23.99` below paper.
- New meaningful quantization runs still quantize all 196 transformer-block linear layers, but until the FP16 baseline is closer, their accuracy should be treated as diagnostic rather than a paper reproduction.
- Code update after this run: `run_lutllm_qat.py` now supports `--reconstruction-loss-ratio`, `--task-loss-ratio`, `--weight-decay`, and a `lutllm_paper` train source that approximates the paper's FineWeb 512-token pretrain plus WikiQA finetune data mix. `pq_lut_lm/paper_eval.py` also fixes supervised prompt truncation so long SQuAD/WikiQA prompts preserve completion labels instead of masking nearly the whole answer.

July 4 continued first-step reproduction attempts, all with all 196 transformer-block linears quantized:

| Run | Stage | GLUE Avg | Gap vs Paper Act GLUE | MMLU-Pro | Gap vs Paper Act MMLU | SQuADv2 F1 | Gap vs Paper Act SQuAD | WikiText PPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Paper `+ Act. Quant.` target | target | 87.20 | 0.00 | 31.80 | 0.00 | 70.30 | 0.00 | - |
| `lutllm_base_instruction_g8_all196_fixedlabels_seq512_ka128_recon01_dense1e7_steqat1500_actonly_squad64_ppl64` | fixed-label, `Ka=128`, recon 0.1 + dense LR `1e-7`, 1500 steps | 79.43 | -7.77 | 7.81 | -23.99 | 31.96 | -38.34 | 66.77 |
| `lutllm_base_instruction_g8_all196_fixedlabels_seq512_ka256_recon1_task03_centersonly_lr1e5_steqat1500_squad64_ppl64` | fixed-label, `Ka=256`, centers-only, task 0.3 + recon 1.0, LR `1e-5` | 78.39 | -8.81 | 12.50 | -19.30 | 27.27 | -43.03 | 64.89 |
| `lutllm_base_instruction_g8_all196_fixedlabels_seq512_ka256_centersonly_lr1e4_recon01_steqat2000_squad64_ppl64` | fixed-label, `Ka=256`, centers-only, recon 0.1, LR `1e-4`, 2000 steps | 78.12 | -9.08 | 12.50 | -19.30 | 33.76 | -36.54 | 38.71 |
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

Current best first-step rows are split by metric, not solved: fixed-label `Ka=128` dense gives the best GLUE so far, fixed-label `Ka=256` centers-only gives the best PPL, soft-hard gives the best SQuAD F1, and the older pre-label-fix `Ka=256` row still gives the best MMLU-Pro. The fixed-label `Ka=256` conservative row recovers MMLU-Pro to `21.88`, but GLUE and SQuAD remain far below paper. The newest `Ka=256` dense+centroid run does not improve that tradeoff, and the two newest strict `c_a=64` public-paper-configuration dense+centroid QAT runs also do not close the gap. None is close to the paper's first-stage row yet.

SQuAD baseline prompt probe on `Qwen/Qwen3-1.7B-Base`, 64 SQuAD v2 validation examples, FP16/BF16 inference only:

| Prompt Probe | Best F1 | Gap vs Paper FP16 SQuAD |
|---|---:|---:|
| Current instruction prompt, max 24 new tokens | 31.86 | -40.94 |
| Best simple probe (`short_span`, max 16/24/32) | 34.41 | -38.39 |
| Best simple probe (`qa_only`, max 16/24/32) | 34.40 | -38.40 |

Interpretation: the SQuAD gap is not explained by the repo's current generation length or a simple prompt variant. The public checkpoint/protocol remains far below the paper FP16 SQuAD baseline before quantization.

Trained activation-LUT diagnostic:

| Run | Samples | GLUE Avg | Gap vs Paper Act GLUE | MMLU-Pro | Gap vs Paper Act MMLU | SQuADv2 F1 | Gap vs Paper Act SQuAD | WikiText PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `actlutfit_centers_all196_ca64_fit50_cv2048_lr3e3_clr3e4_temp05_squad32_ppl64` | 32/task | 51.56 | -35.64 | 6.25 | -25.55 | 28.12 | -42.17 | 98,629.29 |

This layerwise trained-LUT path fits expanded activation-LUT values and activation centers locally for all 196 target linears. It is closer to the paper phrase "trained lookup tables" than fixed-center LUT fitting, but it performs much worse than the STE activation-QAT path. Local LUT fitting is therefore not the missing ingredient by itself.

Task-adapted checkpoint diagnostic:

| Run | Stage | GLUE Avg | Gap vs Paper Target | MMLU-Pro | Gap vs Paper Target | SQuADv2 F1 | Gap vs Paper Target | WikiText PPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `eval_taskadapt_fixedlabels_qwen3_1p7b_base_instruction_paperall_1000_g8_squad64` | FP16 after fixed-label 1000-update task adaptation | 84.64 | -4.16 vs FP16 | 37.50 | +4.40 vs FP16 | 32.81 | -39.99 vs FP16 | - |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_task03_recon1_steqat1500_squad64_ppl64` | `Ka=256`, centers-only, task 0.3 + recon 1.0 | 77.34 | -9.86 vs Act | 25.00 | -6.80 vs Act | 33.73 | -36.57 vs Act | 54.94 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka128_dense_wd001_lr1e4_dense1e7_recon01_task1_steqat1500_squad64_ppl64` | `Ka=128`, dense+centroids, task 1.0 + recon 0.1 | 77.08 | -10.12 vs Act | 17.19 | -14.61 vs Act | 28.87 | -41.43 vs Act | 67.32 |
| `lutllm_taskadapt_fixedlabels1000_all196_ca64_km3_sample2048_centersonly_lr5e5_task1_steqat2000_squad64_ppl64` | strict `c_a=64`, KMeans 3, sample limit 2048, centers-only task loss | 74.22 | -12.98 vs Act | 10.94 | -20.86 vs Act | 24.11 | -46.19 vs Act | 130.17 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_km2_sample1024_centersonly_lr5e5_task03_recon1_steqat2000_squad64_ppl64` | `Ka=256`, KMeans 2, sample limit 1024, task 0.3 + recon 1.0 | 71.61 | -15.59 vs Act | 10.94 | -20.86 vs Act | 33.73 | -36.57 vs Act | 89.26 |
| `lutllm_taskadapt_fixedlabels1000_all196_subdim1_ka256_centersonly_lr1e5_recon1_task03_steqat1500_squad64_ppl64` | `subdim=1`, `Ka=256`, centers-only, task 0.3 + recon 1.0 | 66.93 | -20.27 vs Act | 14.06 | -17.74 vs Act | 30.65 | -39.65 vs Act | 77.18 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_lr1e4_recon01_task1_steqat1500_squad64_ppl64` | `Ka=256`, centers-only, task 1.0 + recon 0.1, LR `1e-4` | 66.15 | -21.05 vs Act | 12.50 | -19.30 vs Act | 32.46 | -37.84 vs Act | 47.25 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_l2_centersonly_lr1e5_recon1_task03_steqat2000_squad64_ppl64` | `Ka=256`, L2 assignment, task 0.3 + recon 1.0 | 60.42 | -26.78 vs Act | 10.94 | -20.86 vs Act | 28.57 | -41.73 vs Act | 173.79 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_dense_wd0_lr1e5_dense1e8_recon1_task03_steqat2000_squad64_ppl64` | `Ka=256`, dense+centroids, dense LR `1e-8`, task 0.3 + recon 1.0 | 60.42 | -26.78 vs Act | 7.81 | -23.99 vs Act | 26.49 | -43.81 vs Act | 223.25 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_lr3e5_recon1_task03_steqat3000_squad64_ppl64` | `Ka=256`, centers-only, task 0.3 + recon 1.0, LR `3e-5`, 3000 steps | 58.85 | -28.35 vs Act | 7.81 | -23.99 vs Act | 7.64 | -62.66 vs Act | 229.21 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_recononly_lr1e5_steqat2000_squad64_ppl64` | `Ka=256`, centers-only, reconstruction-only | 57.29 | -29.91 vs Act | 6.25 | -25.55 vs Act | 24.13 | -46.17 vs Act | 248.89 |
| `lutllm_taskadapt_fixedlabels1000_all196_ka256_centersonly_steinput01_lr1e5_recon1_task03_steqat2000_squad64_ppl64` | `Ka=256`, STE input gradient scale 0.1 | 56.77 | -30.43 vs Act | 10.94 | -20.86 vs Act | 1.56 | -68.74 vs Act | 265.69 |

The fixed-label task adaptation improves the FP16 GLUE/MMLU starting point, but SQuAD remains far below the paper FP16 baseline. Quantizing all 196 linears from that checkpoint still drops GLUE by about 7 points in the best task-adapted run. Stronger KMeans initialization, L2 assignment, reconstruction-only training, smaller STE input gradients, dense-weight QAT, `subdim=1`, higher LR, longer training, and the task-heavy LR `1e-4` branch all fail to close the gap, so this route still does not reproduce the paper first-stage row. Full-layer `soft_hard` assignment with `Ka=256` and `Ka=64` OOMs on a 140GB H200 in this unfused PyTorch implementation.

Current FP16 baseline-alignment gap:

| Run | Protocol | Samples | GLUE Avg | Gap vs Paper FP16 GLUE | MMLU-Pro | Gap vs Paper FP16 MMLU |
|---|---|---:|---:|---:|---:|---:|
| Paper FP16 | customized Qwen 3 1.7B | full paper eval | 88.80 | 0.00 | 33.10 | 0.00 |
| `lutllm_base_instruction_g8_all196_paperlike_squad_ka64_steqat1000_actonly_ppl64_chunk1` FP16 | internal instruction 8-shot GLUE, 0-shot MMLU, SQuAD included | 64/task | 83.33 | -5.47 | 39.06 | +5.96 |
| `lutllm_base_instruction_g8_all196_shufcalib_ka64_calib1024_k5_init_actonly_ppl256` FP16 | internal instruction 8-shot GLUE, 0-shot MMLU | 256/task | 82.88 | -5.92 | 29.69 | -3.41 |
| `lutllm_base_instruction_g8_all196_shufcalib_steqat1000_int8_actonly_ppl128` FP16 | internal instruction 8-shot GLUE, 0-shot MMLU | 128/task | 83.07 | -5.73 | 33.59 | +0.49 |
| `baseline_prompt_grid_qwen3_1p7b_base_512_more_shots/instruction_g8_m0_plain` | internal instruction 8-shot GLUE, 0-shot MMLU | 512/task | 82.35 | -6.45 | 28.52 | -4.58 |
| `baseline_prompt_grid_qwen3_1p7b_base_512_more_shots/instruction_g16_m8_plain` | internal instruction 16-shot GLUE, 8-shot MMLU | 512/task | 81.52 | -7.28 | 30.27 | -2.83 |
| `lmeval_qwen3_1p7b_base_glue6_limit1024` | standard EleutherAI `lm_eval` GLUE prompts | 1024/task limit | 71.63 | -17.17 | - | - |
| `lmeval_qwen3_1p7b_glue6_limit1024` | standard `lm_eval`, non-Base public checkpoint | 1024/task limit | 62.01 | -26.79 | - | - |
| `lmeval_taskadapt_glue1024_500_glue6_limit1024` | simple GLUE-only task adaptation, then standard `lm_eval` | 1024/task limit | 67.97 | -20.83 | - | - |

Updated public-checkpoint baselines:

| Run | Model | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | SQuADv2 F1 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_qwen3_1p7b_base_instruction_128` | `Qwen/Qwen3-1.7B-Base`, instruction prompt | 78.9 | 68.0 | 78.1 | 85.2 | 80.5 | 86.7 | 36.0 | 33.6 |
| `paper_baseline_qwen3_1p7b_128` | `Qwen/Qwen3-1.7B` | 42.2 | 70.3 | 46.1 | 28.1 | 52.3 | 77.3 | 12.2 | 23.4 |
| `paper_baseline_qwen3_1p7b_base_128` | `Qwen/Qwen3-1.7B-Base` | 41.4 | 74.2 | 74.2 | 56.2 | 52.3 | 57.0 | 36.0 | 28.9 |
| `paper_baseline_qwen3_1p7b_chat_128` | `Qwen/Qwen3-1.7B`, chat template | 29.7 | 68.0 | 56.2 | 51.6 | 52.3 | 81.2 | 33.9 | 10.9 |

Formal all-196-linear LUT-LLM reproduction attempts on the corrected Base+instruction protocol:

| Run | Stage | Samples | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lutllm_base_instruction_g8_all196_paperlike_squad_ka64_steqat1000_actonly_ppl64_chunk1` | FP16 baseline, SQuAD included | 64 | 84.4 | 73.4 | 81.2 | 87.5 | 76.6 | 96.9 | 39.1 |
| same | centers-only STE Act Quant, 1000 steps, SQuAD included | 64 | 57.8 | 71.9 | 60.9 | 51.6 | 53.1 | 89.1 | 12.5 |
| `lutllm_base_instruction_g8_all196_paperlike_squad_ka64_recon01_dense_steqat1000_actonly_ppl64_chunk1` | reconstruction 0.1 + dense-weight STE Act Quant, SQuAD included | 64 | 54.7 | 75.0 | 70.3 | 68.8 | 75.0 | 87.5 | 18.8 |
| `lutllm_base_instruction_g8_all196_lutllmpaperdata_ka64_recon01_dense_steqat1000_actonly_ppl64_chunk1` | FineWeb/WikiQA + reconstruction 0.1 + dense-weight STE Act Quant | 64 | 32.8 | 67.2 | 57.8 | 34.4 | 50.0 | 84.4 | 12.5 |
| `lutllm_base_instruction_g8_all196_shufcalib_ka64_calib1024_k5_init_actonly_ppl256` | FP16 baseline | 256 | 81.6 | 74.2 | 82.8 | 84.4 | 79.7 | 94.5 | 29.7 |
| same | Act Quant, `Ka=64`, no QAT | 256 | 50.4 | 70.3 | 60.5 | 68.4 | 62.9 | 53.9 | 6.2 |
| `lutllm_base_instruction_g8_all196_shufcalib_steqat1000_int8_actonly_ppl128` | FP16 baseline | 128 | 81.2 | 75.0 | 82.0 | 87.5 | 79.7 | 93.0 | 33.6 |
| same | centers-only STE Act Quant, 1000 steps | 128 | 68.0 | 75.0 | 62.5 | 64.1 | 72.7 | 83.6 | 7.8 |
| `lutllm_base_instruction_g8_all196_subdim4_ka64_steqat1000_actonly_ppl64` | FP16 baseline | 64 | 84.4 | 73.4 | 81.2 | 87.5 | 76.6 | 96.9 | 39.1 |
| same | `subdim=4, Ka=64` centers-only STE Act Quant, 1000 steps | 64 | 29.7 | 40.6 | 29.7 | 67.2 | 50.0 | 59.4 | 7.8 |
| `lutllm_base_instruction_g8_all196_ka128_steqat1000_actonly_ppl64` | FP16 baseline | 64 | 84.4 | 73.4 | 81.2 | 87.5 | 76.6 | 96.9 | 39.1 |
| same | `subdim=2, Ka=128` centers-only STE Act Quant, 1000 steps | 64 | 43.8 | 76.6 | 62.5 | 71.9 | 64.1 | 93.8 | 7.8 |
| `lutllm_base_instruction_all196_batched_traincalib_steqat1000_int8_64_actonly` | FP16 baseline | 64 | 82.8 | 67.2 | 81.2 | 84.4 | 78.1 | 87.5 | 39.1 |
| same | simplified STE Act Quant | 64 | 37.5 | 68.8 | 51.6 | 46.9 | 51.6 | 60.9 | 9.4 |
| `lutllm_base_instruction_all196_batched_traincalib_actlutfit10_int8_final16_v4` | FP16 baseline | 16 | 87.5 | 56.2 | 75.0 | 75.0 | 87.5 | 81.2 | 62.5 |
| same | reconstructed final LUT | 16 | 31.2 | 25.0 | 68.8 | 50.0 | 62.5 | 50.0 | 6.2 |

Current gap to the paper on the latest all-196 diagnostic:

| Stage | GLUE Avg | Paper Target | Gap | MMLU-Pro | Paper Target | Gap | WikiText PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper-like first-step FP16, SQuAD included | 83.33 | 88.80 | -5.47 | 39.06 | 33.10 | +5.96 | 16.45 |
| Paper-like centers-only STE Act Quant, SQuAD included | 64.06 | 87.20 | -23.14 | 12.50 | 31.80 | -19.30 | 207.19 |
| Paper-like reconstruction 0.1 + dense-weight STE Act Quant, SQuAD included | 71.88 | 87.20 | -15.33 | 18.75 | 31.80 | -13.05 | 242.19 |
| FineWeb/WikiQA reconstruction 0.1 + dense-weight STE Act Quant | 54.43 | 87.20 | -32.77 | 12.50 | 31.80 | -19.30 | 109.93 |
| FP16 baseline | 82.88 | 88.80 | -5.92 | 29.69 | 33.10 | -3.41 | 16.45 |
| Act Quant, `Ka=64`, no QAT | 61.07 | 87.20 | -26.13 | 6.25 | 31.80 | -25.55 | 332.60 |
| centers-only STE Act Quant, 1000 steps | 70.96 | 87.20 | -16.24 | 7.81 | 31.80 | -23.99 | 335.62 |
| `subdim=4, Ka=64` centers-only STE Act Quant, 1000 steps | 46.09 | 87.20 | -41.11 | 7.81 | 31.80 | -23.99 | 19,947.91 |
| `subdim=2, Ka=128` centers-only STE Act Quant, 1000 steps | 68.75 | 87.20 | -18.45 | 7.81 | 31.80 | -23.99 | 217.62 |

Simple RTN weight-only sanity check on the same corrected Base+instruction protocol (`Qwen/Qwen3-1.7B-Base`, 256 rows/task, 8-shot GLUE, 0-shot MMLU-Pro, SQuAD skipped, 4096-token WikiText PPL). These rows quantize all 196 transformer-block linears, but they are dense dequantized PyTorch linears, not LUT inference.

| Method | Scope | GLUE Avg | Paper Target | Gap | Quant Drop vs Same FP16 | MMLU-Pro | Paper Target | Gap | WikiText PPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper FP16 | customized Qwen 3 1.7B | 88.80 | 88.80 | 0.00 | - | 33.10 | 33.10 | 0.00 | - |
| Same-run FP16 | public Base checkpoint | 82.88 | 88.80 | -5.92 | - | 29.69 | 33.10 | -3.41 | 16.45 |
| Paper RTN INT8 | paper setup | 83.67 | 83.67 | 0.00 | -5.13 | 23.60 | 23.60 | 0.00 | - |
| RTN INT8 per-channel | all 196 linears | 83.07 | 83.67 | -0.60 | +0.19 | 30.08 | 23.60 | +6.48 | 16.59 |
| RTN INT8 group128 | all 196 linears | 83.33 | 83.67 | -0.34 | +0.45 | 28.91 | 23.60 | +5.31 | 16.53 |
| RTN INT8 per-tensor | all 196 linears | 82.49 | 83.67 | -1.18 | -0.39 | 30.08 | 23.60 | +6.48 | 18.06 |

RTN hardware estimate for Qwen 3 1.7B all-196 target linears:

| Method | Codebook / Levels | Packed Weight Payload | Scale Count | FP16 Scale Storage | LUT Lookups / Token | Dense MAC / Token |
|---|---|---:|---:|---:|---:|---:|
| RTN INT8 per-channel | no codebook; 256 integer levels per output-channel scale | 1,344.0 MiB | 573,440 | 1.094 MiB | 0 | 1,409,286,144 |
| RTN INT8 group128 | no codebook; 256 integer levels per 128-weight group scale | 1,344.0 MiB | 11,010,048 | 21.000 MiB | 0 | 1,409,286,144 |
| RTN INT8 per-tensor | no codebook; 256 integer levels per tensor scale | 1,344.0 MiB | 196 | 0.000374 MiB | 0 | 1,409,286,144 |

Interpretation: the simplest weight-only RTN path does not reproduce the paper's RTN degradation. On this public-checkpoint protocol, per-channel and group128 RTN are essentially lossless, and even per-tensor RTN only drops GLUE by `0.39` points. The paper RTN row likely includes a harsher or different quantization/evaluation setup, such as activation quantization, a different scale granularity, or the customized checkpoint/protocol mismatch already visible in FP16.

W8A8 activation-quantization follow-up, still all 196 target linears and the same 256-row protocol:

| Method | Activation Scale | GLUE Avg | Gap vs Paper RTN | Drop vs Same FP16 | MMLU-Pro | Gap vs Paper RTN | WikiText PPL |
|---|---|---:|---:|---:|---:|---:|---:|
| W8A8 dynamic | per token, per linear | 81.90 | -1.77 | -0.98 | 29.30 | +5.70 | 17.30 |
| W8A8 static | per input feature | 80.14 | -3.53 | -2.74 | 23.44 | -0.16 | 31.90 |
| W8A8 static | per tensor | 52.34 | -31.33 | -30.54 | 12.11 | -11.49 | 31.50 |
| SmoothQuant-style W8A8 `alpha=0.5` | smoothed, per tensor, 8 calib batches | 78.52 | -5.15 | -4.36 | 24.61 | +1.01 | 19.28 |
| SmoothQuant-style W8A8 `alpha=0.7` | smoothed, per tensor, 8 calib batches | 81.12 | -2.55 | -1.76 | 23.83 | +0.23 | 21.63 |

These W8A8 rows show that activation quantization can create RTN-like degradation, unlike weight-only RTN. The current SmoothQuant-style scaffold is still below the paper SmoothQuant target (`87.32` GLUE, `31.70` MMLU-Pro), but `alpha=0.7` gets close to the paper RTN MMLU-Pro while keeping GLUE much better than naive per-tensor activation quantization.

After commit `3708f19`, paper-supervised calibration/training batches are shuffled before selecting calibration batches. This avoids the earlier artifact where the first calibration batches came mostly from MNLI. New all-layer act-quant runs with WikiText PPL:

| Run | Scope | Stage | Samples | WikiText PPL | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lutllm_base_instruction_all196_shufcalib_ka64_calib1024_k5_init_actonly_ppl64` | 196 block linears | FP16 baseline | 64 | 16.4 | 82.8 | 67.2 | 81.2 | 84.4 | 78.1 | 87.5 | 39.1 |
| same | 196 block linears | Act Quant, `Ka=64`, no QAT | 64 | 332.6 | 39.1 | 71.9 | 67.2 | 65.6 | 54.7 | 62.5 | 1.6 |
| `lutllm_base_instruction_all196_shufcalib_ka256_calib512_k3_init_actonly_ppl64` | 196 block linears | Act Quant, `Ka=256`, no QAT | 64 | 250.9 | 50.0 | 73.4 | 67.2 | 65.6 | 59.4 | 65.6 | 3.1 |
| `lutllm_base_instruction_all196_shufcalib_steqat1000_int8_64_actonly_ppl64` | 196 block linears | simplified STE Act Quant, 1000 steps | 64 | 413.4 | 51.6 | 68.8 | 62.5 | 75.0 | 60.9 | 70.3 | 9.4 |
| `lutllm_base_instruction_all197_includelmhead_shufcalib_ka64_calib512_k3_actonly_ppl16` | 197 linears incl. `lm_head` | FP16 baseline | 16 | 15.6 | 87.5 | 56.2 | 75.0 | 75.0 | 87.5 | 81.2 | 62.5 |
| same | 197 linears incl. `lm_head` | Act Quant, `Ka=64`, no QAT | 16 | 403.4 | 18.8 | 50.0 | 50.0 | 62.5 | 56.2 | 62.5 | 6.2 |

Shuffled-calibration hardware scale:

| Run | Quantized Linears | Activation Centers | Expanded Act-LUT FP16 | Lookups / Token | Act Code Bits / Token | Centroid Distance Vectors / Token |
|---|---:|---:|---:|---:|---:|---:|
| all196 `Ka=64` | 196 | 33,030,144 | 86,016.0 MiB | 704,643,072 | 1,548,288 | 16,515,072 |
| all196 `Ka=128` | 196 | 66,060,288 | 172,032.0 MiB | 704,643,072 | 1,806,336 | 33,030,144 |
| all196 `Ka=256` | 196 | 132,120,576 | 344,064.0 MiB | 704,643,072 | 2,064,384 | 66,060,288 |
| all196 `subdim=4, Ka=64` | 196 | 33,030,144 | 43,008.0 MiB | 352,321,536 | 774,144 | 8,257,536 |
| all197 `Ka=64`, incl. `lm_head` | 197 | 33,161,216 | 105,008.0 MiB | 860,225,536 | 1,554,432 | 16,580,608 |

Additional all-196 attempts:

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

Interpretation: WikiText loss can reduce PPL versus task-supervised centers-only QAT, but it is still far from FP16 and damages downstream accuracy. Naively training dense linear weights alongside activation centers does not recover accuracy. The tested soft-hard STE and input-gradient scaling variants do not improve act-only PPL. `subdim=4` halves lookups and expanded LUT size but destroys PPL in this scaffold. A per-output affine correction reduces final-LUT PPL relative to the local k-means final run, but accuracy remains near random. The first output-aware weight-code reassignment and least-squares centroid-refinement attempts also do not recover PPL or accuracy. The compact final LUT path is still dominated by missing GPTVQ-style reconstruction-aware weight quantization, not just by scale/bias error.

The all-196 final LUT run uses compact INT8 LUT storage `2,688.0 MiB`, packed weight codes `336.0 MiB`, and `704,643,072` table lookups per token. Its expanded FP16 activation-LUT intermediate would be `86,016.0 MiB`, which is why direct Act-LUT evaluation is very slow in the PyTorch prototype. Earlier 7-linear runs are now treated only as debugging/profiling runs, not formal reproduction results.

Simplified full-layer QAT smoke on scai7:

| Stage | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | MMLU-Pro |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline, same 16 rows | 25.0 | 62.5 | 37.5 | 37.5 | 62.5 | 87.5 | 43.8 |
| + Act. Quant., simplified STE | 25.0 | 37.5 | 62.5 | 62.5 | 62.5 | 56.2 | 6.2 |
| + Weight Quant., final LUT | 43.8 | 37.5 | 62.5 | 56.2 | 62.5 | 50.0 | 6.2 |

Stronger paper-supervised QAT attempts:

| Run | Scope | Train | Eval | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | SQuADv2 F1 | MMLU-Pro |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_lutllm_qwen3_1p7b_all_paperqat100_affine` | 196 linears | 100 steps | final LUT, 32 rows/task | 40.6 | 37.5 | 40.6 | 68.8 | 56.2 | 43.8 | - | 6.2 |
| `paper_lutllm_qwen3_1p7b_all_paperqat1000_actonly` | 196 linears | 1000 steps | Act Quant, 128 rows/task | 35.9 | 68.0 | 46.1 | 32.0 | 52.3 | 53.9 | - | 13.3 |
| `paper_lutllm_qwen3_1p7b_all_actlutfit5_int4_fast16` | 196 linears | direct LUT fit | Act LUT, 16 rows/task | 25.0 | 62.5 | 50.0 | 43.8 | 62.5 | 50.0 | - | 6.2 |
| `paper_lutllm_qwen3_1p7b_all_actlutfit5_int4_expanded_final16_fast` | 196 linears | direct LUT fit | reconstructed final LUT, 16 rows/task | 25.0 | 62.5 | 31.2 | 37.5 | 62.5 | 50.0 | - | 6.2 |
| `paper_lutllm_qwen3_1p7b_7linear_paperqat500_affine` | 7 linears | 500 steps | Act Quant, 64 rows/task | 48.4 | 67.2 | 75.0 | 71.9 | 50.0 | 75.0 | 9.8 | 14.1 |
| `paper_lutllm_qwen3_1p7b_7linear_paperqat500_affine` | 7 linears | 500 steps | final LUT, 64 rows/task | 48.4 | 67.2 | 53.1 | 46.9 | 50.0 | 75.0 | 4.1 | 14.1 |

The best full-layer `+ Act Quant.` attempt so far still falls well short of the paper's `+ Act. Quant.` row. A follow-up direct activation-LUT fitting path, inferred from the paper's "trained LUT -> reconstructed weights" description, also does not recover accuracy. The remaining gap is likely in the unavailable QAT/GPTVQ/evaluation details rather than just in running a few more steps.

Final LUT hardware scale for `Qwen/Qwen3-1.7B`:

| Final LUT Bits | Quantized Linears | Compact LUT | Weight Codes Packed | Lookups / Token | Act Code Bits / Token | Theoretical Expanded LUT FP16 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 196 | 2,688.0 MiB | 336.0 MiB | 704,643,072 | 1,548,288 | 86,016.0 MiB |
| 4 | 196 | 1,344.0 MiB | 336.0 MiB | 704,643,072 | 1,548,288 | 86,016.0 MiB |

### Earlier PQ/LUT PTQ Feasibility Results

These are not paper reproductions. They are full-layer post-training quantization tests that show the naive PTQ path is too lossy.

| Model | Method | Codebook | Baseline PPL | Quant PPL | Quant MMLU Smoke | Compact LUT | Lookups / Token |
|---|---|---|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B` | LUT-LLM-style + affine | `subdim=2, Ka=64, Kw=16` | 24.78 | 658.91 | 25.0% | 2,499 MiB | 655,097,856 |
| `Qwen/Qwen2.5-1.5B` | PQ + affine | `subdim=8, Ka=128, Kw=64` | 24.78 | 2,656.55 | 0.0% | 994 MiB | 163,774,464 |
| `Qwen/Qwen2.5-7B` | LUT-LLM-style compact | `subdim=2, Ka=64, Kw=16` | 13.31 | 2,420.01 | 25.0% | 12,446 MiB | 3,262,644,224 |
| `Qwen/Qwen2.5-7B` | PQ compact + affine | `subdim=8, Ka=64, Kw=64` | 13.31 | 3,047.48 | 0.0% | 1,106 MiB | 815,661,056 |

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a small smoke test:

```bash
python run_eval.py \
  --model-id Qwen/Qwen2.5-0.5B \
  --output-dir results/tiny \
  --seq-len 128 \
  --ppl-tokens 512 \
  --calib-batches 1 \
  --mmlu-samples 4 \
  --subdim 8 \
  --ka 4 \
  --kw 4 \
  --kmeans-iters 2 \
  --max-linears 2
```

Example real-model runs:

```bash
python run_eval.py \
  --model-id Qwen/Qwen2.5-1.5B \
  --output-dir results/qwen2p5_1p5b \
  --seq-len 256 \
  --ppl-tokens 4096 \
  --calib-batches 4 \
  --mmlu-samples 64 \
  --subdim 32 \
  --ka 8 \
  --kw 16 \
  --kmeans-iters 4

python run_eval.py \
  --model-id Qwen/Qwen2.5-7B \
  --output-dir results/qwen2p5_7b \
  --seq-len 256 \
  --ppl-tokens 4096 \
  --calib-batches 4 \
  --mmlu-samples 64 \
  --subdim 32 \
  --ka 8 \
  --kw 16 \
  --kmeans-iters 4
```

By default, all transformer block linear layers matching
`(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$` are quantized. `lm_head` is excluded unless `--include-lm-head` is passed.

## LUT-LLM-Style Mode

The default `--method pq` is the original baseline in this repo. It uses one weight codebook per input sub-vector across a whole linear layer.

Use `--method lutllm` for a closer post-training approximation of the LUT-LLM artifact:

- `subdim=2`, matching the vector length used by the released HLS testbench.
- `Ka=64` activation centroids and `Kw=16` weight centroids.
- `weight_group_size=256`, so each input vector position has separate weight codebooks per output block.
- Chebyshev nearest-centroid search, matching the artifact reference code.
- 8-bit min/max quantization of the 2D LUT values before expansion/dequantization.

Example:

```bash
python run_eval.py \
  --method lutllm \
  --model-id Qwen/Qwen2.5-1.5B \
  --output-dir results/qwen15b_lutllm_7linear \
  --seq-len 64 \
  --ppl-tokens 512 \
  --calib-tokens 512 \
  --calib-batches 4 \
  --calib-vectors-per-layer 256 \
  --mmlu-samples 8 \
  --kmeans-iters 1 \
  --sample-limit 256 \
  --max-linears 7
```

This is still not the full LUT-LLM training recipe. The paper describes QAT with STE and fused lookup/reduce kernels; this repo currently implements a PTQ approximation of the visible quantization layout.

Two extra switches are useful for the scai7 full-layer runs:

- `--lut-storage compact` stores the base `M * groups * Ka * Kw` LUT instead of the expanded `M * Ka * out_features` LUT. This is slower in PyTorch but avoids huge expanded LUT tensors for 7B experiments.
- `--output-correction {bias,affine}` fits a simple per-output post-hoc correction on calibration activations. This is not part of the LUT-LLM paper, but it is a useful PTQ diagnostic because it separates scale/bias error from centroid/codebook error.
- `--weight-code-reassign-iters N` runs an experimental output-reconstruction-aware coordinate reassignment of final weight codes after local k-means weight VQ. This is a diagnostic toward GPTVQ, not a full GPTVQ implementation.
- `--weight-center-refine-iters N` runs an experimental least-squares update of final weight centroids against calibration-output reconstruction; it currently overfits and is retained as a negative diagnostic.
- `--act-train-mode {hard,soft,soft_hard}` and `--act-ste-input-scale` test inferred activation-STE variants. The tested soft-hard and input-gradient-scale variants did not improve the all-196 act-only result.

## Output Files

Each run writes:

- `summary.json`: model quality, runtime, and aggregate hardware estimate.
- `hardware_stats.json`: per-module PQ+LUT dimensions and lookup counts.
- `config.json`: exact CLI settings.

Important hardware fields:

- `base_lut_entries`: `M * Ka * Kw`, the compact LUT entries for a linear layer.
- `weight_code_count`: `M * out_features`, one weight centroid id per output channel and subspace.
- `act_code_bits_per_token`: `M * ceil(log2(Ka))`.
- `lookups_per_token`: `M * out_features`, assuming one LUT lookup per subspace per output feature.
- `adds_per_token`: `(M - 1) * out_features`, for reducing subspace partial sums.

## Caveats

This is a research scaffold, not an optimized inference path:

- Activation codebook training is post-training calibration only.
- The PyTorch forward uses gather and accumulation loops, so GPU timing is not representative of an FPGA design.
- Zero-shot MMLU here is a compact local implementation, not a byte-identical replacement for `lm-eval-harness`.
- Small codebooks are expected to hurt quality. Increase `Ka`, `Kw`, calibration data, or use OPQ/QAT before drawing final accuracy conclusions.
