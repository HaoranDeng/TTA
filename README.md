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
- FP16 is not yet aligned to the paper. The closest public-checkpoint protocol found so far is `Qwen/Qwen3-1.7B-Base` with instruction few-shot prompts, but it is still `-6.45` GLUE points below the paper FP16 row on a 512-example sweep.
- The latest all-196-linear run on the closest protocol (`instruction`, 8-shot GLUE, 256 rows/task) gives FP16 GLUE `82.88`, which is still `-5.92` below paper FP16. Its no-QAT activation-quantized result gives GLUE `61.07`, which is `-26.13` below the paper `+ Act. Quant.` row.
- New meaningful quantization runs still quantize all 196 transformer-block linear layers, but until the FP16 baseline is closer, their accuracy should be treated as diagnostic rather than a paper reproduction.

Current FP16 baseline-alignment gap:

| Run | Protocol | Samples | GLUE Avg | Gap vs Paper FP16 GLUE | MMLU-Pro | Gap vs Paper FP16 MMLU |
|---|---|---:|---:|---:|---:|---:|
| Paper FP16 | customized Qwen 3 1.7B | full paper eval | 88.80 | 0.00 | 33.10 | 0.00 |
| `lutllm_base_instruction_g8_all196_shufcalib_ka64_calib1024_k5_init_actonly_ppl256` FP16 | internal instruction 8-shot GLUE, 0-shot MMLU | 256/task | 82.88 | -5.92 | 29.69 | -3.41 |
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
| `lutllm_base_instruction_g8_all196_shufcalib_ka64_calib1024_k5_init_actonly_ppl256` | FP16 baseline | 256 | 81.6 | 74.2 | 82.8 | 84.4 | 79.7 | 94.5 | 29.7 |
| same | Act Quant, `Ka=64`, no QAT | 256 | 50.4 | 70.3 | 60.5 | 68.4 | 62.9 | 53.9 | 6.2 |
| `lutllm_base_instruction_all196_batched_traincalib_steqat1000_int8_64_actonly` | FP16 baseline | 64 | 82.8 | 67.2 | 81.2 | 84.4 | 78.1 | 87.5 | 39.1 |
| same | simplified STE Act Quant | 64 | 37.5 | 68.8 | 51.6 | 46.9 | 51.6 | 60.9 | 9.4 |
| `lutllm_base_instruction_all196_batched_traincalib_actlutfit10_int8_final16_v4` | FP16 baseline | 16 | 87.5 | 56.2 | 75.0 | 75.0 | 87.5 | 81.2 | 62.5 |
| same | reconstructed final LUT | 16 | 31.2 | 25.0 | 68.8 | 50.0 | 62.5 | 50.0 | 6.2 |

Current gap to the paper on the latest all-196 diagnostic:

| Stage | GLUE Avg | Paper Target | Gap | MMLU-Pro | Paper Target | Gap | WikiText PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline | 82.88 | 88.80 | -5.92 | 29.69 | 33.10 | -3.41 | 16.45 |
| Act Quant, `Ka=64`, no QAT | 61.07 | 87.20 | -26.13 | 6.25 | 31.80 | -25.55 | 332.60 |

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
