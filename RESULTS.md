# Initial Results on scai6

Date: 2026-06-21  
GPU: NVIDIA H200  
Implementation: PyTorch PQ+LUT module, not a fused kernel  
Quantized modules: transformer block linears matching `(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$`; `lm_head` excluded  
PQ config: `subdim=32`, `Ka=16`, `Kw=16`, `kmeans_iters=2`, `calib_vectors_per_layer=256`  
Evaluation: WikiText-2 raw test, 1,016 scored tokens; MMLU `cais/mmlu` zero-shot, first 16 test rows

These are pilot numbers for hardware/accuracy feasibility. The MMLU sample is intentionally small, so use it as a quick regression signal rather than a benchmark-quality score.

## Quality: 16x16 Codebook

| Model | Baseline PPL | PQ+LUT PPL | Baseline MMLU | PQ+LUT MMLU |
|---|---:|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B` | 19.93 | 3,171,846.39 | 31.25% | 12.50% |
| `Qwen/Qwen2.5-7B` | 15.06 | 144,182.17 | 43.75% | 31.25% |

The current post-training PQ+LUT replacement is far too lossy when applied to every transformer block linear layer with small codebooks. The perplexity blow-up is the clearest signal. Larger codebooks, better activation calibration, OPQ/rotation, layer-wise mixed precision, or QAT are needed before this is accuracy-viable.

## Codebook Sweep

Follow-up runs increased both activation and weight codebooks. The evaluation protocol remained the same, except the `64x64` runs used `calib_vectors_per_layer=512`, and the `128x128` runs used `calib_vectors_per_layer=1024`.

| Model | Ka | Kw | PQ+LUT PPL | PQ+LUT MMLU | Base LUT FP16 | Weight Codes Packed | Act Code Bits / Token | LUT Lookups / Token | Centroid Scalar Ops / Token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B` | 16 | 16 | 3,171,846.39 | 12.50% | 7.77 MiB | 19.52 MiB | 63,616 | 40,943,616 | 8,142,848 |
| `Qwen/Qwen2.5-1.5B` | 64 | 64 | 42,343.67 | 31.25% | 124.25 MiB | 29.29 MiB | 95,424 | 40,943,616 | 32,571,392 |
| `Qwen/Qwen2.5-1.5B` | 128 | 128 | 13,244.57 | 25.00% | 497.00 MiB | 34.17 MiB | 111,328 | 40,943,616 | 65,142,784 |
| `Qwen/Qwen2.5-7B` | 16 | 16 | 144,182.17 | 31.25% | 17.28 MiB | 97.23 MiB | 141,568 | 203,915,264 | 18,120,704 |
| `Qwen/Qwen2.5-7B` | 64 | 64 | 73,700.92 | 31.25% | 276.50 MiB | 145.85 MiB | 212,352 | 203,915,264 | 72,482,816 |
| `Qwen/Qwen2.5-7B` | 128 | 128 | 317,051.37 | 25.00% | 1,106.00 MiB | 170.16 MiB | 247,744 | 203,915,264 | 144,965,632 |

Observations:

- Larger codebooks are feasible in memory on H200 for this prototype, and they are straightforward for the compact FPGA representation.
- `lookups_per_token` is unchanged when Ka/Kw increase because it depends on `sum(M * out_features)` for the selected linears.
- Compact LUT storage scales with `Ka * Kw`; activation centroid search scales with `Ka`; packed weight code storage scales only with `ceil(log2(Kw))`.
- Accuracy is not monotonic. The 1.5B model improves from `16x16` to `128x128`, but remains unusable by PPL. The 7B model improves at `64x64` and then regresses at `128x128`, likely because this independent post-training activation/weight PQ is unstable without rotations, better calibration, or fine-tuning.

## LUT-LLM-Style PTQ Approximation

The repository now has a separate `--method lutllm` path. It follows the released LUT-LLM artifact where possible: `subdim=2`, `Ka=64`, `Kw=16`, weight codebooks per 256-output block, Chebyshev activation search, and 8-bit quantized 2D LUT values. It is still post-training quantization, not the paper's full QAT/STE training recipe.

| Run | Model | Quantized Linears | Baseline PPL | LUT-LLM PTQ PPL | Baseline MMLU | LUT-LLM PTQ MMLU |
|---|---|---:|---:|---:|---:|---:|
| `scai7_lutllm_qwen05_1linear` | `Qwen/Qwen2.5-0.5B` | 1 | 30.66 | 30.48 | 0.00% | 0.00% |
| `scai7_lutllm_qwen05_7linear` | `Qwen/Qwen2.5-0.5B` | 7 | 34.60 | 868.41 | 12.50% | 37.50% |
| `scai7_lutllm_qwen15b_7linear` | `Qwen/Qwen2.5-1.5B` | 7 | 24.78 | 56.90 | 50.00% | 25.00% |

Hardware estimates for these partial runs:

| Run | Quantized LUT Storage | Weight Codes Packed | Act Code Bits / Token | LUT Lookups / Token |
|---|---:|---:|---:|---:|
| `scai7_lutllm_qwen05_1linear` | 1.75 MiB | 0.19 MiB | 2,688 | 401,408 |
| `scai7_lutllm_qwen05_7linear` | 30.50 MiB | 3.55 MiB | 30,720 | 7,454,720 |
| `scai7_lutllm_qwen15b_7linear` | 89.25 MiB | 11.16 MiB | 54,528 | 23,396,352 |

The 1.5B partial result is much closer than the earlier whole-model naive PQ baseline, but it only replaces one transformer block. Full-model replication likely needs the paper's training recipe rather than pure KMeans PTQ.

## Full-Layer scai7 Runs

Date: 2026-06-22  
GPU: NVIDIA H200 on scai7  
Evaluation: WikiText-2 raw test with 504 scored tokens for 1.5B, 126 scored tokens for 7B compact; MMLU zero-shot smoke set with 8 rows for 1.5B and 4 rows for 7B compact. The MMLU numbers below are useful as regression signals, not benchmark-quality MMLU.

All 1.5B runs quantize every transformer block linear matching `(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$`, 196 linears total. `lm_head` remains dense.

### LUT-LLM-Style Full-Layer PTQ

These runs use the LUT-LLM-style codebook shape: `subdim=2`, `Ka=64`, `Kw=16`, `weight_group_size=256`, and 8-bit LUT values. The compact FPGA LUT has `M * groups * Ka * Kw` entries; the PyTorch expanded LUT is only for unfused GPU evaluation speed.

| Run | Distance | Correction | Baseline PPL | LUT PPL | Baseline MMLU | LUT MMLU | Compact LUT | Weight Codes | Lookups / Token | Act Code Bits / Token |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scai7_lutllm_qwen15b_all` | Chebyshev | none | 24.78 | 32,418.57 | 50.0% | 25.0% | 2,499.00 MiB | 312.38 MiB | 655,097,856 | 1,526,784 |
| `scai7_lutllm_qwen15b_all_affine` | Chebyshev | affine output | 24.78 | 658.91 | 50.0% | 25.0% | 2,499.00 MiB | 312.38 MiB | 655,097,856 | 1,526,784 |
| `scai7_lutllm_qwen15b_all_l2` | L2 | none | 24.78 | 11,574.71 | 50.0% | 37.5% | 2,499.00 MiB | 312.38 MiB | 655,097,856 | 1,526,784 |
| `scai7_lutllm_qwen15b_all_l2_affine` | L2 | affine output | 24.78 | 1,032.68 | 50.0% | 0.0% | 2,499.00 MiB | 312.38 MiB | 655,097,856 | 1,526,784 |

The output affine correction is the only tested LUT-LLM-style PTQ change that substantially improved PPL. It reduced the all-layer 1.5B PPL from `32,418.57` to `658.91` with Chebyshev assignment. L2 plus affine was worse at `1,032.68`. These results are still far from the FP16 baseline PPL of `24.78`, which supports the earlier conclusion: reproducing LUT-LLM's reported accuracy likely requires the paper's QAT/STE training path, not only post-training KMeans plus LUT substitution.

### PQ Full-Layer Variants

These are standard independent activation/weight PQ variants with L2 assignment and per-output affine correction. They keep FP16 LUT values in this prototype (`lut_quant_bits=0`).

| Run | PQ Config | Baseline PPL | PQ PPL | Baseline MMLU | PQ MMLU | Compact LUT | Weight Codes | Lookups / Token | Act Code Bits / Token |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scai7_pq_qwen15b_s16_k64_affine` | `subdim=16`, `Ka=64`, `Kw=64` | 24.78 | 4,500.05 | 50.0% | 25.0% | 248.50 MiB | 58.57 MiB | 81,887,232 | 190,848 |
| `scai7_pq_qwen15b_s8_k64_affine` | `subdim=8`, `Ka=64`, `Kw=64` | 24.78 | 2,761.34 | 50.0% | 25.0% | 497.00 MiB | 117.14 MiB | 163,774,464 | 381,696 |
| `scai7_pq_qwen15b_s4_k64_affine` | `subdim=4`, `Ka=64`, `Kw=64` | 24.78 | 211,263.19 | 50.0% | 25.0% | 994.00 MiB | 234.28 MiB | 327,548,928 | 763,392 |
| `scai7_pq_qwen15b_s8_ka128_kw64_affine` | `subdim=8`, `Ka=128`, `Kw=64` | 24.78 | 2,656.55 | 50.0% | 0.0% | 994.00 MiB | 117.14 MiB | 163,774,464 | 445,312 |

Among these PQ runs, `subdim=8`, `Ka=128`, `Kw=64` had the lowest PPL (`2,656.55`), while `subdim=8`, `Ka=64`, `Kw=64` was close (`2,761.34`) with lower activation-code and LUT footprint. Smaller `subdim=4` was not better; it increased lookup and LUT cost and made PPL much worse. The MMLU smoke set is too small to rank close methods reliably.

### 7B Compact Check

The 7B run uses compact LUT storage to avoid materializing the theoretical expanded LUT in GPU memory. The PyTorch compact forward is slower but closer to the FPGA representation.

| Run | Model | Method | Config | Baseline PPL | Quant PPL | Baseline MMLU | Quant MMLU | Compact LUT | Weight Codes | Lookups / Token |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `scai7_pq_qwen7b_s8_k64_affine_compact` | `Qwen/Qwen2.5-7B` | PQ | `subdim=8`, `Ka=64`, `Kw=64`, affine | 13.31 | 3,047.48 | 25.0% | 0.0% | 1,106.00 MiB | 583.41 MiB | 815,661,056 |
| `scai7_lutllm_qwen7b_all_compact` | `Qwen/Qwen2.5-7B` | LUT-LLM-style | `subdim=2`, `Ka=64`, `Kw=16`, compact | 13.31 | 2,420.01 | 25.0% | 25.0% | 12,446.00 MiB | 1,555.75 MiB | 3,262,644,224 |

For the completed 7B PQ compact run, the theoretical expanded LUT would be `99,568.00 MiB` FP16, while compact storage is `1,106.00 MiB`. For the 7B LUT-LLM-style compact run, the theoretical expanded LUT would be `398,272.00 MiB` FP16, while compact 8-bit LUT storage is `12,446.00 MiB`. The compact representation is the one that matters for FPGA sizing.

## Hardware Estimate: 16x16 Codebook

All estimates are per token for one full model forward through the quantized linear modules. `lookups_per_token` assumes one compact LUT lookup per output feature per PQ subspace.

| Model | Quantized Linears | Base LUT Entries | Base LUT FP16 | Weight Codes Packed | Act Code Bits / Token | LUT Lookups / Token | Adds / Token |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B` | 196 | 4,071,424 | 7.77 MiB | 19.52 MiB | 63,616 | 40,943,616 | 40,298,496 |
| `Qwen/Qwen2.5-7B` | 196 | 9,060,352 | 17.28 MiB | 97.23 MiB | 141,568 | 203,915,264 | 202,524,672 |

The PyTorch implementation also materializes an expanded LUT for speed. That is not required for an FPGA design, but it explains GPU memory usage:

| Model | Expanded LUT FP16 |
|---|---:|
| `Qwen/Qwen2.5-1.5B` | 1,249.50 MiB |
| `Qwen/Qwen2.5-7B` | 6,223.00 MiB |

## Runtime Notes

| Model | Load | Calibration | Quantization | PQ PPL Eval | PQ MMLU Eval |
|---|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B` | 205.77s | 0.12s | 28.05s | 3.14s | 6.31s |
| `Qwen/Qwen2.5-7B` | 927.75s | 0.48s | 66.86s | 6.66s | 13.89s |

The 7B load time includes the initial Hugging Face weight download on scai6. Timing is not the main result here because the forward path is unfused PyTorch gather/accumulate code.

## Files

- `results/qwen15b_all/summary.json`
- `results/qwen15b_all/hardware_stats.json`
- `results/qwen7b_all/summary.json`
- `results/qwen7b_all/hardware_stats.json`
