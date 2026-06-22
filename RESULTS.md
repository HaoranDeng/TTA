# Initial Results on scai6

Date: 2026-06-21  
GPU: NVIDIA H200  
Implementation: PyTorch PQ+LUT module, not a fused kernel  
Quantized modules: transformer block linears matching `(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$`; `lm_head` excluded  
PQ config: `subdim=32`, `Ka=16`, `Kw=16`, `kmeans_iters=2`, `calib_vectors_per_layer=256`  
Evaluation: WikiText-2 raw test, 1,016 scored tokens; MMLU `cais/mmlu` zero-shot, first 16 test rows

These are pilot numbers for hardware/accuracy feasibility. The MMLU sample is intentionally small, so use it as a quick regression signal rather than a benchmark-quality score.

## Quality

| Model | Baseline PPL | PQ+LUT PPL | Baseline MMLU | PQ+LUT MMLU |
|---|---:|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B` | 19.93 | 3,171,846.39 | 31.25% | 12.50% |
| `Qwen/Qwen2.5-7B` | 15.06 | 144,182.17 | 43.75% | 31.25% |

The current post-training PQ+LUT replacement is far too lossy when applied to every transformer block linear layer with small codebooks. The perplexity blow-up is the clearest signal. Larger codebooks, better activation calibration, OPQ/rotation, layer-wise mixed precision, or QAT are needed before this is accuracy-viable.

## Hardware Estimate

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
