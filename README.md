# PQ+LUT LLM Evaluation

Minimal experiments for applying product-quantized lookup-table linear layers to real causal language models.

The goal is not to provide a fast GPU implementation. The PyTorch implementation is deliberately simple so it can answer early research questions:

- What happens to language-model quality when selected `nn.Linear` layers are replaced by PQ+LUT approximations?
- How many LUT entries, activation codes, weight codes, table lookups, and additions would an FPGA design need?
- How does a baseline model compare with the PQ+LUT model on simple metrics such as WikiText perplexity and zero-shot MMLU accuracy?

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
