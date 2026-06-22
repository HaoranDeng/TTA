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
