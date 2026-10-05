# TTA

A hardware–software co-design project for LLM inference on accelerators built with **3D gain-cell
memory**.

3D gain cells are compact, low-leakage DRAM-like bit cells that can be stacked above the logic in
back-end-of-line layers. They are far denser than SRAM while keeping SRAM-like access latency, so an
accelerator can carry one to two orders of magnitude more on-chip memory than today.

For the purposes of this project, the hardware is abstracted as **a GPU with a much larger SRAM**.
The software question we study is how LLM inference should be restructured to exploit that
capacity: keeping weights and KV cache on chip, trading arithmetic for lookup tables, and choosing
quantisation and table formats against an explicit on-chip memory budget.

## Lookup tables as the key approach

A lookup table (LUT) replaces computation with memory: results are precomputed once, stored, and
retrieved by index at run time. For LLM inference the relevant case is the linear layer. If weights
are quantised to a small set of values, or activations and weights are both mapped to codebooks,
then the partial dot products between an input pattern and a group of weights can be tabulated in
advance. Inference then becomes a sequence of table reads and additions rather than multiplies.

This trade is unattractive on conventional hardware because the tables are large and on-chip
memory is scarce, so lookups spill into slow off-chip DRAM. Dense 3D gain-cell memory removes that
constraint: tables that would not fit in SRAM today can stay on chip, and the memory itself becomes
the compute substrate. LUT-based inference is therefore the central technique this project builds
on. The main design questions are how to form the tables (bit-plane partial sums, vector-quantised
codebooks, or hybrids), how large they must be to preserve accuracy, and how many lookups and
additions each token costs under a given on-chip memory budget.

## Reporting efficiency

Unless stated otherwise, the efficiency of any method in this project is reported as three operation
counts, measured per generated token over the whole model:

- **Lookups**: number of table reads.
- **Additions**: number of integer or floating-point additions, including accumulation of partial
  sums.
- **Multiplications**: number of multiplications, including any scaling or dequantisation steps.

Counts are reported alongside the on-chip memory the method needs (tables, codebooks, weight
indices and scales) and its accuracy, so that every result can be read as an accuracy–memory–
operation trade-off. Wall-clock timings on GPUs are supplementary and do not replace these counts.

## Results

Qwen3-0.6B, all 196 decoder linears, counts per generated token. Perplexity on WikiText-2 (256-token
contexts). INT8 per-token activations throughout.

| Method | Lookups | Additions | Multiplications | On-chip memory | PPL |
|---|---:|---:|---:|---:|---:|
| FP16 dense | 0 | 440 M | 440 M | 840 MiB | 42.51 |
| W8A8 dense | 0 | 440 M | 441 M | 420 MiB | 43.90 |
| Bit-plane table, g=4, INT16 | 881 M | 881 M | 0.9 M | 3 361 MiB | 43.90 |
| Bit-plane table, g=5, INT8 | 705 M | 705 M | 706 M | 3 028 MiB | 43.69 |
| Bit-plane table, g=8, full | 440 M | 440 M | 0.9 M | 13 440 MiB | 43.90 |
| Bit-plane table, g=8, top-64 entries | 236 M | 733 M | 0.9 M | 6 720 MiB | 43.90 |
| Joint activation–weight VQ, v=2 (PTQ) | 220 M | 233 M | 0.3 M | 574 MiB | 1.2 × 10^7 |
| Joint activation–weight VQ, v=4 (PTQ) | 110 M | 123 M | 0.3 M | 312 MiB | 5.1 × 10^8 |
