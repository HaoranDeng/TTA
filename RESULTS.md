# Results

Status as of 2026-10-05. All numbers come from runs on the internal cluster; the code that produced
them will be added to this repository as it is cleaned up.

## Conventions

- **Scope.** Unless stated otherwise, a method replaces all 196 decoder linear projections
  (q, k, v, o, gate, up, down in each of 28 layers). Embeddings, the LM head, attention QK/AV
  products, normalisation and non-linearities stay in FP16 and are excluded from the counts.
- **Operation counts** are per generated token over those 196 linears, following the README:
  lookups (table reads), additions (including accumulation), multiplications (including scaling and
  dequantisation). Counts are derived from layer shapes and the kernel's data flow; they are not
  profiler measurements.
- **Memory** is the on-chip storage the method needs for tables, codebooks, weight indices and
  scales, assuming nothing is shared with other layers unless noted.
- **Accuracy protocols differ between sections** and are stated in each. The two WikiText-2
  perplexities used here (256-token contexts on 8 192 test tokens in Section 1; 2 048-token contexts
  on the full test split in Section 2) are not comparable with each other.

Dense reference points:

| Model | Decoder linears | MACs / token | Output elems / token | Distinct input elems / token |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 196 | 440.4 M | 344 064 | 200 704 |
| Qwen3-1.7B | 196 | 1 409.3 M | 573 440 | 344 064 |

## 1. Table-based linear layers on Qwen3-0.6B

Date 2026-09-30. Every decoder linear runs through a Triton lookup kernel; outputs are checked
against an independent PyTorch reference. Activations are dynamic per-token symmetric INT8, weights
per-output-channel symmetric INT8. Calibration: first 512 tokens of WikiText-2 train. Perplexity:
non-overlapping 256-token contexts, 8 160 scored tokens of WikiText-2 test. No training of any kind.

Methods:

- **Bit-plane tables.** Weights are grouped `g` at a time along the input dimension. For each
  group and output, a table holds the 2^g subset sums of the group's INT8 weights. Each of the 8
  activation bit planes selects one entry per group; entries are shifted and accumulated.
  `g=4` stores exact INT16 sums; `g=5` additionally quantises each table to INT8 with one FP32
  scale per (group, output).
- **Joint activation–weight VQ** (LUT-LLM shape, PTQ only). Activation codebook of 64 centres per
  `v`-dimensional slice, weight codebook of 16 centres shared by 512 outputs, 4-bit packed weight
  indices, 8-bit affine-quantised dot-product table. Centres from farthest-point init plus 16 Lloyd
  iterations; Chebyshev nearest-centre search at inference. No QAT and no GPTVQ, so this is a proxy,
  not the published LUT-LLM accuracy.

| Method | Lookups | Additions | Multiplications | On-chip memory | PPL (256-ctx) | GPU decode ms / token |
|---|---:|---:|---:|---:|---:|---:|
| FP16 dense | 0 | 440.4 M | 440.4 M | 840 MiB (weights) | 42.51 | 15.2 |
| W8A8 dense reference | 0 | 440.4 M | 441.3 M | 420 MiB (weights) | 43.90 | 33.6 |
| Bit-plane g=4, INT16 table | 880.8 M | 880.8 M | 0.9 M | 3 361 MiB | 43.90 | 27.4 |
| Bit-plane g=5, INT8 table | 705.3 M | 705.3 M | 706.2 M | 3 028 MiB | 43.69 | 27.4 |
| Joint VQ v=2 (PTQ proxy) | 220.2 M | 233.0 M | 0.3 M | 574 MiB | 1.2 × 10^7 | 27.3 |
| Joint VQ v=4 (PTQ proxy) | 110.1 M | 123.0 M | 0.3 M | 312 MiB | 5.1 × 10^8 | 27.1 |

Notes on the counts:

- Bit-plane lookups = 8 planes × ⌈d/g⌉ groups × m outputs, summed over layers; one shift-and-add
  per lookup. The 0.9 M multiplications are the per-token activation quantisation (200 704) plus two
  output scales per output element (688 128).
- The INT8 table variant multiplies every looked-up value by its (group, output) table scale before
  accumulation, so its multiplication count equals its lookup count. Hoisting that scale would need a
  different table layout.
- VQ lookups = (d/v) × m. Additions include 12.8 M subtractions for the Chebyshev centre search
  (64 centres × every input element). Multiplications are one affine scale per output.
- Bit-plane g=4 is bit-exact with the W8A8 reference (identical NLL), so the table introduces no
  approximation; g=5 INT8 tables change PPL by less than 0.3. Both VQ variants collapse without QAT.
- The 8 MiB-per-layer budget from the LUT-LLM FPGA setting is met by 0 / 56 / 196 / 196 layers for
  g=4 / g=5 / v=2 / v=4 respectively. GPU timings are supplementary only (unfused Triton kernels on
  an L40S, 128-token prompt plus 32 decode steps).

## 2. Weight-quantisation sweep with INT8 activations

Dates 2026-10-03 to 2026-10-04. Fake quantisation in PyTorch on all 196 decoder linears, FP16
matmul. Perplexity: WikiText-2 test, 2 048-token contexts, full split. Downstream: lm-eval 0-shot
ARC-e, ARC-c, HellaSwag, PIQA, WinoGrande, LAMBADA (acc; "avg 6" is their mean) and 5-way MMLU for
1.7B. Calibration for SmoothQuant / AWQ / GPTQ / GPTVQ: 128 × 2 048 WikiText-2 train tokens,
sequential layer-by-layer.

Weight formats: `w8c` INT8 per-channel; `w4c` INT4 per-channel; `w4g` INT4 asymmetric, group 128;
`gptq` scalar GPTQ at `w4g`; `vq2_w4` GPTVQ with 2-D codewords, 256 centroids per 16 384 weights,
8-bit codebook entries (`cb6`, `cb5`: 6- and 5-bit entries so that g=4 / g=8 subset sums fit one
byte); `vq2_w3`, `vq4_w2`: 3- and 2-bit indices. Activation formats: `a8_tok` / `a4_tok` per-token;
`g128` per-128-channel groups; `rot` random orthogonal rotation then per-token.

**Operation counts for this section** assume the g=8 bit-plane table as the execution model, since
these rows do not run a lookup kernel themselves. With a full table, lookups = additions =
(planes/8) × MACs regardless of weight format: 1 409.3 M for A8 and 704.6 M for A4 on 1.7B
(440.4 M / 220.2 M on 0.6B). Multiplications are the activation quantisation plus output scales
(1.49 M on 1.7B, 0.89 M on 0.6B); group-128 weight scales add 11.0 M (3.4 M on 0.6B); VQ codebooks
add none because the table stores dequantised sums. A16 rows are dense FP16: 0 lookups,
1 409.3 M multiplications. Memory below is weight storage at the stated bits / weight; the full g=8
INT16 table would be 84 GiB on 1.7B (26 GiB on 0.6B), which is why Section 3 looks at partial
tables.

Two measured activation statistics are reported per row: **direct cost**, the mean number of
additions needed to form a (group, plane) subset sum without a table, `cost(p) = min(popcount(p),
8 − popcount(p) + 1)`, and **zero fraction** of quantised activation elements.

### Qwen3-1.7B

| Config | Lookups | Mults | Weight mem | PPL | avg 6 | MMLU | Direct cost | Zero frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0 | 1 409 M | 2 688 MiB | 16.67 | 0.595 | 0.557 | – | – |
| w8c_a8_tok | 1 409 M | 1.5 M | 1 344 MiB | 16.34 | 0.589 | 0.549 | 2.91 | 0.19 |
| sq_w8c_a8_tok | 1 409 M | 1.5 M | 1 344 MiB | 16.45 | 0.592 | 0.558 | 2.98 | 0.14 |
| w4g_a16 | 0 | 1 409 M | 698 MiB | 20.32 | 0.557 | 0.521 | – | – |
| w4c_a16 | 0 | 1 409 M | 672 MiB | 28.43 | 0.484 | 0.415 | – | – |
| w4g_a8_tok | 1 409 M | 12.5 M | 698 MiB | 20.26 | 0.545 | 0.509 | 2.91 | 0.19 |
| sq_w4g_a8_tok | 1 409 M | 12.5 M | 698 MiB | 20.34 | 0.554 | 0.518 | 2.99 | 0.14 |
| awq_w4g_a8_tok | 1 409 M | 12.5 M | 698 MiB | 18.90 | 0.564 | 0.502 | 3.02 | 0.13 |
| gptq_w4g_a8_tok | 1 409 M | 12.5 M | 698 MiB | 20.30 | 0.557 | 0.517 | 2.92 | 0.19 |
| vq2_w4_a8_tok | 1 409 M | 1.5 M | 714 MiB | 17.91 | 0.540 | 0.503 | 2.91 | 0.19 |
| vq2_w4_cb6_a8_tok | 1 409 M | 1.5 M | 704 MiB | 19.09 | 0.534 | 0.499 | 2.92 | 0.19 |
| vq2_w4_cb5_a8_tok | 1 409 M | 1.5 M | 699 MiB | 26.18 | 0.499 | 0.457 | 2.91 | 0.19 |
| vq2_w3_a8_tok | 1 409 M | 1.5 M | 515 MiB | 29.10 | 0.472 | 0.405 | 2.93 | 0.18 |
| vq4_w2_a8_tok | 1 409 M | 1.5 M | 420 MiB | 46.25 | 0.394 | 0.236 | 2.94 | 0.18 |
| sq_w4c_a8_tok | 1 409 M | 1.5 M | 672 MiB | 27.85 | 0.452 | 0.369 | 3.00 | 0.14 |
| awq_w4c_a8_tok | 1 409 M | 1.5 M | 672 MiB | 22.89 | 0.461 | 0.433 | 3.00 | 0.14 |
| w4g_a8_rot | 1 409 M | 12.5 M | 698 MiB | 27.58 | 0.509 | 0.350 | 3.27 | 0.01 |
| w4g_a4_tok | 705 M | 12.5 M | 698 MiB | 18 860 | 0.293 | 0.238 | 0.95 | 0.80 |
| sq_w4g_a4_tok | 705 M | 12.5 M | 698 MiB | 12 948 | 0.306 | 0.234 | 1.71 | 0.62 |
| w4g_a4_g128 | 705 M | 12.5 M | 698 MiB | 38.18 | 0.393 | 0.275 | 2.10 | 0.51 |
| w4g_a4_rot | 705 M | 12.5 M | 698 MiB | 731.75 | 0.359 | 0.255 | 3.10 | 0.21 |

Additions equal lookups for every table row. Rotation rows also need a d × d rotation per input
stream (not counted above; 0.7 G multiplications per token on 1.7B), so they are not candidates for
the table design and are kept only as an accuracy reference.

Per-task accuracies for 1.7B (acc, 0-shot):

| Config | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | LAMBADA |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 0.697 | 0.429 | 0.604 | 0.725 | 0.609 | 0.508 |
| w8c_a8_tok | 0.682 | 0.417 | 0.606 | 0.716 | 0.607 | 0.503 |
| awq_w4g_a8_tok | 0.657 | 0.385 | 0.584 | 0.707 | 0.589 | 0.466 |
| gptq_w4g_a8_tok | 0.655 | 0.393 | 0.569 | 0.697 | 0.587 | 0.438 |
| vq2_w4_a8_tok | 0.560 | 0.398 | 0.578 | 0.682 | 0.594 | 0.427 |
| vq2_w4_cb6_a8_tok | 0.561 | 0.392 | 0.572 | 0.670 | 0.605 | 0.404 |
| vq2_w4_cb5_a8_tok | 0.474 | 0.360 | 0.539 | 0.670 | 0.572 | 0.378 |

### Qwen3-0.6B

| Config | Lookups | Mults | Weight mem | PPL | avg 6 | Direct cost | Zero frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0 | 440 M | 840 MiB | 20.96 | 0.502 | – | – |
| w8c_a8_tok | 440 M | 0.9 M | 420 MiB | 21.36 | 0.493 | 3.09 | 0.13 |
| sq_w8c_a8_tok | 440 M | 0.9 M | 420 MiB | 21.24 | 0.497 | 3.15 | 0.09 |
| w4g_a16 | 0 | 440 M | 218 MiB | 25.71 | 0.453 | – | – |
| w4c_a16 | 0 | 440 M | 210 MiB | 48.23 | 0.395 | – | – |
| w4g_a8_tok | 440 M | 4.3 M | 218 MiB | 26.30 | 0.446 | 3.10 | 0.12 |
| sq_w4g_a8_tok | 440 M | 4.3 M | 218 MiB | 24.72 | 0.465 | 3.15 | 0.09 |
| sq_w4c_a8_tok | 440 M | 0.9 M | 210 MiB | 46.02 | 0.387 | 3.16 | 0.08 |
| w4g_a8_rot | 440 M | 4.3 M | 218 MiB | 28.05 | 0.449 | 3.27 | 0.01 |
| w4g_a4_tok | 220 M | 4.3 M | 218 MiB | 170 112 | 0.301 | 1.43 | 0.68 |
| w4g_a4_g128 | 220 M | 4.3 M | 218 MiB | 87.19 | 0.334 | 2.62 | 0.36 |
| w4g_a4_rot | 220 M | 4.3 M | 218 MiB | 58.08 | 0.370 | 3.11 | 0.20 |

Observations:

- INT8 per-token activations with INT8 per-channel weights are within 0.4 PPL and 1 point of
  average accuracy of FP16 on both models. This is the working floor for the table design.
- Per-token INT4 activations collapse on both models. The low "direct cost" of those rows is an
  artefact: 60–80 % of quantised activations are zero because a few outlier channels set the
  per-token scale. Group-128 scaling or rotation recovers part of the accuracy but not enough.
- Among 4-bit weight formats on 1.7B, GPTVQ 2-D codebooks give the best perplexity (17.91 vs 18.90
  for AWQ) but a lower task average (0.540 vs 0.564), with ARC-easy accounting for most of the gap.
  6-bit codebook entries cost about 1 PPL; 5-bit entries cost 8 PPL and 4 points of accuracy.
- Weight format has no measurable effect on the activation bit-pattern statistics, as expected.

## 3. Activation bit-pattern frequency and partial tables

Dates 2026-10-01 to 2026-10-03, Qwen3-0.6B. For each of the four distinct input streams per layer
(q/k/v input, o input, gate/up input, down input), activations are quantised per token to A bits,
split into bit planes, grouped g channels at a time, and the frequency of every g-bit index is
recorded per group. Fit set: about 1.0 M tokens of WikiText-2 train plus MMLU-Pro text; test set:
299 k tokens of WikiText-2 test. The question is whether a table that stores only hot entries, with
direct popcount computation as fallback, keeps most of the benefit of a full table.

Counts below are per (group, plane, output) unit, L, and in absolute per-token terms for 0.6B
(L = 440.4 M for A8 g=8, 220.2 M for A4 g=8). A table hit costs 1 lookup + 1 addition; a miss costs
`cost(p)` additions. Entries are ranked on the fit set and evaluated on the test set. Each kept
entry serves a pattern and its complement, so "top-k" means k of the 128 half-table entries per
(group, output). Table memory assumes INT16 entries and no index overhead.

### A8, g=8

| Table policy | Lookups / L | Adds / L | Lookups / token | Adds / token | Table memory |
|---|---:|---:|---:|---:|---:|
| Full table (128 entries) | 1.000 | 1.000 | 440.4 M | 440.4 M | 13 440 MiB |
| No table (direct popcount) | 0 | 3.034 | 0 | 1 336 M | 0 |
| Top-16 by frequency | 0.249 | 2.678 | 109.8 M | 1 179 M | 1 680 MiB |
| Top-16 by frequency × (cost − 1) | 0.192 | 2.551 | 84.5 M | 1 123 M | 1 680 MiB |
| Top-64 by frequency × (cost − 1) | 0.537 | 1.664 | 236.4 M | 732.8 M | 6 720 MiB |
| Entries with cost ≥ 4 (91 entries) | 0.657 | 1.304 | 289.2 M | 574.3 M | 9 555 MiB |

### A4, g=8 (accuracy of per-token A4 is not acceptable; included for the cost picture only)

| Table policy | Lookups / L | Adds / L | Lookups / token | Adds / token | Table memory |
|---|---:|---:|---:|---:|---:|
| Full table | 1.000 | 1.000 | 220.2 M | 220.2 M | 13 440 MiB |
| No table (direct popcount) | 0 | 0.835 | 0 | 183.9 M | 0 |
| Top-16 by frequency × (cost − 1) | 0.098 | 0.697 | 21.6 M | 153.5 M | 1 680 MiB |
| Entries with cost ≥ 2 (127 entries) | 0.439 | 0.439 | 96.7 M | 96.7 M | 13 335 MiB |

Observations:

- With INT8 activations the bit patterns are close to uniform: 16 hot half-entries out of 128 cover
  only 19 % of accesses, and the saving over direct computation is 0.5 additions per unit. Ranking
  by frequency × (cost − 1) is consistently better than ranking by frequency alone.
- The direct-compute cost of 3.0 additions per unit against 1 lookup + 1 addition for a full table
  frames the trade: a full g=8 table replaces about 2 additions per unit with one table read, at
  13 GiB for 0.6B. Where that is worthwhile depends on the gain-cell read energy and bandwidth
  relative to an adder, which is the hardware-side input this project needs.
- The A4 statistics are dominated by zeros and are not a usable design point until A4 accuracy is
  fixed (Section 2).

## 4. Earlier feasibility and LUT-LLM reproduction (June to August 2026)

These runs predate the current protocol and use small evaluation sets (a few hundred to a few
thousand scored tokens; 8–64 rows per task), so treat them as directional. Models are Qwen2.5-1.5B
and Qwen3-1.7B-Base; "all layers" means the same 196-linear scope.

| Run | Method | Lookups / token | Additions / token | Table / code memory | Result |
|---|---|---:|---:|---:|---|
| Qwen2.5-1.5B, PQ+LUT, subdim 32, Ka=Kw=16 | independent activation / weight PQ, FP16 tables | 40.9 M | 40.3 M | 7.8 MiB tables + 19.5 MiB codes | PPL 3.2 × 10^6 vs 19.9 FP16: unusable |
| Qwen2.5-1.5B, PQ+LUT, subdim 8, Ka=128, Kw=64, affine | as above with per-output affine correction | 163.8 M | ≈163.8 M | 994 MiB + 117 MiB | PPL 2 657 vs 24.8 FP16 |
| Qwen2.5-1.5B, LUT-LLM-shape PTQ, v=2, Ca=64, Kw=16, INT8 LUT | Chebyshev assignment, affine output | 655.1 M | ≈655.1 M | 2 499 MiB + 312 MiB | PPL 659 vs 24.8 FP16 (32 419 without affine) |
| Qwen3-1.7B-Base, LUT-LLM activation VQ, v=2, Ca=64, 1 000 STE steps | centres-only QAT, all 196 linears | 704.6 M | ≈704.6 M | 2 688 MiB INT8 LUT + 336 MiB codes (final-LUT form) | GLUE avg 71.0 vs 83.1 same-run FP16 (paper: 87.2); MMLU-Pro 7.8 vs 31.8 paper; PPL 336 vs 16.45 |
| Qwen3-1.7B-Base, same with v=4 | halves lookups | 352.3 M | ≈352.3 M | – | PPL 19 948: collapses |

Conclusion of that phase: LUT-LLM-shaped joint VQ needs the paper's full training stack (long
FineWeb/WikiQA QAT with fused STE kernels, trained-table reconstruction, GPTVQ) to reach its reported
accuracy; neither PTQ nor short QAT reproduces it with the public artifact. That motivated the
bit-plane direction in Sections 1–3, which is exact with respect to W8A8 and needs no training.

## 5. Current picture

| Design point (Qwen3-0.6B unless noted) | Lookups | Additions | Multiplications | On-chip memory | Accuracy |
|---|---:|---:|---:|---:|---|
| W8A8 dense | 0 | 440 M | 441 M | 420 MiB | ≈ FP16 |
| Bit-plane g=4 INT16, full table | 881 M | 881 M | 0.9 M | 3 361 MiB | = W8A8 |
| Bit-plane g=8, full table | 440 M | 440 M | 0.9 M | 13 440 MiB | = W8A8 |
| Bit-plane g=8, top-64 cost-aware partial table | 236 M | 733 M | 0.9 M | 6 720 MiB | = W8A8 |
| Joint VQ v=2 (LUT-LLM shape), PTQ | 220 M | 233 M | 0.3 M | 574 MiB | collapses |
| Joint VQ v=2, QAT (1.7B, our reproduction) | 705 M | 705 M | ≈0.6 M | 3 024 MiB | large loss |

Open items: a hardware cost for one gain-cell table read versus one INT16/INT32 addition, which
decides where on the bit-plane table-size curve to sit; an A8 activation format with fewer zeros and
smaller tables (per-group or outlier-aware scaling without rotation); and whether weight-side VQ
(Section 2) can be combined with bit-plane tables to shrink table memory without touching accuracy.

## Source directories (internal cluster)

- `~/lut_bitplane_comparison_20260930/`: Sections 1–3 (`results/summary.json`,
  `results/quant_bench/`, `results/freq/`).
- `~/workspace/TTA_clean/`: Section 4 (`RESULTS.md`, `PAPER_REPRO.md`).
