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

Every method comparison keeps these six columns and includes **FP16 dense, the LUT-LLM baseline,
and the current method** together. Report FPGA operation costs or a clearly labelled reference
weighted score alongside them. State precision, memory placement, access width and reuse assumptions;
distinguish estimates from synthesis results and hardware measurements. Missing measurements remain
unmeasured.

## Results (historical, 256-token contexts)

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

## PQ validation and reference weighted cost (2026-10-06)

This validation covers **Qwen3-0.6B only**, with at most **two GPUs running concurrently**.
The accuracy requirement is a **relative WikiText-2 PPL increase of at most 3%** versus FP16
under the same evaluation protocol.

The current method is weight-only symmetric product quantisation: disjoint 2D weight vectors,
K=2048 shared representative centroids, sign/swap symmetry, 14-bit indices per weight pair,
FP16 scales per row and 128 weights, and per-token INT8 activations. Precomputed dot products
are stored as INT16 table entries. No activation VQ or online nearest-centroid search is used.

Counts below cover all 196 decoder linears per generated token; M means one million. Embedding,
LM head, attention QK/AV, normalisation and nonlinear operations are excluded equally. PPL uses
WikiText-2 raw test, 2048-token non-overlapping complete contexts and 298,862 scored tokens.
These PPL values must not be compared directly with the historical 256-token results above.

| Method | Lookups | Additions | Multiplications | On-chip memory | PPL | Reference weighted score (mJ/token) |
|---|---:|---:|---:|---:|---:|---:|
| FP16 dense | 0 | 440 M | 440 M | 840 MiB | 20.955169 | 0.748000 |
| LUT-LLM shape baseline, v=2 | 220.201 M | 233 M | 0.300 M | 574 MiB | Not evaluated under this protocol | 0.930354 |
| Symmetric PQ, K=2048 | 220.201 M | 220.086 M | 4.072 M | 502.237 MiB | 21.386942 (+2.06%) | 0.930091 |

The LUT-LLM row uses this project's joint activation–weight VQ resource baseline
(v=2, Ca=64, Cw=16, G=512). It is not a reproduction of the original paper's QAT accuracy.
The memory column retains the project's whole-model resident-storage convention, including
compressed weights, tables and scales; the PQ audit also includes peak linear scratch space.
It does not represent the streamed on-chip buffer capacity of a deployed FPGA.

PQ passes the 3% PPL requirement: its relative increase is 2.06046%, below the absolute PPL
limit of 21.583824. PPL was measured with reconstructed quantised weights as a numerical reference.
The actual compressed lookup kernel was checked separately on all 196 linears and a short-text
consistency run; the full PPL evaluation was not rerun through that kernel.

### Reference cost calculation

Use the author's workshop illustration: **3.8 pJ per lookup, 0.4 pJ per FP32 addition and
1.3 pJ per FP32 multiplication**. These are **7 nm reference values, not measured V80 FPGA
operation energies**. Here they are applied uniformly as a logical-operation score, even though
the methods use different precisions.

```text
E_ref (pJ/token) = 3.8 * N_lookup + 0.4 * N_add + 1.3 * N_mul
E_ref (mJ/token) = E_ref (pJ/token) / 1,000,000,000

FP16:    (0 + 0.4*440,000,000 + 1.3*440,000,000) / 1e9 = 0.748000000
LUT-LLM: (3.8*220,200,960 + 0.4*233,000,000 + 1.3*300,000) / 1e9
         = 0.930353648
PQ:      (3.8*220,200,960 + 0.4*220,086,272 + 1.3*4,071,620) / 1e9
         = 0.836763648 + 0.088034509 + 0.005293106
         = 0.930091263
```

PQ also performs 286,916 reciprocals per token, listed separately from multiplications. If each
reciprocal is provisionally charged at the multiplication coefficient, add 0.000372991 mJ/token:
the PQ score becomes **0.930464254 mJ/token**, approximately **0.012% above LUT-LLM**.
Without that allowance it is approximately 0.028% below LUT-LLM. The baseline arithmetic counts
are rounded, so both differences should be interpreted as **effectively equal**, not as an energy
advantage. The reciprocal allowance is an assumption, not a measured cost or an energy upper bound.

Lookup contributes about 90% of the PQ score. Fewer additions save about 0.005165 mJ/token,
while extra multiplications add about 0.004903 mJ/token, almost cancelling the saving.

This score is **not a ranking of actual FPGA energy**. In particular, dense weight/activation reads
are not included, and the common coefficients do not account for operation precision, table size,
physical access width, row reuse, routing or static power. LUT-LLM can read a row and reuse it through
multiplexers, so one logical lookup need not equal one SRAM transaction. FPGA energy comparisons
require a matched implementation and memory-access model or hardware measurements.

Sources:

- [LUT-LLM paper, Sections II-C and IV-C](https://arxiv.org/html/2511.06174v2)
- [Jason Cong workshop slides, page 8: reference energy illustration](https://publish.illinois.edu/ai-hw-workshop/files/2026/01/jason_cong.pdf#page=8)
- [Jouppi et al., ISCA 2021, Table 2: underlying 7 nm operation energies](https://parsa.epfl.ch/course-info/cs723/papers/tpuv4i.pdf#page=3)
  (The source's SRAM energies are per 64-bit access.)

## Cluster usage limits

- Per user, use at most **2 GPUs total across scai3–5** and **2 GPUs total across scai6–7**, counting all projects and jobs.
- Before exceeding either limit, post the **reason, GPU count and server(s), and duration** in `#scai-servers`. Unannounced excess jobs may be killed without warning; exceptions are handled case by case.
- Connect through VPN. Check CPU/GPU usage with `htop` and `nvidia-smi` before launching; never use a GPU occupied by another user.
- Select GPUs explicitly with `CUDA_VISIBLE_DEVICES` and limit CPU threads, e.g. `OMP_NUM_THREADS=4`. Run CPU-heavy jobs on scai1–5, not scai6–7.
- Consolidate jobs that each use **less than 50% of GPU memory** onto one GPU, accounting for peak memory needs.
- Monitor running jobs regularly and release memory when finished, including Jupyter kernels.
