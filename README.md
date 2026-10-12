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

## Experimental results

The results are organized into two comparisons. **Table 1 uses Qwen3-1.7B with 2048-token
contexts; Table 2 uses historical Qwen3-0.6B results with 256-token contexts. Their PPL values
must not be compared across tables.** Earlier rounds and detailed audits are linked below.

Operation counts are logical estimates per generated token across all 196 decoder linears;
M = 10^6. Embeddings, LM head, attention QK/AV, normalization and nonlinearities are excluded
consistently. Memory assumes resident tables and compressed representations; the historical
table-only estimates are explicitly labeled.

### Table 1. Comparison with PQ — Qwen3-1.7B

Round 21, pure PTQ with no extra NLL adaptation. Each family's configuration was selected on
30 document-monitor articles and frozen before final accuracy evaluation. These are the
selected configurations, not a selection of the best TEST or MMLU scores.

| Method | Lookups (M/token) | Additions (M/token) | Multiplications (M/token) | On-chip memory (MiB) | PPL ↓ | MMLU ↑ | E_ref (mJ/token) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 dense | 0.000 | 1408.713 | 1409.286 | 2688.000 | 16.669954 | 60.348% (reused) | 2.395557 |
| LUT-LLM shape resource baseline | 704.643 | 726.663 | 0.573 | 1764.000 | Unmeasured | Unmeasured | 2.969054 |
| PQ mixed g64/g128 | 704.643 | 705.053 | 13.836 | 1273.157 | 16.469460 | 59.614% | 2.977652 |
| RQ mixed g64/g128 | 704.643 | 705.151 | 14.541 | 1274.501 | 16.600685 | 59.215% | 2.978607 |
| IB mixed g64/g128 | 704.643 | 705.610 | 21.095 | 1287.001 | 16.374142 | 59.201% | 2.987311 |

- **Accuracy:** WikiText-2 raw TEST, nonoverlapping 2048-token contexts, 298,862 scored tokens.
  MMLU uses all 57 subjects and 14,042 questions, five development examples per subject,
  raw A/B/C/D continuation likelihood, no chat template or generated reasoning. FP16 MMLU is
  reused from round 20; PQ/RQ/IB MMLU was measured in round 21. Historical TEST and MMLU have
  been viewed repeatedly; the separate 30-article audit does not establish full decontamination.
- **Representation:** W8A9, 480 deployed centers, 12-bit indices, shared resident INT32 tables,
  FP16 scales and FP32 accumulation. Memory includes indices, tables, codebooks, scales and
  peak single-token scratch. RQ combines 15+32 residual components offline; IB uses a local
  distortion-plus-entropy objective. LUT-LLM is a same-shape resource reference with accuracy unmeasured.
- **Cost:** all three quantized rows pass the lookup, arithmetic and memory gates. They also
  perform 516,292 reciprocals/token, included in the arithmetic gate but listed separately from
  multiplications and the displayed E_ref. Charging each reciprocal at the multiplication
  coefficient would add 0.000671180 mJ/token.

[Full round-21 results and fixed-group controls](pq_exploration/study_1p7b_20261008_mixed21/REPORT21_ZH.txt)
· [Raw results](pq_exploration/study_1p7b_20261008_mixed21/round21_summary.json)
· [Frozen selections](pq_exploration/study_1p7b_20261008_mixed21/overfit21_frozen.json)
· [Protocol](pq_exploration/study_1p7b_20261008_mixed21/protocol21.json)

#### Additional benchmarks — Qwen3-1.7B, frozen round-21 models

Completed on **2026-10-10 01:17 UTC (2026-10-09 18:17 PDT)**: **32/32 model-task results,
81,170 questions per model, 324,680 predictions**. These are the same original FP16 and frozen
PQ/RQ/IB mixed configurations in Table 1. The new scores did not tune or select models.
PPL and ordinary MMLU in Table 1 **reuse round 21**, rather than being newly measured here.

**GLUE six-task subset:** zero-shot, complete public validation splits; MNLI uses matched validation.
All scores below are percentages. MRPC and QQP report accuracy / positive-class F1.
No official complete-GLUE aggregate is claimed.

| Method | MNLI | MRPC acc / F1 | QNLI | QQP acc / F1 | RTE | SST-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 48.130 | 59.069 / 69.801 | 51.107 | 63.421 / 65.893 | 70.397 | 85.436 |
| PQ | 45.257 | 65.196 / 76.412 | 52.590 | 77.047 / 69.721 | 68.953 | 84.174 |
| RQ | 46.215 | 63.725 / 75.251 | 51.290 | 61.996 / 65.412 | 75.090 | 85.665 |
| IB | 45.461 | 65.441 / 75.044 | 53.029 | 76.332 / 69.211 | 71.841 | 80.849 |

Question counts per model: MNLI 9,815; MRPC 408; QNLI 5,463; QQP 40,430; RTE 277; SST-2 872.

**SQuAD v2:** 11,873 validation questions, zero-shot raw prompt, greedy generation with newline/EOS stopping and a 256-token limit. The no-answer threshold was fixed at 1.0 before evaluation; no best-threshold fitting on test labels. Deltas are percentage points versus FP16.

| Method | EM | F1 | EM delta (pp) | F1 delta (pp) |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 13.762 | 23.997 | +0.000 | +0.000 |
| PQ | 12.886 | 23.679 | -0.876 | -0.318 |
| RQ | 12.145 | 22.808 | -1.617 | -1.189 |
| IB | 12.735 | 23.514 | -1.028 | -0.483 |

Absolute SQuAD scores are low in this fixed setup: FP16 achieves only 0.101% EM on the 5,945 unanswerable questions. This common rejection failure must not be attributed solely to quantization or compared directly with the paper's different-protocol SQuAD score.

**MMLU-Pro:** all 12,032 test questions; five-shot `leaderboard_mmlu_pro`, first validation examples and direct option likelihood, micro accuracy, no generated CoT or chat template. It is distinct from ordinary MMLU.

| Method | Correct / total | Accuracy | Delta vs FP16 (pp) |
| --- | ---: | ---: | ---: |
| FP16 | 3,868 / 12,032 | 32.148% | +0.000 |
| PQ | 3,813 / 12,032 | 31.690% | -0.457 |
| RQ | 3,832 / 12,032 | 31.848% | -0.299 |
| IB | 3,816 / 12,032 | 31.715% | -0.432 |

Relative to FP16, PQ/RQ/IB PPL changes are **-1.203% / -0.416% / -1.775%**, all within
the maximum **+3% PPL** gate. Resident memory decreases **52.636% / 52.586% / 52.120%**,
while the logical E_ref proxy increases **24.299% / 24.339% / 24.702%**. GLUE changes vary by task;
ordinary MMLU, MMLU-Pro and SQuAD EM/F1 decrease for all three quantized models. Lower PPL alone
does not establish overall quality improvement or elimination of overfitting. Historical WikiText
and MMLU were repeatedly inspected; semantic duplicates and pretraining contamination were not removed.
LUT-LLM paper scores are separately labeled different-protocol references in the report; no matching
LUT-LLM accuracy was measured here. E_ref is **not measured FPGA energy**.

[Final Chinese report](pq_exploration/benchmarks_1p7b_20261008/BENCHMARK_REPORT_ZH.txt)
· [Complete metrics and subject breakdowns](pq_exploration/benchmarks_1p7b_20261008/benchmark_summary.json)
· [Frozen protocol](pq_exploration/benchmarks_1p7b_20261008/protocol.json)
· [Request-binding audit](pq_exploration/benchmarks_1p7b_20261008/request_binding_audit.json)
· [Source/model/data provenance audit](pq_exploration/benchmarks_1p7b_20261008/final_provenance_audit.json)
· [Publication and evidence scope](pq_exploration/benchmarks_1p7b_20261008/PUBLICATION_NOTES.txt)

Full aggregation and independent request binding verified all 32 results and 324,680 predictions.
The retained evidence contains all 2,584 prediction chunks and eight frozen request tensors;
2,740 local files were hash-checked. This Git publication contains the reports, task-result metadata,
frozen protocol and audit summaries. Full predictions, request tensors, model weights and runtime
logs remain in the experiment archive identified in the publication notes.

### Table 2. Comparison with bit-plane tables — Qwen3-0.6B

Historical W8A8 bit-plane and joint-VQ experiments. Measured PPL uses WikiText-2 test,
nonoverlapping 256-token contexts and 8,160 scored tokens. No QAT was performed.

| Method | Lookups (M/token) | Additions (M/token) | Multiplications (M/token) | On-chip memory (MiB) | PPL ↓ | MMLU ↑ | E_ref (mJ/token) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 dense | 0.0 | 440.4 | 440.4 | 840 (weights) | 42.51 | Unmeasured | ≈0.749 |
| W8A8 dense | 0.0 | 440.4 | 441.3 | 420 (weights) | 43.90 | Unmeasured | ≈0.750 |
| Bit-plane g=4, INT16 | 880.8 | 880.8 | 0.9 | 3,361 | 43.90 | Unmeasured | ≈3.701 |
| Bit-plane g=5, INT8 | 705.3 | 705.3 | 706.2 | 3,028 | 43.69 | Unmeasured | ≈3.880 |
| Bit-plane g=8, full half-table | 440.4 | 440.4 | 0.9 | 13,440 (table only) | 43.90 (ref.) | Unmeasured | ≈1.851 |
| Bit-plane g=8, top-64 + exact fallback | 236.4 | 732.8 | 0.9 | 6,720 (table only) | 43.90 (ref.) | Unmeasured | ≈1.193 |
| LUT-LLM shape joint VQ, v=2 (PTQ) | 220.2 | 233.0 | 0.3 | 574 | 1.2 × 10^7 | Unmeasured | ≈0.930 |
| LUT-LLM shape joint VQ, v=4 (PTQ) | 110.1 | 123.0 | 0.3 | 312 | 5.1 × 10^8 | Unmeasured | ≈0.468 |

- **Measured versus estimated:** g=4/g=5 and joint-VQ accuracy comes from the historical
  kernel evaluations. The g=8 rows describe full/partial-table execution costs; “ref.” carries
  over the W8A8 reference PPL because the design retains exact sums and exact fallback.
  It is not a separate g=8 end-to-end PPL measurement. MMLU was not measured in this experiment.
- **Storage and access:** g=4 uses INT16 entries; g=5 uses INT8 entries plus FP32 table scales,
  causing an extra multiplication per lookup. The g=8 figures use INT16 half-tables with
  complement symmetry (128 entries for full, 64 for top-64); their memory covers table entries
  only and excludes index/metadata overhead. Top-64 entries are ranked by frequency × (cost − 1).
  Its lookup/addition counts come from the separate bit-pattern study on about 299k TEST tokens.
- **LUT-LLM reference:** joint VQ uses 64 activation centers, 16 weight centers shared by
  512 outputs and INT8 dot tables. The very high PPL values are measured failures of this PTQ
  proxy; they do not reproduce the paper's QAT accuracy.
- **E_ref:** approximate values are calculated from the rounded operation counts shown above.

[Historical measurements and bit-pattern cost model](https://github.com/HaoranDeng/TTA/blob/dc945f5/RESULTS.md)

### Reference cost and detailed records

`E_ref = (3.8 × lookups + 0.4 × additions + 1.3 × multiplications) / 10^9` mJ/token,
using operation counts rather than counts in millions. This is a uniform 7 nm logical reference
score, **not measured FPGA energy or a hardware energy ranking**. A lookup denotes one logical
table entry; physical access width, row reuse, metadata reads, dense operand reads and routing
are not modeled. Different entry widths and reuse patterns can change actual memory traffic.
[Coefficient sources and detailed assumptions](https://github.com/HaoranDeng/TTA/blob/0b4b48462496e5f9b1af19f18376fc1beabfadf1/README.md#reference-cost-calculation).

Earlier experiments remain available in the
[round-18 0.6B report](pq_exploration/study_20261006/PHASE18_REPORT_ZH.txt),
[round-19 1.7B transfer report](pq_exploration/study_1p7b_20261007_a100/REPORT19_ZH.txt)
and [round-20 capacity report](pq_exploration/study_1p7b_20261008_ops20/REPORT20_ZH.txt).
These reports keep pure PTQ separate from NLL adaptation and its matched FP16 controls.
[Round 22](pq_exploration/study_1p7b_20261008_types22/REPORT22_PRELIMINARY_ZH.txt) has numerical
verification only; its new holdout PPL and MMLU remain unmeasured.
The [README before this reorganization](https://github.com/HaoranDeng/TTA/blob/0b4b48462496e5f9b1af19f18376fc1beabfadf1/README.md)
preserves the complete earlier presentation.

## Cluster usage limits

- Per user, use at most **2 GPUs total across scai3–5** and **2 GPUs total across scai6–7**, counting all projects and jobs.
- Before exceeding either limit, post the **reason, GPU count and server(s), and duration** in `#scai-servers`. Unannounced excess jobs may be killed without warning; exceptions are handled case by case.
- Connect through VPN. Check CPU/GPU usage with `htop` and `nvidia-smi` before launching; never use a GPU occupied by another user.
- Select GPUs explicitly with `CUDA_VISIBLE_DEVICES` and limit CPU threads, e.g. `OMP_NUM_THREADS=4`. Run CPU-heavy jobs on scai1–5, not scai6–7.
- Consolidate jobs that each use **less than 50% of GPU memory** onto one GPU, accounting for peak memory needs.
- Monitor running jobs regularly and release memory when finished, including Jupyter kernels.
