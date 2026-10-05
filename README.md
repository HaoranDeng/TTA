# TTA: LLM Inference Co-Design for 3D Gain-Cell Memory

This repository hosts the software side of a hardware–software co-design project. The hardware
premise is a new on-chip memory technology, the **3D gain cell**, which lets an accelerator carry
far more fast, local memory than SRAM allows today. The software question is: **how should LLM
inference be restructured when the accelerator has a very large on-chip memory?**

For most of the work here the hardware can be abstracted as **a GPU-like device with an SRAM-class
memory that is one to two orders of magnitude larger than current on-chip caches.** Compute
throughput and off-chip bandwidth are assumed to stay roughly where they are; what changes is how
much state can be kept close to the compute units.

## Why 3D gain cells

A gain cell is a small two- or three-transistor DRAM-like bit cell. Built with low-leakage
oxide-semiconductor transistors, it holds its state for long periods without the refresh pressure
of conventional DRAM, and because it is fabricated in back-end-of-line layers it can be stacked in
3D above the logic. Compared with 6T SRAM this gives:

- much higher bit density per unit of silicon area,
- logic-compatible integration, so the memory sits next to the datapath rather than behind an
  off-chip interface,
- non-destructive reads at latencies closer to SRAM than to DRAM.

Taken together, this makes a "big SRAM" design point realistic: tens to hundreds of megabytes of
on-chip memory on an accelerator die, instead of the few tens of megabytes available now.

## What changes for LLM inference

LLM decoding is dominated by memory traffic, not arithmetic. A large on-chip memory changes the
trade-offs in several places at once:

1. **Weights and KV cache on chip.** Small and medium models, or large portions of larger ones,
   can stay resident; off-chip bandwidth stops being the first bottleneck.
2. **Lookup tables replace arithmetic.** With abundant local memory, a matrix–vector product can be
   computed from precomputed partial sums or codebook tables instead of multiply–accumulate units.
   Table-based schemes trade memory capacity for compute, which is exactly the trade a gain-cell
   device wants to make.
3. **Capacity becomes a design knob.** Table size, codebook size, and quantisation precision can be
   chosen against an explicit memory budget rather than against a fixed cache.

## Research directions

The experiments in this project explore that design space from the software side.

- **Table-based linear layers.** Bit-plane subset-sum tables (grouping INT8 weights and looking up
  partial sums by activation bit patterns), joint activation–weight vector quantisation with
  precomputed dot-product tables, and LUT-LLM-style codebook inference. The core measurements are
  table size per layer, lookups and additions per token, and accuracy.
- **LUT-friendly weight compression.** Vector-quantised weights (GPTVQ-style 2-D and 4-D codebooks)
  with codebook entry widths chosen so that table partial sums stay within a fixed integer width,
  compared against scalar GPTQ, AWQ and SmoothQuant baselines under INT8 per-token activations.
- **Activation statistics and cost-aware tables.** Measuring the bit-pattern frequency of real
  activations and selecting which table entries to materialise given a capacity budget, so that
  rare patterns fall back to direct computation.
- **Co-design cost model.** Relating on-chip capacity, lookup count, accumulator width, and
  perplexity / downstream accuracy for Qwen-class models, to feed back into the memory
  organisation of the hardware.

## Evaluation

Models are evaluated with WikiText-2 perplexity and a standard zero-shot suite (ARC, HellaSwag,
PIQA, WinoGrande, LAMBADA, MMLU). Every table-based or quantised variant is reported together with
its on-chip storage footprint and per-token lookup and addition counts, so that accuracy can be read
against the memory it costs.

## Status

The project is in an early, exploratory phase. Experiment code and results are being consolidated
and will be added to this repository incrementally. GPU implementations here are functional
references for accuracy and table-size measurements; they are not performance models of the
gain-cell hardware itself.
