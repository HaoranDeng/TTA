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

Code and results will be added as the project progresses.
