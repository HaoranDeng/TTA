#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pq_lut_lm.eval_utils import (
    evaluate_mmlu_zero_shot,
    evaluate_ppl,
    load_mmlu_rows,
    load_wikitext_texts,
    make_lm_batches,
)
from pq_lut_lm.modeling import DEFAULT_TARGET_REGEX, quantize_model_linears
from pq_lut_lm.pq_linear import PQConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--ppl-tokens", type=int, default=4096)
    parser.add_argument("--calib-tokens", type=int, default=2048)
    parser.add_argument("--calib-batches", type=int, default=4)
    parser.add_argument("--calib-vectors-per-layer", type=int, default=1024)
    parser.add_argument("--mmlu-samples", type=int, default=64)
    parser.add_argument("--mmlu-subject", default="all")
    parser.add_argument("--target-regex", default=DEFAULT_TARGET_REGEX)
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--max-linears", type=int, default=None)
    parser.add_argument("--method", choices=["pq", "lutllm"], default="pq")
    parser.add_argument("--subdim", type=int, default=None)
    parser.add_argument("--ka", type=int, default=None)
    parser.add_argument("--kw", type=int, default=None)
    parser.add_argument("--kmeans-iters", type=int, default=4)
    parser.add_argument("--sample-limit", type=int, default=2048)
    parser.add_argument("--encode-chunk", type=int, default=8192)
    parser.add_argument("--lut-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--lut-storage", choices=["expanded", "compact"], default="expanded")
    parser.add_argument("--distance", choices=["l2", "chebyshev"], default=None)
    parser.add_argument("--weight-group-size", type=int, default=None)
    parser.add_argument("--lut-quant-bits", type=int, default=None)
    parser.add_argument("--weight-code-reassign-iters", type=int, default=0)
    parser.add_argument("--output-correction", choices=["none", "bias", "affine"], default="none")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-pq", action="store_true")
    args = parser.parse_args()
    if args.method == "lutllm":
        if args.subdim is None:
            args.subdim = 2
        if args.ka is None:
            args.ka = 64
        if args.kw is None:
            args.kw = 16
        if args.distance is None:
            args.distance = "chebyshev"
        if args.weight_group_size is None:
            args.weight_group_size = 256
        if args.lut_quant_bits is None:
            args.lut_quant_bits = 8
    else:
        if args.subdim is None:
            args.subdim = 32
        if args.ka is None:
            args.ka = 8
        if args.kw is None:
            args.kw = 16
        if args.distance is None:
            args.distance = "l2"
        if args.weight_group_size is None:
            args.weight_group_size = 0
        if args.lut_quant_bits is None:
            args.lut_quant_bits = 0
    return args


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def dtype_from_arg(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", vars(args))

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_arg(args.dtype)
    if device.type == "cpu":
        dtype = torch.float32

    print(f"Loading tokenizer: {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_id}", flush=True)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_start

    texts = load_wikitext_texts("test")
    ppl_batches = make_lm_batches(
        tokenizer,
        texts,
        seq_len=args.seq_len,
        max_tokens=args.ppl_tokens,
        batch_size=args.eval_batch_size,
    )
    calib_batches = make_lm_batches(
        tokenizer,
        texts,
        seq_len=args.seq_len,
        max_tokens=args.calib_tokens,
        batch_size=1,
    )[: args.calib_batches]
    mmlu_rows = load_mmlu_rows(args.mmlu_samples, subject=args.mmlu_subject, split="test")

    print("Evaluating baseline perplexity", flush=True)
    baseline_ppl = evaluate_ppl(model, ppl_batches, device)
    print("Evaluating baseline MMLU", flush=True)
    baseline_mmlu = evaluate_mmlu_zero_shot(model, tokenizer, mmlu_rows, device)
    baseline_mmlu_public = {k: v for k, v in baseline_mmlu.items() if k != "predictions"}

    report = None
    pq_ppl = None
    pq_mmlu_public = None
    if not args.skip_pq:
        pq_config = PQConfig(
            method=args.method,
            subdim=args.subdim,
            ka=args.ka,
            kw=args.kw,
            kmeans_iters=args.kmeans_iters,
            sample_limit=args.sample_limit,
            encode_chunk=args.encode_chunk,
            lut_dtype=args.lut_dtype,
            lut_storage=args.lut_storage,
            distance=args.distance,
            weight_group_size=args.weight_group_size,
            lut_quant_bits=args.lut_quant_bits,
            weight_code_reassign_iters=args.weight_code_reassign_iters,
            output_correction=args.output_correction,
            seed=args.seed,
        )
        print(f"Calibrating and replacing Linear layers with {args.method} LUT modules", flush=True)
        report = quantize_model_linears(
            model,
            calib_batches,
            pq_config,
            target_regex=args.target_regex,
            include_lm_head=args.include_lm_head,
            max_linears=args.max_linears,
            max_vectors_per_layer=args.calib_vectors_per_layer,
            device=device,
        )
        save_json(out_dir / "hardware_stats.json", {
            "modules": report.module_stats,
            "aggregate": report.aggregate,
            "calibration_seconds": report.calibration_seconds,
            "quantization_seconds": report.quantization_seconds,
        })

        print("Evaluating PQ+LUT perplexity", flush=True)
        pq_ppl = evaluate_ppl(model, ppl_batches, device)
        print("Evaluating PQ+LUT MMLU", flush=True)
        pq_mmlu = evaluate_mmlu_zero_shot(model, tokenizer, mmlu_rows, device)
        pq_mmlu_public = {k: v for k, v in pq_mmlu.items() if k != "predictions"}

    summary = {
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(dtype),
        "load_seconds": load_seconds,
        "baseline": {
            "ppl": baseline_ppl,
            "mmlu_zero_shot": baseline_mmlu_public,
        },
        "pq_lut": None if args.skip_pq else {
            "ppl": pq_ppl,
            "mmlu_zero_shot": pq_mmlu_public,
            "hardware_aggregate": report.aggregate if report else None,
            "calibration_seconds": report.calibration_seconds if report else None,
            "quantization_seconds": report.quantization_seconds if report else None,
        },
    }
    save_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
