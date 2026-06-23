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

from pq_lut_lm.activation_quant import (
    convert_activation_lut_to_pq_lut,
    replace_with_fitted_activation_lut,
)
from pq_lut_lm.eval_utils import load_wikitext_texts, make_lm_batches
from pq_lut_lm.modeling import DEFAULT_TARGET_REGEX
from pq_lut_lm.paper_eval import evaluate_paper_tasks, make_paper_supervised_batches
from pq_lut_lm.pq_linear import PQConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--calib-source", choices=["wikitext", "paper"], default="paper")
    parser.add_argument("--calib-tokens", type=int, default=4096)
    parser.add_argument("--task-calib-samples", type=int, default=64)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--calib-vectors-per-layer", type=int, default=256)
    parser.add_argument("--fit-steps", type=int, default=50)
    parser.add_argument("--fit-lr", type=float, default=1e-2)
    parser.add_argument("--fit-batch-size", type=int, default=128)
    parser.add_argument("--fit-lut-dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--paper-samples", type=int, default=32)
    parser.add_argument("--prompt-style", choices=["plain", "chat"], default="plain")
    parser.add_argument("--skip-squad", action="store_true")
    parser.add_argument("--target-regex", default=DEFAULT_TARGET_REGEX)
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--max-linears", type=int, default=None)
    parser.add_argument("--subdim", type=int, default=2)
    parser.add_argument("--ka", type=int, default=64)
    parser.add_argument("--kw", type=int, default=16)
    parser.add_argument("--kmeans-iters", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=256)
    parser.add_argument("--encode-chunk", type=int, default=8192)
    parser.add_argument("--distance", choices=["l2", "chebyshev"], default="chebyshev")
    parser.add_argument("--weight-group-size", type=int, default=256)
    parser.add_argument("--lut-quant-bits", type=int, default=4)
    parser.add_argument("--lut-storage", choices=["expanded", "compact"], default="compact")
    parser.add_argument("--output-correction", choices=["none", "bias", "affine"], default="none")
    parser.add_argument("--eval-baseline", action="store_true")
    parser.add_argument("--eval-act-lut", action="store_true")
    parser.add_argument("--eval-final-lut", action="store_true")
    parser.add_argument("--skip-final-lut", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def dtype_from_arg(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def strip_labels(batches: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{k: v for k, v in batch.items() if k != "labels"} for batch in batches]


def select_batches(batches: list[dict[str, torch.Tensor]], count: int) -> list[dict[str, torch.Tensor]]:
    if count <= 0:
        return batches
    return batches[:count]


def make_calibration_batches(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, torch.Tensor]]:
    if args.calib_source == "paper":
        batches = make_paper_supervised_batches(
            tokenizer,
            max_samples_per_task=args.task_calib_samples,
            batch_size=1,
            max_length=args.seq_len,
            include_squad=not args.skip_squad,
            prompt_style=args.prompt_style,
        )
        return strip_labels(select_batches(batches, args.calib_batches))
    texts = load_wikitext_texts("train")
    batches = make_lm_batches(tokenizer, texts, args.seq_len, args.calib_tokens, batch_size=1)
    return select_batches(batches, args.calib_batches)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", vars(args))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_arg(args.dtype)
    if device.type == "cpu":
        dtype = torch.float32
    fit_lut_dtype = dtype_from_arg(args.fit_lut_dtype)

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

    calib_batches = make_calibration_batches(args, tokenizer)
    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(dtype),
        "load_seconds": load_seconds,
        "paper_samples": args.paper_samples,
        "inferred_lutllm_path": (
            "fit activation lookup-table values locally, reconstruct dense weights from "
            "trained tables with least squares, then apply activation-weight VQ"
        ),
    }

    if args.eval_baseline:
        print("Evaluating FP16 baseline on paper tasks", flush=True)
        summary["fp16_baseline"] = evaluate_paper_tasks(
            model,
            tokenizer,
            device,
            args.paper_samples,
            include_squad=not args.skip_squad,
            prompt_style=args.prompt_style,
        )
        save_json(out_dir / "summary.json", summary)

    print("Freezing dense model weights", flush=True)
    for param in model.parameters():
        param.requires_grad_(False)

    config = PQConfig(
        method="act_lut_fit",
        subdim=args.subdim,
        ka=args.ka,
        kw=args.kw,
        kmeans_iters=args.kmeans_iters,
        sample_limit=args.sample_limit,
        encode_chunk=args.encode_chunk,
        lut_dtype="float16",
        lut_storage=args.lut_storage,
        distance=args.distance,
        weight_group_size=args.weight_group_size,
        lut_quant_bits=args.lut_quant_bits,
        output_correction=args.output_correction,
        seed=args.seed,
    )

    print("Fitting activation lookup tables layerwise", flush=True)
    act_report = replace_with_fitted_activation_lut(
        model,
        calib_batches,
        config,
        target_regex=args.target_regex,
        include_lm_head=args.include_lm_head,
        max_linears=args.max_linears,
        max_vectors_per_layer=args.calib_vectors_per_layer,
        fit_steps=args.fit_steps,
        fit_lr=args.fit_lr,
        fit_batch_size=args.fit_batch_size,
        fit_lut_dtype=fit_lut_dtype,
        device=device,
    )
    summary["act_lut_fit"] = {
        "hardware_aggregate": act_report.aggregate,
        "calibration_seconds": act_report.calibration_seconds,
        "fit_seconds": act_report.quantization_seconds,
    }
    save_json(out_dir / "act_lut_fit_hardware_stats.json", {
        "modules": act_report.module_stats,
        "aggregate": act_report.aggregate,
        "calibration_seconds": act_report.calibration_seconds,
        "fit_seconds": act_report.quantization_seconds,
    })
    save_json(out_dir / "summary.json", summary)

    if args.eval_act_lut:
        print("Evaluating direct activation-LUT model on paper tasks", flush=True)
        summary["act_lut_fit"]["paper_eval"] = evaluate_paper_tasks(
            model,
            tokenizer,
            device,
            args.paper_samples,
            include_squad=not args.skip_squad,
            prompt_style=args.prompt_style,
        )
        save_json(out_dir / "summary.json", summary)

    if args.skip_final_lut:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    if device.type == "cuda":
        torch.cuda.empty_cache()

    config.method = "act_lut_reconstructed_weight_vq"
    print("Reconstructing weights from trained LUTs and converting to final activation-weight LUT", flush=True)
    final_report = convert_activation_lut_to_pq_lut(
        model,
        calib_batches,
        config,
        max_linears=args.max_linears,
        max_vectors_per_layer=args.calib_vectors_per_layer,
        device=device,
    )
    summary["reconstructed_final_lut"] = {
        "hardware_aggregate": final_report.aggregate,
        "calibration_seconds": final_report.calibration_seconds,
        "quantization_seconds": final_report.quantization_seconds,
    }
    save_json(out_dir / "final_lut_hardware_stats.json", {
        "modules": final_report.module_stats,
        "aggregate": final_report.aggregate,
        "calibration_seconds": final_report.calibration_seconds,
        "quantization_seconds": final_report.quantization_seconds,
    })
    save_json(out_dir / "summary.json", summary)

    if args.eval_final_lut:
        print("Evaluating reconstructed final LUT on paper tasks", flush=True)
        summary["reconstructed_final_lut"]["paper_eval"] = evaluate_paper_tasks(
            model,
            tokenizer,
            device,
            args.paper_samples,
            include_squad=not args.skip_squad,
            prompt_style=args.prompt_style,
        )
        save_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
