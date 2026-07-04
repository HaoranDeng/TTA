#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pq_lut_lm.eval_utils import evaluate_ppl, load_wikitext_texts, make_lm_batches
from pq_lut_lm.modeling import DEFAULT_TARGET_REGEX
from pq_lut_lm.paper_eval import evaluate_paper_tasks, make_paper_supervised_batches
from pq_lut_lm.w8a8_quant import W8A8Config, replace_with_w8a8_quant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--paper-samples", type=int, default=64)
    parser.add_argument("--eval-ppl", action="store_true")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--ppl-tokens", type=int, default=4096)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument("--prompt-style", choices=["plain", "chat"], default="plain")
    parser.add_argument("--prompt-template", choices=["simple", "instruction", "lm_eval"], default="instruction")
    parser.add_argument("--glue-shot-count", type=int, default=8)
    parser.add_argument("--mmlu-shot-count", type=int, default=0)
    parser.add_argument("--skip-squad", action="store_true")
    parser.add_argument("--eval-baseline", action="store_true")
    parser.add_argument("--eval-quantized", action="store_true")
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--weight-granularity", choices=["per_tensor", "per_channel", "per_group"], default="per_channel")
    parser.add_argument("--weight-group-size", type=int, default=128)
    parser.add_argument(
        "--activation-granularity",
        choices=["dynamic_per_token", "dynamic_per_tensor", "static_per_tensor", "static_per_feature"],
        default="dynamic_per_token",
    )
    parser.add_argument("--activation-percentile", type=float, default=1.0)
    parser.add_argument("--smoothquant-alpha", type=float, default=-1.0)
    parser.add_argument("--smoothquant-min-scale", type=float, default=1e-5)
    parser.add_argument("--smoothquant-max-scale", type=float, default=1e5)
    parser.add_argument("--calib-source", choices=["wikitext", "paper"], default="paper")
    parser.add_argument("--calib-tokens", type=int, default=4096)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--task-calib-samples", type=int, default=64)
    parser.add_argument("--calib-vectors-per-layer", type=int, default=1024)
    parser.add_argument("--target-regex", default=DEFAULT_TARGET_REGEX)
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--max-linears", type=int, default=None)
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


def make_calibration_batches(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, torch.Tensor]]:
    if args.calib_source == "paper":
        batches = make_paper_supervised_batches(
            tokenizer,
            max_samples_per_task=args.task_calib_samples,
            batch_size=1,
            max_length=args.seq_len,
            include_squad=not args.skip_squad,
            prompt_style=args.prompt_style,
            prompt_template=args.prompt_template,
        )
        random.Random(args.seed).shuffle(batches)
        return strip_labels(batches[: args.calib_batches])
    return make_lm_batches(
        tokenizer,
        load_wikitext_texts("train"),
        args.seq_len,
        args.calib_tokens,
        batch_size=1,
    )[: args.calib_batches]


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

    ppl_batches = None
    if args.eval_ppl:
        ppl_batches = make_lm_batches(
            tokenizer,
            load_wikitext_texts("test"),
            args.seq_len,
            args.ppl_tokens,
            batch_size=args.ppl_batch_size,
        )

    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(dtype),
        "load_seconds": load_seconds,
        "paper_samples": args.paper_samples,
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
            prompt_template=args.prompt_template,
            glue_shot_count=args.glue_shot_count,
            mmlu_shot_count=args.mmlu_shot_count,
        )
        save_json(out_dir / "summary.json", summary)
        if args.eval_ppl and ppl_batches is not None:
            print("Evaluating FP16 baseline perplexity", flush=True)
            summary["fp16_baseline"]["wikitext_ppl"] = evaluate_ppl(model, ppl_batches, device)
            save_json(out_dir / "summary.json", summary)

    calibration_batches = None
    if args.activation_granularity.startswith("static_") or args.smoothquant_alpha >= 0.0:
        print("Preparing activation calibration batches", flush=True)
        calibration_batches = make_calibration_batches(args, tokenizer)

    print("Applying W8A8 fake quantization", flush=True)
    qconfig = W8A8Config(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        weight_granularity=args.weight_granularity,
        weight_group_size=args.weight_group_size,
        activation_granularity=args.activation_granularity,
        activation_percentile=args.activation_percentile,
        smoothquant_alpha=args.smoothquant_alpha,
        smoothquant_min_scale=args.smoothquant_min_scale,
        smoothquant_max_scale=args.smoothquant_max_scale,
        target_regex=args.target_regex,
        include_lm_head=args.include_lm_head,
        max_linears=args.max_linears,
    )
    report = replace_with_w8a8_quant(
        model,
        qconfig,
        calibration_batches=calibration_batches,
        max_vectors_per_layer=args.calib_vectors_per_layer,
        device=device,
    )
    save_json(out_dir / "w8a8_hardware_stats.json", report)
    summary["w8a8_quantization"] = {
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "weight_granularity": args.weight_granularity,
        "weight_group_size": args.weight_group_size if args.weight_granularity == "per_group" else 0,
        "activation_granularity": args.activation_granularity,
        "activation_percentile": args.activation_percentile,
        "smoothquant_alpha": args.smoothquant_alpha,
        "hardware_aggregate": report["aggregate"],
        "calibration_seconds": report["calibration_seconds"],
        "quantization_seconds": report["quantization_seconds"],
    }
    save_json(out_dir / "summary.json", summary)

    if args.eval_quantized:
        print("Evaluating W8A8 quantized model on paper tasks", flush=True)
        summary["w8a8_quantized"] = evaluate_paper_tasks(
            model,
            tokenizer,
            device,
            args.paper_samples,
            include_squad=not args.skip_squad,
            prompt_style=args.prompt_style,
            prompt_template=args.prompt_template,
            glue_shot_count=args.glue_shot_count,
            mmlu_shot_count=args.mmlu_shot_count,
        )
        save_json(out_dir / "summary.json", summary)
        if args.eval_ppl and ppl_batches is not None:
            print("Evaluating W8A8 quantized perplexity", flush=True)
            summary["w8a8_quantized"]["wikitext_ppl"] = evaluate_ppl(model, ppl_batches, device)
            save_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
