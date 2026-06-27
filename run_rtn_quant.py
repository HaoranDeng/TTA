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

from pq_lut_lm.eval_utils import evaluate_ppl, load_wikitext_texts, make_lm_batches
from pq_lut_lm.modeling import DEFAULT_TARGET_REGEX
from pq_lut_lm.paper_eval import evaluate_paper_tasks
from pq_lut_lm.rtn_quant import RTNConfig, replace_with_rtn_quant


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
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--granularity", choices=["per_tensor", "per_channel", "per_group"], default="per_channel")
    parser.add_argument("--group-size", type=int, default=128)
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

    print("Applying RTN weight-only quantization", flush=True)
    qconfig = RTNConfig(
        bits=args.bits,
        granularity=args.granularity,
        group_size=args.group_size,
        target_regex=args.target_regex,
        include_lm_head=args.include_lm_head,
        max_linears=args.max_linears,
    )
    report = replace_with_rtn_quant(model, qconfig)
    save_json(out_dir / "rtn_hardware_stats.json", report)
    summary["rtn_quantization"] = {
        "bits": args.bits,
        "granularity": args.granularity,
        "group_size": args.group_size if args.granularity == "per_group" else 0,
        "hardware_aggregate": report["aggregate"],
        "quantization_seconds": report["quantization_seconds"],
    }
    save_json(out_dir / "summary.json", summary)

    if args.eval_quantized:
        print("Evaluating RTN quantized model on paper tasks", flush=True)
        summary["rtn_quantized"] = evaluate_paper_tasks(
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
            print("Evaluating RTN quantized perplexity", flush=True)
            summary["rtn_quantized"]["wikitext_ppl"] = evaluate_ppl(model, ppl_batches, device)
            save_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
