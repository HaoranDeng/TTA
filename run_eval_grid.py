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

from pq_lut_lm.paper_eval import evaluate_paper_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--paper-samples", type=int, default=64)
    parser.add_argument("--skip-squad", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--variants",
        default="simple:0:0:plain,instruction:0:0:plain,instruction:3:3:plain,instruction:5:5:plain",
        help="Comma-separated prompt_template:glue_shots:mmlu_shots:prompt_style specs.",
    )
    return parser.parse_args()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def dtype_from_arg(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def parse_variant(spec: str) -> dict[str, Any]:
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(f"Variant must be prompt_template:glue_shots:mmlu_shots:prompt_style, got {spec!r}")
    prompt_template, glue_shots, mmlu_shots, prompt_style = parts
    return {
        "name": f"{prompt_template}_g{glue_shots}_m{mmlu_shots}_{prompt_style}",
        "prompt_template": prompt_template,
        "glue_shot_count": int(glue_shots),
        "mmlu_shot_count": int(mmlu_shots),
        "prompt_style": prompt_style,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [parse_variant(spec.strip()) for spec in args.variants.split(",") if spec.strip()]
    save_json(out_dir / "config.json", {**vars(args), "parsed_variants": variants})

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_arg(args.dtype)
    if device.type == "cpu":
        dtype = torch.float32

    print(f"Loading tokenizer: {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_id}", flush=True)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    load_seconds = time.perf_counter() - start

    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(dtype),
        "load_seconds": load_seconds,
        "paper_samples": args.paper_samples,
        "variants": {},
    }
    for variant in variants:
        print(f"Evaluating {variant['name']}", flush=True)
        results = evaluate_paper_tasks(
            model,
            tokenizer,
            device,
            max_samples_per_task=args.paper_samples,
            include_squad=not args.skip_squad,
            prompt_style=variant["prompt_style"],
            prompt_template=variant["prompt_template"],
            glue_shot_count=variant["glue_shot_count"],
            mmlu_shot_count=variant["mmlu_shot_count"],
        )
        summary["variants"][variant["name"]] = {
            "config": variant,
            "results": results,
        }
        save_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
