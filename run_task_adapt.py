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

from pq_lut_lm.paper_eval import make_paper_supervised_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--train-samples-per-task", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--prompt-style", choices=["plain", "chat"], default="plain")
    parser.add_argument("--prompt-template", choices=["simple", "instruction", "lm_eval"], default="lm_eval")
    parser.add_argument("--skip-squad", action="store_true")
    parser.add_argument("--skip-mmlu", action="store_true")
    parser.add_argument("--squad-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args()


def dtype_from_arg(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    print("Building supervised adaptation batches", flush=True)
    batches = make_paper_supervised_batches(
        tokenizer,
        max_samples_per_task=args.train_samples_per_task,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_squad=not args.skip_squad,
        include_mmlu=not args.skip_mmlu,
        squad_repeat=args.squad_repeat,
        prompt_style=args.prompt_style,
        prompt_template=args.prompt_template,
    )
    print(f"Prepared {len(batches)} batches", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = time.perf_counter()
    losses: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)

    step = 0
    update = 0
    micro_since_update = 0
    skipped = 0
    while update < args.steps:
        order = list(range(len(batches)))
        random.shuffle(order)
        for idx in order:
            batch = {k: v.to(device) for k, v in batches[idx].items()}
            out = model(**batch)
            if not torch.isfinite(out.loss):
                skipped += 1
                if skipped <= 10 or skipped % 100 == 0:
                    print(f"Skipping non-finite loss batch: skipped={skipped}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                micro_since_update = 0
                continue
            raw_loss = float(out.loss.detach().item())
            loss = out.loss / args.grad_accum_steps
            loss.backward()
            step += 1
            micro_since_update += 1
            if micro_since_update == args.grad_accum_steps:
                update += 1
                micro_since_update = 0
                if args.warmup_steps > 0 and update <= args.warmup_steps:
                    scale = update / args.warmup_steps
                    for group in optimizer.param_groups:
                        group["lr"] = args.lr * scale
                else:
                    for group in optimizer.param_groups:
                        group["lr"] = args.lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                record = {
                    "update": update,
                    "loss": raw_loss,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "skipped": skipped,
                    "seconds": time.perf_counter() - start,
                }
                losses.append(record)
                if update == 1 or update % 10 == 0:
                    print(json.dumps(record, sort_keys=True), flush=True)
                    save_json(out_dir / "train_log.json", losses)
                if args.save_every > 0 and update % args.save_every == 0:
                    ckpt_dir = out_dir / f"checkpoint_step{update}"
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                if update >= args.steps:
                    break

    save_json(out_dir / "train_log.json", losses)
    final_dir = out_dir / "checkpoint_final"
    print(f"Saving final checkpoint to {final_dir}", flush=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
