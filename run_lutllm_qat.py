#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pq_lut_lm.activation_quant import (
    STEActivationQuantLinear,
    convert_ste_act_quant_to_lut,
    replace_with_ste_act_quant,
    set_ste_reconstruction_loss_enabled,
    ste_reconstruction_loss,
    trainable_act_center_parameters,
)
from pq_lut_lm.eval_utils import load_wikitext_texts, make_lm_batches
from pq_lut_lm.eval_utils import evaluate_ppl
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
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--train-source", choices=["wikitext", "paper", "lutllm_paper"], default="wikitext")
    parser.add_argument("--train-tokens", type=int, default=8192)
    parser.add_argument("--task-train-samples", type=int, default=32)
    parser.add_argument("--fineweb-config", default="sample-10BT")
    parser.add_argument("--fineweb-sequences", type=int, default=256)
    parser.add_argument("--wikiqa-samples", type=int, default=512)
    parser.add_argument("--calib-tokens", type=int, default=1024)
    parser.add_argument("--calib-batches", type=int, default=4)
    parser.add_argument("--calib-vectors-per-layer", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-dense-linears", action="store_true")
    parser.add_argument("--dense-lr", type=float, default=1e-5)
    parser.add_argument("--reconstruction-loss-ratio", type=float, default=0.0)
    parser.add_argument("--paper-samples", type=int, default=16)
    parser.add_argument("--eval-ppl", action="store_true")
    parser.add_argument("--ppl-tokens", type=int, default=4096)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument("--prompt-style", choices=["plain", "chat"], default="plain")
    parser.add_argument("--prompt-template", choices=["simple", "instruction", "lm_eval"], default="simple")
    parser.add_argument("--glue-shot-count", type=int, default=0)
    parser.add_argument("--mmlu-shot-count", type=int, default=0)
    parser.add_argument("--skip-squad", action="store_true")
    parser.add_argument("--train-include-squad", action="store_true")
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
    parser.add_argument("--lut-quant-bits", type=int, default=8)
    parser.add_argument("--weight-code-reassign-iters", type=int, default=0)
    parser.add_argument("--weight-center-refine-iters", type=int, default=0)
    parser.add_argument("--weight-center-refine-reg", type=float, default=1e-4)
    parser.add_argument("--weight-center-refine-blend", type=float, default=1.0)
    parser.add_argument("--act-train-mode", choices=["hard", "soft", "soft_hard"], default="hard")
    parser.add_argument("--act-softmax-temperature", type=float, default=1.0)
    parser.add_argument("--act-ste-input-scale", type=float, default=1.0)
    parser.add_argument("--lut-storage", choices=["expanded", "compact"], default="expanded")
    parser.add_argument("--output-correction", choices=["none", "bias", "affine"], default="none")
    parser.add_argument("--eval-baseline", action="store_true")
    parser.add_argument("--eval-act-quant", action="store_true")
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


def public_eval(result: dict[str, Any]) -> dict[str, Any]:
    return result


def strip_labels(batches: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{k: v for k, v in batch.items() if k != "labels"} for batch in batches]


def shuffled_batches(batches: list[dict[str, torch.Tensor]], seed: int) -> list[dict[str, torch.Tensor]]:
    out = list(batches)
    random.Random(seed).shuffle(out)
    return out


def collate_supervised_rows(
    rows: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(ids.numel() for ids, _ in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, (ids, row_labels) in enumerate(rows):
        input_ids[row, : ids.numel()] = ids
        labels[row, : row_labels.numel()] = row_labels
        attention_mask[row, : ids.numel()] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def make_lutllm_paper_train_batches(
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    fineweb_config: str,
    fineweb_sequences: int,
    wikiqa_samples: int,
    prompt_style: str,
) -> list[dict[str, torch.Tensor]]:
    """Approximate the paper's FineWeb pretrain + WikiQA finetune data mix."""
    from pq_lut_lm.paper_eval import format_completion_for_style, format_prompt_for_style

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    rows: list[tuple[torch.Tensor, torch.Tensor]] = []

    fineweb = load_dataset("HuggingFaceFW/fineweb", fineweb_config, split="train", streaming=True)
    token_buffer: list[int] = []
    for record in fineweb:
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        token_buffer.extend(tokenizer(text, add_special_tokens=False).input_ids)
        token_buffer.append(int(eos_id))
        while len(token_buffer) >= max_length and len(rows) < fineweb_sequences:
            ids = torch.tensor(token_buffer[:max_length], dtype=torch.long)
            labels = ids.clone()
            rows.append((ids, labels))
            token_buffer = token_buffer[max_length:]
        if len(rows) >= fineweb_sequences:
            break

    wikiqa = load_dataset("wiki_qa", split="train")
    positives = (row for row in wikiqa if int(row.get("label", 0)) == 1)
    for row in islice(positives, max(0, wikiqa_samples)):
        prompt = f"Answer the question.\nQuestion: {row['question']}\nAnswer:"
        completion = " " + str(row["answer"]).strip()
        prompt = format_prompt_for_style(tokenizer, prompt, prompt_style)
        completion = format_completion_for_style(completion, prompt_style)
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        full_ids = tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if full_ids.numel() > max_length:
            full_ids = full_ids[-max_length:]
            prompt_len = min(prompt_ids.numel(), full_ids.numel() - 1)
        else:
            prompt_len = prompt_ids.numel()
        labels = full_ids.clone()
        labels[:prompt_len] = -100
        if (labels != -100).any():
            rows.append((full_ids, labels))

    batches = []
    current: list[tuple[torch.Tensor, torch.Tensor]] = []
    for row in rows:
        current.append(row)
        if len(current) == batch_size:
            batches.append(collate_supervised_rows(current, int(pad_id)))
            current = []
    if current:
        batches.append(collate_supervised_rows(current, int(pad_id)))
    return batches


def dense_linear_parameters_under_ste(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, STEActivationQuantLinear):
            for param in module.linear.parameters():
                param.requires_grad_(True)
                params.append(param)
    return params


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

    texts = load_wikitext_texts("train")
    train_include_squad = args.train_include_squad or not args.skip_squad
    if args.train_source == "paper":
        train_batches = make_paper_supervised_batches(
            tokenizer,
            max_samples_per_task=args.task_train_samples,
            batch_size=1,
            max_length=args.seq_len,
            include_squad=train_include_squad,
            prompt_style=args.prompt_style,
            prompt_template=args.prompt_template,
        )
        train_batches = shuffled_batches(train_batches, args.seed)
        calib_batches = strip_labels(train_batches[: args.calib_batches])
    elif args.train_source == "lutllm_paper":
        train_batches = make_lutllm_paper_train_batches(
            tokenizer,
            max_length=args.seq_len,
            batch_size=1,
            fineweb_config=args.fineweb_config,
            fineweb_sequences=args.fineweb_sequences,
            wikiqa_samples=args.wikiqa_samples,
            prompt_style=args.prompt_style,
        )
        train_batches = shuffled_batches(train_batches, args.seed)
        calib_batches = strip_labels(train_batches[: args.calib_batches])
    else:
        train_batches = make_lm_batches(tokenizer, texts, args.seq_len, args.train_tokens, batch_size=1)
        calib_batches = make_lm_batches(tokenizer, texts, args.seq_len, args.calib_tokens, batch_size=1)[: args.calib_batches]
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
        summary["fp16_baseline"] = public_eval(
            evaluate_paper_tasks(
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
        )
        save_json(out_dir / "summary.json", summary)
    if args.eval_ppl and ppl_batches is not None:
        print("Evaluating FP16 baseline perplexity", flush=True)
        summary.setdefault("fp16_baseline", {})["wikitext_ppl"] = evaluate_ppl(model, ppl_batches, device)
        save_json(out_dir / "summary.json", summary)

    print("Freezing dense model weights", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)

    config = PQConfig(
        method="lutllm",
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
        weight_code_reassign_iters=args.weight_code_reassign_iters,
        weight_center_refine_iters=args.weight_center_refine_iters,
        weight_center_refine_reg=args.weight_center_refine_reg,
        weight_center_refine_blend=args.weight_center_refine_blend,
        act_train_mode=args.act_train_mode,
        act_softmax_temperature=args.act_softmax_temperature,
        act_ste_input_scale=args.act_ste_input_scale,
        output_correction=args.output_correction,
        seed=args.seed,
    )

    print("Initializing STE activation quantizers", flush=True)
    act_report = replace_with_ste_act_quant(
        model,
        calib_batches,
        config,
        target_regex=args.target_regex,
        include_lm_head=args.include_lm_head,
        max_linears=args.max_linears,
        max_vectors_per_layer=args.calib_vectors_per_layer,
        device=device,
    )
    save_json(out_dir / "act_quant_hardware_stats.json", {
        "modules": act_report.module_stats,
        "aggregate": act_report.aggregate,
        "calibration_seconds": act_report.calibration_seconds,
        "initialization_seconds": act_report.initialization_seconds,
    })

    center_params = trainable_act_center_parameters(model)
    dense_params = dense_linear_parameters_under_ste(model) if args.train_dense_linears else []
    center_param_count = sum(p.numel() for p in center_params)
    dense_param_count = sum(p.numel() for p in dense_params)
    message = f"Training {center_param_count:,} activation-center parameters"
    if dense_params:
        message += f" and {dense_param_count:,} dense linear parameters"
    print(message, flush=True)
    param_groups: list[dict[str, Any]] = [{"params": center_params, "lr": args.lr}]
    if dense_params:
        param_groups.append({"params": dense_params, "lr": args.dense_lr})
    opt = torch.optim.AdamW(param_groups)
    train_losses = []
    task_losses = []
    reconstruction_losses = []
    use_reconstruction_loss = args.reconstruction_loss_ratio > 0.0
    set_ste_reconstruction_loss_enabled(model, use_reconstruction_loss)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_start = time.perf_counter()
    model.train()
    for step in range(args.train_steps):
        batch = train_batches[step % len(train_batches)]
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.get("labels", batch["input_ids"].clone())
        opt.zero_grad(set_to_none=True)
        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        out = model(**model_inputs, labels=labels)
        task_loss = out.loss
        recon_loss = ste_reconstruction_loss(model) if use_reconstruction_loss else None
        loss = task_loss
        if recon_loss is not None:
            loss = loss + args.reconstruction_loss_ratio * recon_loss
        loss.backward()
        opt.step()
        value = float(loss.detach().cpu().item())
        task_value = float(task_loss.detach().cpu().item())
        recon_value = float(recon_loss.detach().cpu().item()) if recon_loss is not None else 0.0
        train_losses.append(value)
        task_losses.append(task_value)
        reconstruction_losses.append(recon_value)
        print(
            f"step {step + 1}/{args.train_steps} loss={value:.4f} "
            f"task={task_value:.4f} recon={recon_value:.4f}",
            flush=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary["act_qat_training"] = {
        "steps": args.train_steps,
        "seconds": time.perf_counter() - train_start,
        "losses": train_losses,
        "task_losses": task_losses,
        "reconstruction_losses": reconstruction_losses,
        "reconstruction_loss_ratio": args.reconstruction_loss_ratio,
        "train_dense_linears": args.train_dense_linears,
        "center_param_count": center_param_count,
        "dense_linear_param_count": dense_param_count,
    }
    model.eval()
    save_json(out_dir / "summary.json", summary)

    if args.eval_act_quant:
        print("Evaluating +Act. Quant. on paper tasks", flush=True)
        summary["act_quant"] = public_eval(
            evaluate_paper_tasks(
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
        )
        save_json(out_dir / "summary.json", summary)
    if args.eval_ppl and ppl_batches is not None:
        print("Evaluating +Act. Quant. perplexity", flush=True)
        summary.setdefault("act_quant", {})["wikitext_ppl"] = evaluate_ppl(model, ppl_batches, device)
        save_json(out_dir / "summary.json", summary)

    if args.skip_final_lut:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    print("Converting trained activation quantizers to activation-weight LUT modules", flush=True)
    lut_report = convert_ste_act_quant_to_lut(
        model,
        calib_batches,
        config,
        target_regex=args.target_regex,
        include_lm_head=args.include_lm_head,
        max_linears=args.max_linears,
        max_vectors_per_layer=args.calib_vectors_per_layer,
        device=device,
    )
    save_json(out_dir / "final_lut_hardware_stats.json", {
        "modules": lut_report.module_stats,
        "aggregate": lut_report.aggregate,
        "calibration_seconds": lut_report.calibration_seconds,
        "quantization_seconds": lut_report.quantization_seconds,
    })
    summary["final_lut"] = {
        "hardware_aggregate": lut_report.aggregate,
        "calibration_seconds": lut_report.calibration_seconds,
        "quantization_seconds": lut_report.quantization_seconds,
    }
    save_json(out_dir / "summary.json", summary)

    if args.eval_final_lut:
        print("Evaluating +Weight Quant. final LUT on paper tasks", flush=True)
        summary["final_lut"]["paper_eval"] = public_eval(
            evaluate_paper_tasks(
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
        )
        save_json(out_dir / "summary.json", summary)
    if args.eval_ppl and ppl_batches is not None:
        print("Evaluating +Weight Quant. final LUT perplexity", flush=True)
        summary["final_lut"]["wikitext_ppl"] = evaluate_ppl(model, ppl_batches, device)
        save_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
