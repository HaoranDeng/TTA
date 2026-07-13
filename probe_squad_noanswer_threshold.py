#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pq_lut_lm.eval_utils import score_completions
from pq_lut_lm.paper_eval import _squad_score, format_prompt_for_style
from probe_squad_prompts import postprocess_variants, prompt_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-style", choices=["plain", "chat"], default="plain")
    parser.add_argument("--prompt-name", default="current_instruction")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def dtype_from_arg(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def best_threshold(rows: list[dict[str, Any]], score_name: str) -> dict[str, Any]:
    scores = sorted({float(row[score_name]) for row in rows})
    if not scores:
        return {"threshold": 0.0, "f1": 0.0}
    thresholds = [scores[0] - 1.0]
    thresholds.extend((scores[i] + scores[i + 1]) / 2.0 for i in range(len(scores) - 1))
    thresholds.append(scores[-1] + 1.0)

    best = {"threshold": thresholds[0], "f1": -1.0, "empty_count": 0}
    for threshold in thresholds:
        total = 0.0
        empty_count = 0
        for row in rows:
            pred = "" if row[score_name] >= threshold else row["prediction"]
            empty_count += int(pred == "")
            total += _squad_score(pred, {"text": row["answers"]})
        f1 = 100.0 * total / max(len(rows), 1)
        if f1 > best["f1"]:
            best = {"threshold": threshold, "f1": f1, "empty_count": empty_count}
    return best


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_arg(args.dtype)
    if device.type == "cpu":
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()

    rows = list(load_dataset("squad_v2", split="validation").select(range(args.samples)))
    predictions = []
    total_current = 0.0
    start = time.perf_counter()
    for row in rows:
        variants = prompt_variants(row)
        if args.prompt_name not in variants:
            raise ValueError(f"Unknown prompt name {args.prompt_name}; choices: {sorted(variants)}")
        prompt = format_prompt_for_style(tokenizer, variants[args.prompt_name], args.prompt_style)
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        generated = model.generate(
            **ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(generated[0, ids.input_ids.shape[1] :], skip_special_tokens=True).strip()
        pred = postprocess_variants(raw)["current"]
        answers = row["answers"].get("text", [])
        current_f1 = _squad_score(pred, row["answers"])
        total_current += current_f1

        no_answer_scores = score_completions(model, tokenizer, prompt, ["No Answer", " No Answer"], device)
        no_answer_logp = max(no_answer_scores)
        no_answer_avg_logp = no_answer_logp / 2.0
        if pred:
            answer_scores = score_completions(model, tokenizer, prompt, [pred, " " + pred], device)
            answer_logp = max(answer_scores)
            answer_len = max(1, len(tokenizer(pred, add_special_tokens=False).input_ids))
        else:
            answer_logp = -1e30
            answer_len = 1
        predictions.append(
            {
                "id": row.get("id"),
                "question": row.get("question", ""),
                "answers": answers,
                "raw_prediction": raw,
                "prediction": pred,
                "current_f1": current_f1,
                "no_answer_logp": no_answer_logp,
                "no_answer_avg_logp": no_answer_avg_logp,
                "answer_logp": answer_logp,
                "answer_avg_logp": answer_logp / answer_len,
                "margin_logp": no_answer_logp - answer_logp,
                "margin_avg_logp": no_answer_avg_logp - (answer_logp / answer_len),
            }
        )
        print(
            len(predictions),
            f"current_f1={current_f1:.2f}",
            f"no_answer_logp={no_answer_logp:.2f}",
            f"margin={predictions[-1]['margin_logp']:.2f}",
            flush=True,
        )

    summary = {
        "model_id": args.model_id,
        "samples": len(rows),
        "prompt_style": args.prompt_style,
        "prompt_name": args.prompt_name,
        "max_new_tokens": args.max_new_tokens,
        "current_f1": 100.0 * total_current / max(len(rows), 1),
        "no_answer_gold_count": sum(1 for row in predictions if not row["answers"]),
        "current_empty_prediction_count": sum(1 for row in predictions if row["prediction"] == ""),
        "threshold_oracles": {
            "no_answer_logp": best_threshold(predictions, "no_answer_logp"),
            "no_answer_avg_logp": best_threshold(predictions, "no_answer_avg_logp"),
            "margin_logp": best_threshold(predictions, "margin_logp"),
            "margin_avg_logp": best_threshold(predictions, "margin_avg_logp"),
        },
        "seconds": time.perf_counter() - start,
        "predictions": predictions,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "predictions"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
