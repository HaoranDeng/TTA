#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pq_lut_lm.paper_eval import _normalize_answer, _squad_score, format_prompt_for_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-style", choices=["plain", "chat"], default="plain")
    parser.add_argument("--max-new-tokens", default="8,16,24,32")
    return parser.parse_args()


def dtype_from_arg(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def prompt_variants(row: dict[str, Any]) -> dict[str, str]:
    context = row["context"]
    question = row["question"]
    title = row.get("title", "")
    return {
        "current_instruction": (
            "Answer the question from the context. If the answer is not in the context, answer No Answer.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        ),
        "plain_context_question": (
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        ),
        "short_span": (
            "Read the context and answer the question with the shortest exact answer span. "
            "If the answer is missing, answer No Answer.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        ),
        "lighteval_style": (
            f"Title: {title}\n\n"
            f"Background: {context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        ),
        "qa_only": (
            f"{context}\n\n"
            f"Q: {question}\n"
            "A:"
        ),
    }


def postprocess_variants(text: str) -> dict[str, str]:
    raw = text.strip()
    first_line = raw.splitlines()[0].strip() if raw else ""
    no_answer_empty = first_line
    if no_answer_empty.lower().startswith("no answer"):
        no_answer_empty = ""
    prefix_stripped = re.sub(
        r"^(?:the\s+answer\s+is|answer\s*:|answer\s+is|it\s+is)\s+",
        "",
        no_answer_empty,
        flags=re.IGNORECASE,
    ).strip()
    first_sentence = prefix_stripped
    for sep in [".", ";"]:
        if sep in first_sentence:
            first_sentence = first_sentence.split(sep, 1)[0].strip()
            break
    return {
        "raw": raw,
        "current": no_answer_empty,
        "first_line": first_line,
        "prefix_stripped": prefix_stripped,
        "first_sentence": first_sentence,
    }


def exact_match(prediction: str, answers: dict[str, Any]) -> float:
    golds = answers.get("text", []) or [""]
    normalized_prediction = _normalize_answer(prediction)
    return float(any(normalized_prediction == _normalize_answer(gold) for gold in golds))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_arg(args.dtype)
    if device.type == "cpu":
        dtype = torch.float32
    max_new_values = [int(x) for x in args.max_new_tokens.split(",") if x.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    rows = list(load_dataset("squad_v2", split="validation").select(range(args.samples)))
    out: dict[str, Any] = {
        "model_id": args.model_id,
        "samples": args.samples,
        "prompt_style": args.prompt_style,
        "results": {},
    }
    for max_new_tokens in max_new_values:
        for name in prompt_variants(rows[0]):
            post_totals = {
                key: {"f1": 0.0, "em": 0.0, "empty_prediction_count": 0}
                for key in ["raw", "current", "first_line", "prefix_stripped", "first_sentence"]
            }
            no_answer_gold_count = 0
            examples = []
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            for row in rows:
                prompt = prompt_variants(row)[name]
                prompt = format_prompt_for_style(tokenizer, prompt, args.prompt_style)
                ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
                generated = model.generate(
                    **ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                completion = tokenizer.decode(
                    generated[0, ids.input_ids.shape[1] :],
                    skip_special_tokens=True,
                ).strip()
                processed = postprocess_variants(completion)
                if not row["answers"].get("text", []):
                    no_answer_gold_count += 1
                post_scores: dict[str, dict[str, float]] = {}
                for post_name, pred in processed.items():
                    f1 = _squad_score(pred, row["answers"])
                    em = exact_match(pred, row["answers"])
                    post_totals[post_name]["f1"] += f1
                    post_totals[post_name]["em"] += em
                    post_totals[post_name]["empty_prediction_count"] += int(pred == "")
                    post_scores[post_name] = {"f1": f1, "em": em}
                examples.append(
                    {
                        "id": row.get("id"),
                        "question": row.get("question", ""),
                        "raw_prediction": completion,
                        "prediction": processed["current"],
                        "postprocessed": processed,
                        "scores": post_scores,
                        "answers": row["answers"].get("text", [])[:3],
                    }
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            key = f"{name}_max{max_new_tokens}"
            postprocess = {}
            for post_name, totals in post_totals.items():
                postprocess[post_name] = {
                    "f1": 100.0 * totals["f1"] / max(len(rows), 1),
                    "em": 100.0 * totals["em"] / max(len(rows), 1),
                    "empty_prediction_count": int(totals["empty_prediction_count"]),
                }
            out["results"][key] = {
                "f1": postprocess["current"]["f1"],
                "em": postprocess["current"]["em"],
                "no_answer_gold_count": no_answer_gold_count,
                "postprocess": postprocess,
                "seconds": time.perf_counter() - start,
                "examples": examples,
            }
            best_post = max(postprocess.items(), key=lambda item: item[1]["f1"])
            print(
                key,
                f"current={postprocess['current']['f1']:.2f}",
                f"best={best_post[0]}:{best_post[1]['f1']:.2f}",
                flush=True,
            )

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
