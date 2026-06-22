from __future__ import annotations

import math
import time
from itertools import islice
from typing import Any, Iterable

import torch
from datasets import load_dataset


def load_wikitext_texts(split: str = "test") -> list[str]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return [row["text"] for row in ds if row["text"].strip()]


def make_lm_batches(
    tokenizer: Any,
    texts: list[str],
    seq_len: int,
    max_tokens: int,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    text = "\n\n".join(texts)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    ids = ids[: max_tokens + 1]
    usable = (ids.numel() - 1) // seq_len * seq_len
    ids = ids[: usable + 1]
    batches = []
    rows = []
    for start in range(0, usable, seq_len):
        rows.append(ids[start : start + seq_len])
        if len(rows) == batch_size:
            input_ids = torch.stack(rows, dim=0)
            batches.append({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})
            rows = []
    if rows:
        input_ids = torch.stack(rows, dim=0)
        batches.append({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})
    return batches


@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        out = model(**batch, labels=labels)
        tokens = int((labels[:, 1:] != -100).sum().item())
        total_loss += float(out.loss.item()) * tokens
        total_tokens += tokens
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    nll = total_loss / max(total_tokens, 1)
    return {
        "nll": nll,
        "ppl": math.exp(min(nll, 20.0)),
        "tokens": total_tokens,
        "seconds": time.perf_counter() - start,
    }


def load_mmlu_rows(max_samples: int, subject: str = "all", split: str = "test") -> list[dict[str, Any]]:
    ds = load_dataset("cais/mmlu", subject, split=split)
    return list(islice(ds, max_samples))


def format_mmlu_prompt(row: dict[str, Any]) -> str:
    choices = row["choices"]
    return (
        f"Question: {row['question']}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer:"
    )


def _answer_index(row: dict[str, Any]) -> int:
    answer = row["answer"]
    if isinstance(answer, int):
        return answer
    return {"A": 0, "B": 1, "C": 2, "D": 3}[str(answer).strip().upper()]


@torch.no_grad()
def score_completions(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    completions: list[str],
    device: torch.device,
) -> list[float]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    encoded = [
        tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        for completion in completions
    ]
    max_len = max(ids.numel() for ids in encoded)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(encoded), max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(encoded):
        input_ids[row, : ids.numel()] = ids
        attention_mask[row, : ids.numel()] = 1

    logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = []
    prompt_len = prompt_ids.numel()
    for row, ids in enumerate(encoded):
        completion_ids = ids[prompt_len:]
        if completion_ids.numel() == 0:
            scores.append(-1e30)
            continue
        score = 0.0
        for i, token_id in enumerate(completion_ids):
            score += float(log_probs[row, prompt_len + i - 1, int(token_id)].item())
        scores.append(score)
    return scores


@torch.no_grad()
def evaluate_mmlu_zero_shot(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    labels = [" A", " B", " C", " D"]
    correct = 0
    predictions = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    model.eval()
    for row in rows:
        prompt = format_mmlu_prompt(row)
        scores = score_completions(model, tokenizer, prompt, labels, device)
        pred = int(max(range(4), key=lambda i: scores[i]))
        gold = _answer_index(row)
        correct += int(pred == gold)
        predictions.append({"pred": pred, "gold": gold, "scores": scores})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "accuracy": correct / max(len(rows), 1),
        "correct": correct,
        "total": len(rows),
        "seconds": time.perf_counter() - start,
        "predictions": predictions,
    }
