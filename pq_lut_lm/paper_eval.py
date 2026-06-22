from __future__ import annotations

import collections
import re
import string
import time
from itertools import islice
from typing import Any

import torch
from datasets import load_dataset

from .eval_utils import score_completions


GLUE_TASKS = ["mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]


def _take(ds: Any, max_samples: int) -> list[dict[str, Any]]:
    rows = list(ds)
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def load_glue_rows(task: str, max_samples: int) -> list[dict[str, Any]]:
    split = "validation_matched" if task == "mnli" else "validation"
    return _take(load_dataset("glue", task, split=split), max_samples)


def glue_prompt_and_labels(task: str, row: dict[str, Any]) -> tuple[str, list[str], int]:
    if task == "sst2":
        prompt = f"Sentence: {row['sentence']}\nSentiment:"
        return prompt, [" negative", " positive"], int(row["label"])
    if task == "mrpc":
        prompt = (
            f"Sentence 1: {row['sentence1']}\n"
            f"Sentence 2: {row['sentence2']}\n"
            "Are these two sentences semantically equivalent?"
        )
        return prompt, [" no", " yes"], int(row["label"])
    if task == "qqp":
        prompt = (
            f"Question 1: {row['question1']}\n"
            f"Question 2: {row['question2']}\n"
            "Are these duplicate questions?"
        )
        return prompt, [" no", " yes"], int(row["label"])
    if task == "mnli":
        prompt = (
            f"Premise: {row['premise']}\n"
            f"Hypothesis: {row['hypothesis']}\n"
            "Relationship:"
        )
        return prompt, [" entailment", " neutral", " contradiction"], int(row["label"])
    if task == "qnli":
        prompt = (
            f"Question: {row['question']}\n"
            f"Sentence: {row['sentence']}\n"
            "Does the sentence answer the question?"
        )
        return prompt, [" yes", " no"], int(row["label"])
    if task == "rte":
        prompt = (
            f"Premise: {row['sentence1']}\n"
            f"Hypothesis: {row['sentence2']}\n"
            "Relationship:"
        )
        return prompt, [" entailment", " not entailment"], int(row["label"])
    raise ValueError(f"Unsupported GLUE task: {task}")


@torch.no_grad()
def evaluate_glue_task(
    model: torch.nn.Module,
    tokenizer: Any,
    task: str,
    max_samples: int,
    device: torch.device,
) -> dict[str, Any]:
    rows = load_glue_rows(task, max_samples)
    correct = 0
    predictions = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    model.eval()
    for row in rows:
        prompt, labels, gold = glue_prompt_and_labels(task, row)
        scores = score_completions(model, tokenizer, prompt, labels, device)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
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


def load_mmlu_pro_rows(max_samples: int, split: str = "test") -> list[dict[str, Any]]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    return list(islice(ds, max_samples if max_samples > 0 else None))


@torch.no_grad()
def evaluate_mmlu_pro(
    model: torch.nn.Module,
    tokenizer: Any,
    max_samples: int,
    device: torch.device,
) -> dict[str, Any]:
    rows = load_mmlu_pro_rows(max_samples)
    correct = 0
    predictions = []
    letters = [chr(ord("A") + i) for i in range(10)]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    model.eval()
    for row in rows:
        options = row["options"]
        prompt = f"Question: {row['question']}\n"
        for i, option in enumerate(options):
            prompt += f"{letters[i]}. {option}\n"
        prompt += "Answer:"
        labels = [f" {letters[i]}" for i in range(len(options))]
        scores = score_completions(model, tokenizer, prompt, labels, device)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        gold = int(row["answer_index"])
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


def _normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def _f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    truth_tokens = _normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _squad_score(prediction: str, answers: dict[str, Any]) -> float:
    golds = answers.get("text", [])
    if not golds:
        golds = [""]
    return max(_f1(prediction, gold) for gold in golds)


@torch.no_grad()
def evaluate_squad_v2(
    model: torch.nn.Module,
    tokenizer: Any,
    max_samples: int,
    device: torch.device,
    max_new_tokens: int = 24,
) -> dict[str, Any]:
    rows = _take(load_dataset("squad_v2", split="validation"), max_samples)
    total_f1 = 0.0
    predictions = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    model.eval()
    for row in rows:
        prompt = (
            "Answer the question from the context. If the answer is not in the context, answer No Answer.\n"
            f"Context: {row['context']}\n"
            f"Question: {row['question']}\n"
            "Answer:"
        )
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0, ids.input_ids.shape[1] :], skip_special_tokens=True).strip()
        if completion.lower().startswith("no answer"):
            completion = ""
        f1 = _squad_score(completion, row["answers"])
        total_f1 += f1
        predictions.append({"prediction": completion, "f1": f1, "answers": row["answers"].get("text", [])[:3]})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "f1": 100.0 * total_f1 / max(len(rows), 1),
        "total": len(rows),
        "seconds": time.perf_counter() - start,
        "predictions": predictions,
    }


@torch.no_grad()
def evaluate_paper_tasks(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_samples_per_task: int,
    include_squad: bool = True,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for task in GLUE_TASKS:
        metric = evaluate_glue_task(model, tokenizer, task, max_samples_per_task, device)
        results[task] = {k: v for k, v in metric.items() if k != "predictions"}
    if include_squad:
        metric = evaluate_squad_v2(model, tokenizer, max_samples_per_task, device)
        results["squad_v2"] = {k: v for k, v in metric.items() if k != "predictions"}
    metric = evaluate_mmlu_pro(model, tokenizer, max_samples_per_task, device)
    results["mmlu_pro"] = {k: v for k, v in metric.items() if k != "predictions"}
    return results
