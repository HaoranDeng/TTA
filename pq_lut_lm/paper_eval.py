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


def format_prompt_for_style(tokenizer: Any, prompt: str, prompt_style: str) -> str:
    if prompt_style == "plain":
        return prompt
    if prompt_style != "chat":
        raise ValueError(f"Unsupported prompt style: {prompt_style}")
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def format_completion_for_style(completion: str, prompt_style: str) -> str:
    if prompt_style == "chat":
        return completion.strip()
    return completion


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


def make_glue_supervised_examples(task: str, max_samples: int) -> list[tuple[str, str]]:
    examples = []
    for row in load_glue_rows(task, max_samples):
        prompt, labels, gold = glue_prompt_and_labels(task, row)
        examples.append((prompt, labels[gold]))
    return examples


@torch.no_grad()
def evaluate_glue_task(
    model: torch.nn.Module,
    tokenizer: Any,
    task: str,
    max_samples: int,
    device: torch.device,
    prompt_style: str = "plain",
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
        prompt = format_prompt_for_style(tokenizer, prompt, prompt_style)
        labels = [format_completion_for_style(label, prompt_style) for label in labels]
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


def format_mmlu_pro_prompt(row: dict[str, Any]) -> tuple[str, list[str]]:
    letters = [chr(ord("A") + i) for i in range(10)]
    options = row["options"]
    prompt = f"Question: {row['question']}\n"
    for i, option in enumerate(options):
        prompt += f"{letters[i]}. {option}\n"
    prompt += "Answer:"
    labels = [f" {letters[i]}" for i in range(len(options))]
    return prompt, labels


def make_mmlu_pro_supervised_examples(max_samples: int) -> list[tuple[str, str]]:
    examples = []
    for row in load_mmlu_pro_rows(max_samples, split="validation"):
        prompt, labels = format_mmlu_pro_prompt(row)
        examples.append((prompt, labels[int(row["answer_index"])]))
    return examples


@torch.no_grad()
def evaluate_mmlu_pro(
    model: torch.nn.Module,
    tokenizer: Any,
    max_samples: int,
    device: torch.device,
    prompt_style: str = "plain",
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
        prompt, labels = format_mmlu_pro_prompt(row)
        prompt = format_prompt_for_style(tokenizer, prompt, prompt_style)
        labels = [format_completion_for_style(label, prompt_style) for label in labels]
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
    prompt_style: str = "plain",
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
        prompt = format_prompt_for_style(tokenizer, prompt, prompt_style)
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


def make_squad_supervised_examples(max_samples: int) -> list[tuple[str, str]]:
    examples = []
    for row in _take(load_dataset("squad_v2", split="train"), max_samples):
        answers = row["answers"].get("text", [])
        answer = answers[0] if answers else " No Answer"
        prompt = (
            "Answer the question from the context. If the answer is not in the context, answer No Answer.\n"
            f"Context: {row['context']}\n"
            f"Question: {row['question']}\n"
            "Answer:"
        )
        examples.append((prompt, " " + answer.strip()))
    return examples


def make_paper_supervised_batches(
    tokenizer: Any,
    max_samples_per_task: int,
    batch_size: int,
    max_length: int,
    include_squad: bool = True,
    prompt_style: str = "plain",
) -> list[dict[str, torch.Tensor]]:
    examples: list[tuple[str, str]] = []
    per_task = max(1, max_samples_per_task)
    for task in GLUE_TASKS:
        examples.extend(make_glue_supervised_examples(task, per_task))
    examples.extend(make_mmlu_pro_supervised_examples(per_task))
    if include_squad:
        examples.extend(make_squad_supervised_examples(per_task))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded = []
    for prompt, completion in examples:
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
        encoded.append((full_ids, labels))

    batches = []
    rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    for item in encoded:
        rows.append(item)
        if len(rows) == batch_size:
            batches.append(_collate_supervised(rows, pad_id))
            rows = []
    if rows:
        batches.append(_collate_supervised(rows, pad_id))
    return batches


def _collate_supervised(
    rows: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(ids.numel() for ids, _ in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, (ids, row_labels) in enumerate(rows):
        input_ids[i, : ids.numel()] = ids
        labels[i, : row_labels.numel()] = row_labels
        attention_mask[i, : ids.numel()] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def evaluate_paper_tasks(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_samples_per_task: int,
    include_squad: bool = True,
    prompt_style: str = "plain",
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for task in GLUE_TASKS:
        metric = evaluate_glue_task(model, tokenizer, task, max_samples_per_task, device, prompt_style=prompt_style)
        results[task] = {k: v for k, v in metric.items() if k != "predictions"}
    if include_squad:
        metric = evaluate_squad_v2(model, tokenizer, max_samples_per_task, device, prompt_style=prompt_style)
        results["squad_v2"] = {k: v for k, v in metric.items() if k != "predictions"}
    metric = evaluate_mmlu_pro(model, tokenizer, max_samples_per_task, device, prompt_style=prompt_style)
    results["mmlu_pro"] = {k: v for k, v in metric.items() if k != "predictions"}
    return results
